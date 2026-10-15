// pgrun: the behaviour-lock suite's host. It plays the hypervisor to the
// library's C API (reims_vgpu.h) and the guest's memory, and writes what the
// machine does. docs/testing.md is its specification.
//
//   pgrun SCENARIO.jsonl OUT.jsonl --setup blocks|map|isolate [--fixtures DIR]
//         [--phase K --handoff DIR] [--build DIR]
//
// --build names the build under test, the directory the runner was built
// against: the binding record reports the library as "<build>" when it lies
// there (or is linked into the runner), and by its path otherwise.
//
// A scenario split by {"op": "phase"} markers is a migration: each phase is
// its own process (a VM restored on another host, or the same host after
// a restart), and --phase K runs the ops of phase K only. The phases hand
// guest RAM and the suspend blobs on through the files in DIR (op
// handoff_save, config handoff_load), which is what a restore carries.
//
// It drives the library only as a hypervisor does (decision 1): the C API's
// header, called the way QEMU calls it:
//   - MMIO runs on a pool thread while the calling thread waits;
//   - display callbacks arrive on the library's display thread, while the
//     scenario runs on a thread of its own;
//   - guest RAM comes from mach_vm_allocate; the task callbacks reserve task
//     memory and mach_vm_remap guest RAM into it;
//   - a frame is encoded on a pool thread into a frame target, then
//     committed, waited for and read back.
// It includes nothing of the library's code but its header, and knows no
// opcode and no layout: the scenario (tests/suite/pgtest) holds all of that.
//
// Output: JSON Lines, one observation a line, keys sorted, flushed as written,
// so a crash leaves everything observed before it. Every observation carries
// "i", the index of the scenario op in progress. Fields named t_* (timing)
// and d_* (diagnostics: signals, messages, pointers) are never compared.

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <reims_vgpu.h>

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <os/lock.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <unistd.h>

// ---- output ----

static int g_out_fd = -1;
// A host death and the log (on_signal). A doorbell's death on a FIFO's
// thread races with the doorbell's own record, which the scenario thread
// writes once the call returns; so what the log holds would be decided by
// which thread the kernel ran first. Instead:
// - g_recordPending is set from the moment a call into the library that
//   the scenario records is entered (an MMIO access in mmio_do; an api,
//   encode, encode_commit or trace_fire op in scenario_main) until its
//   record is written (emit) or the op ends, and g_callThread names the
//   pool thread making the call, if not the scenario thread. A death on any
//   other thread waits for that record, as long as the call returns: a
//   call the death blocks never does, and after 2 s the record is given up,
//   as a death on the calling thread itself gives it up at once.
// - Then g_dying stops every later record (emit), and the death waits for a
//   record being written to finish, so none lands after the decision.
// Whether a record is in the log is thus decided by the program: whether the
// call was entered, and whether it returns, before the host dies.
static _Atomic int g_recordPending;
static _Atomic(pthread_t) g_callThread;
static _Atomic int g_dying, g_writers;
static pthread_t g_scenarioThread;
static _Atomic int64_t g_op = -1;
static uint64_t g_t0;

static uint64_t now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

static NSString *hex64(uint64_t v) { return [NSString stringWithFormat:@"0x%llx", v]; }

static void emit(NSDictionary *obj) {
    NSMutableDictionary *d = [obj mutableCopy];
    if (!d[@"i"]) d[@"i"] = @(atomic_load(&g_op));
    NSData *json = [NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];
    NSMutableData *line = [json mutableCopy];
    [line appendBytes:"\n" length:1];
    atomic_fetch_add(&g_writers, 1);
    if (atomic_load(&g_dying)) {
        atomic_fetch_sub(&g_writers, 1);
        for (;;) pause();  // the host is dying: nothing more is recorded
    }
    write(g_out_fd, line.bytes, line.length);  // one write(2) a line
    atomic_store(&g_recordPending, 0);
    atomic_fetch_sub(&g_writers, 1);
}

static void fatal_scenario(NSString *why) {
    emit(@{@"ev" : @"scenario_error", @"why" : why});
    fprintf(stderr, "pgrun: scenario error: %s\n", why.UTF8String);
    _exit(65);
}

// ---- threads: the API's rules (reims_vgpu.h, each callback type) ----
//
// t_runner marks a thread of the runner's: the main thread, the scenario
// thread, and a pool thread while it runs one of the runner's jobs (an MMIO
// access, an encode). t_inCall marks a thread inside a call into the
// library. Each callback checks the rule its type states, and a broken rule
// is a scenario_error that names it:
// - a display callback runs on none of the runner's threads, and never
//   while another display callback runs;
// - a task, memory, trace or raise callback runs on a thread that is not
//   the runner's, or on one of the runner's inside a call into the library
//   (a call that may call back);
// - an IOSurface raise runs on none of the runner's threads, and never
//   while another runs.

static __thread int t_runner, t_inCall;
static _Atomic int g_displayCallbacks, g_iosfcCallbacks;

// A call into the library, which may call back on this thread.
#define LIB(...)                   \
    ({                             \
        t_inCall++;                \
        __auto_type r_ = (__VA_ARGS__); \
        t_inCall--;                \
        r_;                        \
    })
#define LIBV(...)     \
    do {              \
        t_inCall++;   \
        __VA_ARGS__;  \
        t_inCall--;   \
    } while (0)

static void device_callback_rule(const char *name) {
    if (t_runner && !t_inCall)
        fatal_scenario([NSString stringWithFormat:@"thread rule: %s ran on a runner thread outside a library call", name]);
}

static void display_callback_enter(const char *name) {
    if (t_runner) fatal_scenario([NSString stringWithFormat:@"thread rule: display callback %s ran on a runner thread", name]);
    if (atomic_fetch_add(&g_displayCallbacks, 1))
        fatal_scenario([NSString stringWithFormat:@"thread rule: display callback %s overlapped another", name]);
}

static void display_callback_exit(void) { atomic_fetch_sub(&g_displayCallbacks, 1); }

// ---- the API's statuses ----

// A status's name, as the header spells it; a status this runner does not
// know is its number.
static NSString *status_name(RvgStatus s) {
    static const char *const names[] = {
        [RVG_OK] = "RVG_OK",
        [RVG_ERR_INVALID_ARGUMENT] = "RVG_ERR_INVALID_ARGUMENT",
        [RVG_ERR_UNSUPPORTED] = "RVG_ERR_UNSUPPORTED",
        [RVG_ERR_DISPLAY_PORT_COUNT] = "RVG_ERR_DISPLAY_PORT_COUNT",
        [RVG_ERR_NO_METAL_DEVICE] = "RVG_ERR_NO_METAL_DEVICE",
        [RVG_ERR_MEMORY_CONFLICT] = "RVG_ERR_MEMORY_CONFLICT",
        [RVG_ERR_MEMORY_MAP] = "RVG_ERR_MEMORY_MAP",
        [RVG_ERR_ISOLATION_WITHOUT_MAP] = "RVG_ERR_ISOLATION_WITHOUT_MAP",
        [RVG_ERR_METAL] = "RVG_ERR_METAL",
        [RVG_ERR_PORT_OUT_OF_RANGE] = "RVG_ERR_PORT_OUT_OF_RANGE",
        [RVG_ERR_PORT_IN_USE] = "RVG_ERR_PORT_IN_USE",
        [RVG_ERR_SERIAL_IN_USE] = "RVG_ERR_SERIAL_IN_USE",
        [RVG_ERR_NO_MEMORY_MAP] = "RVG_ERR_NO_MEMORY_MAP",
        [RVG_ERR_SUSPEND] = "RVG_ERR_SUSPEND",
        [RVG_ERR_RESUME] = "RVG_ERR_RESUME",
        [RVG_ERR_GPU] = "RVG_ERR_GPU",
        [RVG_ERR_FRAME_STATE] = "RVG_ERR_FRAME_STATE",
    };
    if (s < sizeof(names) / sizeof(names[0]) && names[s]) return @(names[s]);
    return [NSString stringWithFormat:@"%u", s];
}

static NSString *status_message(RvgStatus s) { return @(rvg_status_message(s)); }

// ---- values ----

static uint64_t num(id v) {
    if (!v || v == [NSNull null]) return 0;
    if ([v isKindOfClass:[NSString class]]) return strtoull([v UTF8String], NULL, 0);
    return [v unsignedLongLongValue];
}

static NSData *from_hex(NSString *s) {
    NSMutableData *d = [NSMutableData dataWithCapacity:s.length / 2];
    const char *c = s.UTF8String;
    for (size_t i = 0; c[i] && c[i + 1]; i += 2) {
        char b[3] = {c[i], c[i + 1], 0};
        uint8_t v = (uint8_t)strtoul(b, NULL, 16);
        [d appendBytes:&v length:1];
    }
    return d;
}

static NSString *to_hex(const uint8_t *p, size_t n) {
    NSMutableString *s = [NSMutableString stringWithCapacity:n * 2];
    for (size_t i = 0; i < n; i++) [s appendFormat:@"%02x", p[i]];
    return s;
}

static NSString *sha256(const uint8_t *p, size_t n) {
    uint8_t md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(p, (CC_LONG)n, md);
    return to_hex(md, sizeof(md));
}

// A double: a number, or a string strtod reads ("nan", "inf", "-inf"), for
// the values JSON cannot spell.
static double dbl(id v) {
    if ([v isKindOfClass:[NSString class]]) return strtod([v UTF8String], NULL);
    return [v doubleValue];
}

// ---- guest memory: one or more RAM ranges, each from mach_vm_allocate ----

typedef struct {
    uint64_t gpa, size;
    uint8_t *host;    // what the device sees
    uint8_t *shadow;  // what the scenario wrote; a diff is the device's
} ram_range;

static ram_range *g_ram;  // as many as the scenario lists; the memory map has no limit
static int g_nram;

static ram_range *ram_for(uint64_t gpa, uint64_t len) {
    for (int i = 0; i < g_nram; i++)
        if (gpa >= g_ram[i].gpa && gpa + len <= g_ram[i].gpa + g_ram[i].size && gpa + len >= gpa) return &g_ram[i];
    return NULL;
}

static uint8_t *gpa_ptr(uint64_t gpa, uint64_t len) {
    ram_range *r = ram_for(gpa, len);
    if (!r) fatal_scenario([NSString stringWithFormat:@"gpa 0x%llx+0x%llx is outside guest RAM", gpa, len]);
    return r->host + (gpa - r->gpa);
}

// Every RAM range, as the memory map describes guest RAM to the library.
// Freed by the caller.
static RvgMemoryRange *ram_map(void) {
    RvgMemoryRange *m = calloc(g_nram ? g_nram : 1, sizeof(*m));
    for (int i = 0; i < g_nram; i++) m[i] = (RvgMemoryRange){g_ram[i].gpa, g_ram[i].size, g_ram[i].host};
    return m;
}

static NSString *g_fixtures;
static NSString *g_handoff;  // --handoff DIR: where phases hand RAM and blobs on
static NSString *g_build;    // --build DIR: the build under test, resolved

static uint32_t mmio(BOOL iosfc, BOOL write, uint64_t off, uint32_t val);

// ---- events: interrupts, callbacks to the host ----
//
// Recorded from whatever thread the library calls on, never answered by
// calling back into the device. Each gets a sequence number, so a checkpoint
// can report the order of events the implementation orders (decision 8), and
// interrupts snapshot the guest's stamp slots as the interrupt found them.

static os_unfair_lock g_evLock = OS_UNFAIR_LOCK_INIT;
static pthread_mutex_t g_waitMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_waitCond = PTHREAD_COND_INITIALIZER;
static NSMutableArray<NSMutableDictionary *> *g_events;  // under g_evLock
static NSUInteger g_evReported;
static uint64_t g_seq;
static _Atomic uint32_t g_nirq, g_niosfcIrq, g_ncb, g_nblk;
static uint64_t g_stampGPA;
static uint32_t g_stampSlots = 2;

static void kick_waiters(void) {
    pthread_mutex_lock(&g_waitMutex);
    pthread_cond_broadcast(&g_waitCond);
    pthread_mutex_unlock(&g_waitMutex);
}

static void record_event(NSString *kind, NSDictionary *fields, _Atomic uint32_t *counter) {
    NSMutableDictionary *e = [fields mutableCopy] ?: [NSMutableDictionary dictionary];
    os_unfair_lock_lock(&g_evLock);
    e[@"kind"] = kind;
    e[@"seq"] = @(g_seq++);
    e[@"d_i"] = @(atomic_load(&g_op));
    e[@"t_us"] = @((now_ns() - g_t0) / 1000);
    [g_events addObject:e];
    if (counter) atomic_fetch_add(counter, 1);
    os_unfair_lock_unlock(&g_evLock);
    kick_waiters();
}

static void record_irq(uint32_t vector) {
    NSMutableArray *stamps = [NSMutableArray array];
    if (g_stampGPA) {
        volatile uint32_t *s = (volatile uint32_t *)gpa_ptr(g_stampGPA, 4 * g_stampSlots);
        for (uint32_t i = 0; i < g_stampSlots; i++) [stamps addObject:hex64(s[i])];
    }
    record_event(@"irq", @{@"vector" : @(vector), @"stamps" : stamps}, &g_nirq);
}

// Writes what arrived since the last report. Interrupts are reported as a
// count; their stamp snapshots and the order across kinds only on request,
// where the scenario says the implementation orders them.
static uint64_t event_count(void) {
    os_unfair_lock_lock(&g_evLock);
    uint64_t n = g_seq;
    os_unfair_lock_unlock(&g_evLock);
    return n;
}

// Settles a checkpoint: waits until no event has arrived for quiet_ms, within
// a deadline. A checkpoint is only taken after a wait on something the guest
// sees; this lets the interrupts and callbacks that follow it land, so that
// their counts are what the implementation produced, not how fast it ran.
static void quiesce(uint64_t quiet_ms, uint64_t timeout_ms) {
    uint64_t deadline = now_ns() + timeout_ms * 1000000ull;
    uint64_t last = event_count(), since = now_ns();
    while (now_ns() < deadline) {
        usleep(2000);
        uint64_t n = event_count();
        if (n != last) {
            last = n;
            since = now_ns();
        } else if (now_ns() - since >= quiet_ms * 1000000ull) {
            break;
        }
    }
}

static void op_events(NSDictionary *op) {
    if (![op[@"no_quiesce"] boolValue]) quiesce(op[@"quiet_ms"] ? num(op[@"quiet_ms"]) : 40, 3000);
    os_unfair_lock_lock(&g_evLock);
    NSArray *evs = [g_events subarrayWithRange:NSMakeRange(g_evReported, g_events.count - g_evReported)];
    g_evReported = g_events.count;
    os_unfair_lock_unlock(&g_evLock);
    NSUInteger irqs = 0, iosfc = 0;
    BOOL otherVector = NO;
    NSMutableArray *vectors = [NSMutableArray array];
    NSMutableArray *cbs = [NSMutableArray array], *blks = [NSMutableArray array];
    NSMutableArray *order = [NSMutableArray array], *irqStamps = [NSMutableArray array];
    NSArray *collapse = op[@"collapse"] ?: @[];
    // Callbacks to the host from FIFOs running at once arrive in an order
    // only timing decides; sort_blocks reports them sorted, and leaves them
    // out of order.
    BOOL sortBlocks = [op[@"sort_blocks"] boolValue];
    // Tens of thousands of callbacks (the range tracker's limits) are
    // reported as counts per kind and a digest of the records, not listed.
    BOOL digestBlocks = [op[@"blocks_digest"] boolValue];
    NSDictionary *prevCb = nil;
    // Interrupts whose count only timing decides can be collapsed ("irq" in
    // collapse): the pending-stamp pump publishes every ready stamp at the
    // head with one interrupt, so a batch behind an asynchronous command
    // (an exec's completion) raises one to n of them by timing. A run of
    // consecutive interrupts then counts once, and its snapshot (irq_stamps)
    // is its last one's. Only what the checkpoint orders separates runs: with
    // order, a display callback, an IOSurface interrupt or a host callback it
    // lists; without order, nothing, so the count says only whether any came.
    BOOL collapseIrq = [collapse containsObject:@"irq"];
    BOOL orderAll = [op[@"order"] boolValue];
    BOOL inIrqRun = NO;
    // collapse_per_display: a repeat counts once whenever that display's
    // previous callback was of the same kind, whatever came between (an
    // interrupt, another display's callbacks); new_frame coalesces presents
    // however they interleave with the rest.
    BOOL perDisplay = [op[@"collapse_per_display"] boolValue];
    NSMutableDictionary *lastCb = [NSMutableDictionary dictionary];
    for (NSDictionary *e in evs) {
        NSString *k = e[@"kind"];
        if (collapseIrq) {
            if ([k isEqual:@"irq"]) {
                if (inIrqRun && [e[@"vector"] isEqual:vectors.lastObject]) {
                    irqStamps[irqStamps.count - 1] = e[@"stamps"];
                    continue;
                }
                inIrqRun = YES;
            } else if (orderAll && ([k isEqual:@"cb"] || [k isEqual:@"iosfc_irq"] || !(sortBlocks || digestBlocks))) {
                // Sorted or digested host callbacks are not ordered, and a
                // display callback that its kind's collapse drops below is
                // not listed.
                BOOL dropped = [k isEqual:@"cb"] && [collapse containsObject:e[@"cb"]] && prevCb &&
                               [prevCb[@"cb"] isEqual:e[@"cb"]] && [prevCb[@"display"] isEqual:e[@"display"]];
                if (!dropped) inIrqRun = NO;
            }
        }
        // Callbacks whose count only timing decides (newFrame coalesces) can
        // be collapsed: consecutive repeats of one kind for one display count
        // once.
        if ([k isEqual:@"cb"] && [collapse containsObject:e[@"cb"]]) {
            NSDictionary *p = perDisplay ? lastCb[e[@"display"] ?: [NSNull null]] : prevCb;
            if (p && [p[@"cb"] isEqual:e[@"cb"]] && [p[@"display"] isEqual:e[@"display"]]) continue;
        }
        prevCb = [k isEqual:@"cb"] ? e : nil;
        if ([k isEqual:@"cb"]) lastCb[e[@"display"] ?: [NSNull null]] = e;
        NSMutableDictionary *c = [e mutableCopy];
        [c removeObjectsForKeys:@[ @"kind", @"seq" ]];
        if ([k isEqual:@"irq"]) {
            irqs++;
            [vectors addObject:e[@"vector"]];
            if ([e[@"vector"] unsignedIntValue] != 0) otherVector = YES;
            [irqStamps addObject:e[@"stamps"]];
            [order addObject:@"irq"];
        } else if ([k isEqual:@"iosfc_irq"]) {
            iosfc++;
            [order addObject:@"iosfc_irq"];
        } else if ([k isEqual:@"cb"]) {
            [cbs addObject:c];
            [order addObject:[NSString stringWithFormat:@"cb:%@", e[@"cb"]]];
        } else {
            [blks addObject:c];
            if (!sortBlocks && !digestBlocks) [order addObject:[NSString stringWithFormat:@"blk:%@", e[@"blk"]]];
        }
    }
    NSMutableDictionary *o = [@{@"ev" : @"events", @"irqs" : @(irqs)} mutableCopy];
    if (iosfc) o[@"iosfc_irqs"] = @(iosfc);
    // sort_callbacks: displays presenting at once call back in an order only
    // timing decides; report the callbacks sorted, and leave them out of
    // order.
    if ([op[@"sort_callbacks"] boolValue]) {
        NSString * (^ckey)(NSDictionary *) = ^NSString *(NSDictionary *e) {
          NSData *j = [NSJSONSerialization dataWithJSONObject:e options:NSJSONWritingSortedKeys error:nil];
          return [[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding];
        };
        [cbs sortUsingComparator:^NSComparisonResult(NSDictionary *x, NSDictionary *y) {
          return [ckey(x) compare:ckey(y)];
        }];
        [order filterUsingPredicate:[NSPredicate predicateWithFormat:@"NOT (SELF BEGINSWITH 'cb:')"]];
    }
    if (cbs.count) o[@"callbacks"] = cbs;
    // What the expected file keeps of a host callback's record: timing (t_*)
    // and diagnostics (d_*) would order ties, and change digests, by timing.
    NSString * (^key)(NSDictionary *) = ^NSString *(NSDictionary *e) {
      NSMutableDictionary *k = [NSMutableDictionary dictionary];
      for (NSString *f in e)
          if (![f hasPrefix:@"t_"] && ![f hasPrefix:@"d_"]) k[f] = e[f];
      NSData *j = [NSJSONSerialization dataWithJSONObject:k options:NSJSONWritingSortedKeys error:nil];
      return [[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding];
    };
    if (sortBlocks)
        [blks sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
          return [key(a) compare:key(b)];
        }];
    if (digestBlocks && blks.count) {
        NSMutableDictionary *counts = [NSMutableDictionary dictionary];
        NSMutableString *lines = [NSMutableString string];
        for (NSDictionary *b in blks) {
            counts[b[@"blk"]] = @([counts[b[@"blk"]] unsignedIntegerValue] + 1);
            [lines appendString:key(b)];
            [lines appendString:@"\n"];
        }
        NSData *d = [lines dataUsingEncoding:NSUTF8StringEncoding];
        o[@"blocks_count"] = counts;
        o[@"blocks_sha256"] = sha256(d.bytes, d.length);
    } else if (blks.count)
        o[@"blocks"] = blks;
    // The vector is always 0 today (reims-vgpu-core's DeviceCore::interrupt);
    // it is reported only when one is not, so that a change shows without
    // every file carrying it.
    if (otherVector) o[@"irq_vectors"] = vectors;
    if ([op[@"order"] boolValue]) o[@"order"] = order;
    if ([op[@"irq_stamps"] boolValue]) o[@"irq_stamps"] = irqStamps;
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

// ---- crash and hang reporting (docs/testing.md, section 5.5) ----

static void on_signal(int sig) {
    // The record of an MMIO call in progress lands before the death, if the
    // call returns (see g_recordPending).
    pthread_t self = pthread_self();
    uint64_t t0 = now_ns();
    if (!pthread_equal(self, g_scenarioThread) && !pthread_equal(self, atomic_load(&g_callThread))) {
        uint64_t deadline = now_ns() + 2000000000ull;
        while (atomic_load(&g_recordPending) && now_ns() < deadline) {
        }
    }
    atomic_store(&g_dying, 1);
    if (!pthread_equal(self, g_scenarioThread)) {
        uint64_t deadline = now_ns() + 2000000000ull;
        while (atomic_load(&g_writers) && now_ns() < deadline) {
        }
    }
    char buf[160];
    int n = snprintf(buf, sizeof(buf), "{\"d_signal\":%d,\"d_wait_us\":%llu,\"ev\":\"crash\",\"i\":%lld}\n", sig,
                     (unsigned long long)((now_ns() - t0) / 1000), (long long)atomic_load(&g_op));
    write(g_out_fd, buf, n);
    signal(sig, SIG_DFL);
    raise(sig);
}

static void on_alarm(int sig) {
    char buf[160];
    int n = snprintf(buf, sizeof(buf), "{\"ev\":\"watchdog\",\"i\":%lld}\n", (long long)atomic_load(&g_op));
    write(g_out_fd, buf, n);
    _exit(3);
}

// ---- the machine ----

static NSString *g_setup;
static id<MTLDevice> g_mtl;
static RvgDevice *g_dev;
static RvgIOSurfaceDevice *g_iosfc;
static NSDictionary *g_config;
// The displays, by the scenario's display index: NULL for one refused or
// released. A device's release empties the list, so the next display is 0.
static RvgDisplay **g_displays;
static NSUInteger g_ndisplays, g_displaysCap;
static NSMutableDictionary<NSString *, NSData *> *g_blobs;

// The image that answers p: "<build>" when it is the build under test,
// whatever its layout (a library inside the --build directory, or one
// linked into the runner itself), else its real path, so that a system or
// stray library shows.
static NSString *image_of(const void *p) {
    Dl_info info, self;
    if (!dladdr(p, &info) || !info.dli_fname) return @"(none)";
    if (dladdr((const void *)image_of, &self) && self.dli_fbase == info.dli_fbase) return @"<build>";
    char real[PATH_MAX];
    NSString *s = @(realpath(info.dli_fname, real) ? real : info.dli_fname);
    if (g_build && [s hasPrefix:[g_build stringByAppendingString:@"/"]]) return @"<build>";
    return s;
}

// The task callbacks, as QEMU implements them for hosts before macOS 15.4
// (vendor/qemu/hw/display/apple-gfx.m): a task is a reservation of host
// memory, and mapping remaps guest RAM into it.

typedef struct {
    uint32_t index;
    mach_vm_address_t address;
    uint64_t len;
    bool destroyed;
} host_task;

static _Atomic uint32_t g_ntasks;

// A host that refuses (reims_vgpu.h: create_task and add_trace_range may
// answer NULL; map_memory, unmap_memory and read_memory return bool). The
// descriptor's "refuse" names, per callback, the calls that fail: a list of
// 1-based ordinals among that callback's calls since the device was created,
// or true for every call. A refused call does nothing and reports failure,
// as a hypervisor out of address space or refusing a range would; its record
// says "refused". Host callbacks come in command order (decision 8), so an
// ordinal names a call the expected file already locks.
static NSDictionary *g_refuse;
static NSMutableDictionary<NSString *, NSNumber *> *g_refuseCount;  // under g_evLock

static BOOL refuse(NSString *blk) {
    id spec = g_refuse[blk];
    if (!spec) return NO;
    os_unfair_lock_lock(&g_evLock);
    NSUInteger n = [g_refuseCount[blk] unsignedIntegerValue] + 1;
    g_refuseCount[blk] = @(n);
    os_unfair_lock_unlock(&g_evLock);
    if ([spec isKindOfClass:[NSNumber class]]) return [spec boolValue];
    return [spec containsObject:@(n)];
}

static NSArray *ranges_json(uint32_t n, const RvgPhysicalMemoryRange *r) {
    NSMutableArray *a = [NSMutableArray array];
    for (uint32_t i = 0; i < n; i++) [a addObject:@[ hex64(r[i].physical_address), hex64(r[i].physical_length) ]];
    return a;
}

static void *on_create_task(void *ctx, uint64_t vmSize, void **baseAddress) {
    (void)ctx;
    device_callback_rule("create_task");
    mach_vm_address_t a = 0;
    host_task *t = NULL;
    BOOL no = refuse(@"createTask");
    if (!no && mach_vm_allocate(mach_task_self(), &a, vmSize, VM_FLAGS_ANYWHERE) == KERN_SUCCESS) {
        t = calloc(1, sizeof(*t));
        t->index = atomic_fetch_add(&g_ntasks, 1);
        t->address = a;
        t->len = vmSize;
        *baseAddress = (void *)a;
    }
    NSMutableDictionary *e = [@{@"blk" : @"createTask", @"size" : hex64(vmSize), @"task" : t ? @(t->index) : [NSNull null]} mutableCopy];
    if (no) e[@"refused"] = @YES;
    record_event(@"blk", e, &g_nblk);
    return t;
}

static void on_destroy_task(void *ctx, void *task) {
    (void)ctx;
    device_callback_rule("destroy_task");
    host_task *t = task;
    // The library can pass NULL (a root task create_task refused), and can
    // destroy a task twice: the host records what it was given and releases
    // a task once. Tasks stay allocated (a few bytes each) so that a later
    // call naming a destroyed task is seen, not a use-after-free in the
    // host.
    if (!t) {
        record_event(@"blk", @{@"blk" : @"destroyTask", @"task" : [NSNull null]}, &g_nblk);
        return;
    }
    NSMutableDictionary *e = [@{@"blk" : @"destroyTask", @"task" : @(t->index)} mutableCopy];
    if (t->destroyed) e[@"destroyed"] = @YES;
    record_event(@"blk", e, &g_nblk);
    if (t->destroyed) return;
    t->destroyed = true;
    mach_vm_deallocate(mach_task_self(), t->address, t->len);
}

static bool on_map_memory(void *ctx, void *task, uint32_t n, uint64_t voff, bool ro, const RvgPhysicalMemoryRange *r) {
    (void)ctx;
    device_callback_rule("map_memory");
    host_task *t = task;
    // A NULL task (the library passes one, e.g. efi_stale_task's unaligned
    // length) is recorded as given and refused.
    if (!t) {
        record_event(@"blk", @{@"blk" : @"mapMemory", @"task" : [NSNull null], @"voff" : hex64(voff), @"ro" : @(ro),
                               @"ranges" : ranges_json(n, r), @"ret" : @NO},
                     &g_nblk);
        return false;
    }
    // A destroyed task's reservation is gone: the host refuses to map into it
    // rather than remap over whatever now lives there.
    BOOL no = refuse(@"mapMemory"), gone = t->destroyed;
    bool ok = !no && !gone;
    uint64_t v = voff;
    for (uint32_t i = 0; !no && !gone && i < n; i++) {
        mach_vm_address_t target = t->address + v;
        v += r[i].physical_length;
        ram_range *rr = ram_for(r[i].physical_address, r[i].physical_length);
        if (!rr) {
            ok = false;
            continue;
        }
        vm_prot_t cur = 0, max = 0;
        kern_return_t kr = mach_vm_remap(mach_task_self(), &target, r[i].physical_length, vm_page_size - 1,
                                         VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(),
                                         (mach_vm_address_t)(rr->host + (r[i].physical_address - rr->gpa)), false, &cur,
                                         &max, VM_INHERIT_COPY);
        if (kr != KERN_SUCCESS) ok = false;
    }
    NSMutableDictionary *e = [@{@"blk" : @"mapMemory", @"task" : @(t->index), @"voff" : hex64(voff), @"ro" : @(ro),
                                @"ranges" : ranges_json(n, r), @"ret" : @(ok)} mutableCopy];
    if (no) e[@"refused"] = @YES;
    if (gone) e[@"destroyed"] = @YES;
    record_event(@"blk", e, &g_nblk);
    return ok;
}

static bool on_unmap_memory(void *ctx, void *task, uint64_t voff, uint64_t len) {
    (void)ctx;
    device_callback_rule("unmap_memory");
    host_task *t = task;
    if (!t) {
        record_event(@"blk", @{@"blk" : @"unmapMemory", @"task" : [NSNull null], @"voff" : hex64(voff), @"len" : hex64(len)},
                     &g_nblk);
        return true;
    }
    BOOL no = refuse(@"unmapMemory"), gone = t->destroyed;
    mach_vm_address_t a = t->address + voff;
    kern_return_t kr = KERN_SUCCESS;
    if (!no && !gone) kr = mach_vm_allocate(mach_task_self(), &a, len, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE);
    NSMutableDictionary *e = [@{@"blk" : @"unmapMemory", @"task" : @(t->index), @"voff" : hex64(voff), @"len" : hex64(len),
                                @"d_kr" : @(kr)} mutableCopy];
    if (gone) e[@"destroyed"] = @YES;
    if (no) {
        e[@"refused"] = @YES;
        e[@"ret"] = @NO;
    }
    record_event(@"blk", e, &g_nblk);
    return !no;  // QEMU always answers true
}

static bool on_read_memory(void *ctx, uint64_t pa, uint64_t len, void *dst) {
    (void)ctx;
    device_callback_rule("read_memory");
    BOOL no = refuse(@"readMemory");
    ram_range *rr = no ? NULL : ram_for(pa, len);
    if (rr) memcpy(dst, rr->host + (pa - rr->gpa), len);
    NSMutableDictionary *e = [@{@"blk" : @"readMemory", @"pa" : hex64(pa), @"len" : hex64(len), @"ret" : @(rr != NULL)} mutableCopy];
    if (no) e[@"refused"] = @YES;
    record_event(@"blk", e, &g_nblk);
    return rr != NULL;
}

static void on_raise_interrupt(void *ctx, uint32_t vector) {
    (void)ctx;
    device_callback_rule("raise_interrupt");
    record_irq(vector);
}

// The callbacks the scenario names ("blocks": which of the five), or all of
// them (nil).
static void install_task_callbacks(RvgHostCallbacks *cb, NSDictionary *which) {
    BOOL all = which == nil;
    if (all || [which[@"createTask"] boolValue]) cb->create_task = on_create_task;
    if (all || [which[@"destroyTask"] boolValue]) cb->destroy_task = on_destroy_task;
    if (all || [which[@"mapMemory"] boolValue]) cb->map_memory = on_map_memory;
    if (all || [which[@"unmapMemory"] boolValue]) cb->unmap_memory = on_unmap_memory;
    if (all || [which[@"readMemory"] boolValue]) cb->read_memory = on_read_memory;
}

// Trace ranges: the host keeps each handler, and the scenario says when the
// guest wrote (op trace_fire), as a host's write tracking would.

@interface PGRunTrace : NSObject
@property uint64_t gpa, len;
@property uint32_t index;
@property RvgTraceRangeHandler *handler;  // one reference, the host's
@end
@implementation PGRunTrace
@end

static NSMutableArray<PGRunTrace *> *g_traces;  // under g_evLock
static uint32_t g_ntraceRanges;

static void *on_add_trace_range(void *ctx, RvgPhysicalMemoryRange range, RvgTraceRangeHandler *handler) {
    (void)ctx;
    device_callback_rule("add_trace_range");
    if (refuse(@"addTraceRange")) {
        record_event(@"blk", @{@"blk" : @"addTraceRange", @"gpa" : hex64(range.physical_address),
                               @"len" : hex64(range.physical_length), @"trace" : [NSNull null], @"refused" : @YES},
                     &g_nblk);
        return NULL;
    }
    PGRunTrace *t = [PGRunTrace new];
    t.gpa = range.physical_address;
    t.len = range.physical_length;
    t.handler = rvg_trace_range_handler_retain(handler);
    os_unfair_lock_lock(&g_evLock);
    t.index = g_ntraceRanges++;
    [g_traces addObject:t];
    os_unfair_lock_unlock(&g_evLock);
    record_event(@"blk", @{@"blk" : @"addTraceRange", @"gpa" : hex64(t.gpa), @"len" : hex64(t.len), @"trace" : @(t.index)},
                 &g_nblk);
    return (__bridge_retained void *)t;
}

static void on_remove_trace_range(void *ctx, void *range) {
    (void)ctx;
    device_callback_rule("remove_trace_range");
    // As destroy_task: a NULL range (one add_trace_range refused) is recorded
    // as given.
    if (!range) {
        record_event(@"blk", @{@"blk" : @"removeTraceRange", @"trace" : [NSNull null]}, &g_nblk);
        return;
    }
    PGRunTrace *t = (__bridge_transfer PGRunTrace *)range;
    os_unfair_lock_lock(&g_evLock);
    [g_traces removeObject:t];
    os_unfair_lock_unlock(&g_evLock);
    record_event(@"blk", @{@"blk" : @"removeTraceRange", @"trace" : @(t.index)}, &g_nblk);
    rvg_trace_range_handler_release(t.handler);
}

// The trace callbacks the scenario names ("trace": true for both, or which).
static void install_trace_callbacks(RvgHostCallbacks *cb, id which) {
    BOOL all = [which isKindOfClass:[NSNumber class]];
    if (all || [which[@"add"] boolValue]) cb->add_trace_range = on_add_trace_range;
    if (all || [which[@"remove"] boolValue]) cb->remove_trace_range = on_remove_trace_range;
}

// Each handler the write touches is invoked through a reference of the
// op's own, taken under the lock, so a remove_trace_range meanwhile never
// frees one under it.
static void op_trace_fire(NSDictionary *op) {
    uint64_t gpa = num(op[@"gpa"]), len = num(op[@"len"]);
    os_unfair_lock_lock(&g_evLock);
    NSUInteger n = 0;
    RvgTraceRangeHandler **hs = calloc(g_traces.count ? g_traces.count : 1, sizeof(*hs));
    for (PGRunTrace *t in g_traces)
        if (!(gpa + len <= t.gpa || gpa >= t.gpa + t.len)) hs[n++] = rvg_trace_range_handler_retain(t.handler);
    os_unfair_lock_unlock(&g_evLock);
    for (NSUInteger i = 0; i < n; i++) LIBV(rvg_trace_range_handler_invoke(hs[i], (RvgPhysicalMemoryRange){gpa, len}));
    for (NSUInteger i = 0; i < n; i++) rvg_trace_range_handler_release(hs[i]);
    free(hs);
    emit(@{@"ev" : @"trace_fire", @"fired" : @(n)});
}

// ---- displays ----
//
// A display's callbacks table carries its index as ctx.

static NSNumber *display_index(void *ctx) { return @((NSUInteger)(uintptr_t)ctx); }

static void on_new_frame(void *ctx) {
    display_callback_enter("new_frame");
    record_event(@"cb", @{@"cb" : @"newFrame", @"display" : display_index(ctx)}, &g_ncb);
    display_callback_exit();
}

static void on_mode_change(void *ctx, uint16_t w, uint16_t h, uint32_t fmt) {
    display_callback_enter("mode_change");
    record_event(@"cb", @{@"cb" : @"modeChange", @"display" : display_index(ctx), @"w" : @(w), @"h" : @(h), @"fmt" : hex64(fmt)},
                 &g_ncb);
    display_callback_exit();
}

static void on_cursor_glyph(void *ctx, const RvgCursorGlyph *g) {
    display_callback_enter("cursor_glyph");
    if (g->struct_size < offsetof(RvgCursorGlyph, rgba_length) + sizeof(g->rgba_length))
        fatal_scenario(@"cursor_glyph: the glyph is smaller than this runner's RvgCursorGlyph");
    size_t n = g->rgba_length;
    NSMutableDictionary *c = [@{
        @"cb" : @"cursorGlyph",
        @"display" : display_index(ctx),
        @"w" : @(g->width),
        @"h" : @(g->height),
        @"hot" : @[ @(g->hot_spot.x), @(g->hot_spot.y) ],
        @"sha256" : sha256(g->rgba, n),
    } mutableCopy];
    if (n <= 1024) c[@"hex"] = to_hex(g->rgba, n);
    record_event(@"cb", c, &g_ncb);
    display_callback_exit();
}

// The guest's show byte, boxed as a BOOL: the function takes a BOOL and is
// registered through a cast to RvgCursorShowFn, which passes the byte as the
// guest wrote it.
static void on_cursor_show(void *ctx, BOOL show) {
    display_callback_enter("cursor_show");
    record_event(@"cb", @{@"cb" : @"cursorShow", @"display" : display_index(ctx), @"show" : @(show)}, &g_ncb);
    display_callback_exit();
}

static void on_cursor_move(void *ctx) {
    display_callback_enter("cursor_move");
    record_event(@"cb", @{@"cb" : @"cursorMove", @"display" : display_index(ctx)}, &g_ncb);
    display_callback_exit();
}

// modes: [[w, h, refresh rate], ...]; a mode's size is the guest's 16 bits.
// Freed by the caller.
static RvgDisplayMode *modes_from(NSArray *spec) {
    RvgDisplayMode *modes = calloc(spec.count ? spec.count : 1, sizeof(*modes));
    for (NSUInteger i = 0; i < spec.count; i++) {
        NSArray *m = spec[i];
        modes[i] = (RvgDisplayMode){(uint16_t)[m[0] intValue], (uint16_t)[m[1] intValue], 0, dbl(m[2])};
    }
    return modes;
}

static void set_modes(RvgDisplay *d, NSArray *spec) {
    RvgDisplayMode *modes = modes_from(spec);
    LIBV(rvg_display_set_modes(d, modes, spec.count));
    free(modes);
}

static void new_display(NSDictionary *d) {
    NSUInteger index = g_ndisplays;
    if (d[@"queues"] || d[@"no_queue"]) fatal_scenario(@"new_display: the C API has no display queue");
    if (d[@"name"] == [NSNull null]) fatal_scenario(@"new_display: the C API has no nil name");
    NSArray *handlers = d[@"handlers"] ?: @[ @"newFrame", @"modeChange", @"cursorGlyph", @"cursorShow", @"cursorMove" ];
    RvgDisplayCallbacks cb = {.struct_size = sizeof(cb), .ctx = (void *)(uintptr_t)index};
    if ([handlers containsObject:@"newFrame"]) cb.new_frame = on_new_frame;
    if ([handlers containsObject:@"modeChange"]) cb.mode_change = on_mode_change;
    if ([handlers containsObject:@"cursorGlyph"]) cb.cursor_glyph = on_cursor_glyph;
    if ([handlers containsObject:@"cursorShow"]) cb.cursor_show = (RvgCursorShowFn)on_cursor_show;
    if ([handlers containsObject:@"cursorMove"]) cb.cursor_move = on_cursor_move;
    NSArray *mm = d[@"mm"] ?: @[ @400, @300 ];  // QEMU's 20" display
    // Absent is the library's default name; a name crosses as its UTF-8.
    NSData *name = [d[@"name"] dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:YES];
    RvgDisplayConfig c = {
        .struct_size = sizeof(c),
        .serial_number = (uint32_t)num(d[@"serial"]),
        .port = num(d[@"port"]),
        .name = d[@"name"] ? (name.length ? name.bytes : "") : NULL,
        .name_length = name.length,
        .width_millimeters = dbl(mm[0]),
        .height_millimeters = dbl(mm[1]),
        .callbacks = &cb,
    };
    RvgDisplay *disp = NULL;
    RvgStatus st = LIB(rvg_display_create(g_dev, &c, &disp));
    if (st != RVG_OK) disp = NULL;
    if (g_ndisplays == g_displaysCap) {
        g_displaysCap = g_displaysCap ? 2 * g_displaysCap : 16;
        g_displays = realloc(g_displays, g_displaysCap * sizeof(*g_displays));
    }
    g_displays[g_ndisplays++] = disp;
    NSMutableDictionary *o = [@{@"ev" : @"display", @"display" : @(index), @"ok" : @(disp != NULL)} mutableCopy];
    if (!disp) o[@"status"] = status_name(st);
    if (disp && d[@"modes"]) {
        set_modes(disp, d[@"modes"]);
        o[@"modes_valid"] = @([d[@"modes"] count]);
    }
    emit(o);
}

static RvgDisplay *display_at(NSDictionary *op) {
    NSUInteger i = (NSUInteger)num(op[@"display"]);
    return i < g_ndisplays ? g_displays[i] : NULL;
}

// Releases every display, in index order, and empties the list.
static void release_displays(void) {
    for (NSUInteger i = 0; i < g_ndisplays; i++)
        if (g_displays[i]) LIBV(rvg_display_release(g_displays[i]));
    g_ndisplays = 0;
}

// ---- device creation ----

static void create_device(NSDictionary *cfg) {
    NSDictionary *dv = cfg[@"descriptor"] ?: @{};
    for (NSString *k in @[ @"create", @"device", @"readback", @"mmioLength", @"enableProtectedContent" ])
        if (dv[k]) fatal_scenario([@"the C API has no device input " stringByAppendingString:k]);
    os_unfair_lock_lock(&g_evLock);
    g_refuse = dv[@"refuse"];
    g_refuseCount = [NSMutableDictionary dictionary];
    os_unfair_lock_unlock(&g_evLock);
    RvgHostCallbacks cb = {.struct_size = sizeof(cb)};
    RvgDeviceConfig c = {.struct_size = sizeof(c), .callbacks = &cb};
    // One display port unless the scenario says; 0 is the system's default
    // Metal device, and no device has the ID UINT64_MAX.
    c.display_port_count = dv[@"port_count"] ? (uint32_t)num(dv[@"port_count"]) : 1;
    c.metal_registry_id = [dv[@"no_mtl_device"] boolValue] ? UINT64_MAX : 0;
    BOOL blocks = [g_setup isEqual:@"blocks"];
    if (blocks || dv[@"blocks"]) install_task_callbacks(&cb, dv[@"blocks"]);
    RvgMemoryRange *ranges = NULL;
    RvgMemoryMap map = {0};
    if ((!blocks || [dv[@"map_too"] boolValue]) && ![dv[@"no_map"] boolValue]) {
        if (dv[@"map_ranges"]) {
            // [gpa, len, host gpa]: a range registered as the scenario says,
            // for refusals such as overlaps.
            NSArray *spec = dv[@"map_ranges"];
            ranges = calloc(spec.count ? spec.count : 1, sizeof(*ranges));
            for (NSUInteger i = 0; i < spec.count; i++) {
                NSArray *r = spec[i];
                ram_range *rr = ram_for(num(r[2]), 1);
                void *host = rr ? rr->host + (num(r[2]) - rr->gpa) : NULL;
                ranges[i] = (RvgMemoryRange){num(r[0]), num(r[1]), host};
            }
            map = (RvgMemoryMap){ranges, spec.count};
        } else {
            ranges = ram_map();
            map = (RvgMemoryMap){ranges, (size_t)g_nram};
        }
        c.memory_map = &map;
    }
    if ([g_setup isEqual:@"isolate"]) c.process_isolation = true;
    // apple-gfx-mmio: the mapper goes with the IOSurface host device, and
    // argument buffers stay on.
    if ([cfg[@"iosfc"] boolValue]) c.use_iosurface_mapper = true;
    if (dv[@"enableArgumentBuffers"]) c.disable_argument_buffers = ![dv[@"enableArgumentBuffers"] boolValue];
    if (dv[@"trace"]) install_trace_callbacks(&cb, dv[@"trace"]);
    if (![dv[@"no_raise"] boolValue]) cb.raise_interrupt = on_raise_interrupt;
    RvgStatus st = LIB(rvg_device_create(&c, &g_dev));
    if (st != RVG_OK) g_dev = NULL;
    free(ranges);
    NSMutableDictionary *o = [@{@"ev" : @"device", @"ok" : @(g_dev != NULL)} mutableCopy];
    if (!g_dev) o[@"status"] = status_name(st);
    emit(o);
    if (!g_dev) return;
    for (NSDictionary *d in cfg[@"displays"]) new_display(d);
}

static void release_all(void) {
    release_displays();
    if (g_dev) LIBV(rvg_device_release(g_dev));
    g_dev = NULL;
}

// ---- the IOSurface host device ----

static void on_iosfc_raise(void *ctx, uint32_t vector) {
    (void)ctx;
    if (t_runner) fatal_scenario(@"thread rule: the IOSurface raise_interrupt ran on a runner thread");
    if (atomic_fetch_add(&g_iosfcCallbacks, 1)) fatal_scenario(@"thread rule: IOSurface raise_interrupt calls overlapped");
    record_event(@"iosfc_irq", @{@"vector" : @(vector)}, &g_niosfcIrq);
    atomic_fetch_sub(&g_iosfcCallbacks, 1);
}

// An IOSurface host device over guest RAM (none with no_map, which it
// refuses), as apple-gfx-mmio makes one.
static RvgStatus new_iosfc(BOOL noMap, RvgIOSurfaceDevice **out) {
    RvgMemoryRange *ranges = ram_map();
    RvgMemoryMap map = {ranges, (size_t)g_nram};
    RvgIOSurfaceCallbacks cb = {.struct_size = sizeof(cb), .raise_interrupt = on_iosfc_raise};
    RvgIOSurfaceDeviceConfig c = {.struct_size = sizeof(c), .memory_map = noMap ? NULL : &map, .callbacks = &cb};
    *out = NULL;
    RvgStatus st = LIB(rvg_iosurface_device_create(&c, out));
    if (st != RVG_OK) *out = NULL;
    free(ranges);
    return st;
}

static void create_iosfc(void) {
    RvgStatus st = new_iosfc([g_config[@"iosfc_no_map"] boolValue], &g_iosfc);
    NSMutableDictionary *o = [@{@"ev" : @"iosfc", @"ok" : @(g_iosfc != NULL)} mutableCopy];
    if (!g_iosfc) o[@"status"] = status_name(st);
    emit(o);
}

// ---- the handoff between phases (a migration) ----
//
// What a VM restore carries from the process that suspended to the one that
// resumes: guest RAM, and the suspend blobs the host kept (the slots). The
// scenario's shadow of RAM goes with it, so that a diff in the new process
// reports what the device changed, as in one process. An image is a
// directory under --handoff: index.plist (the RAM ranges and the slots) and,
// a range each, host<N>.bin and shadow<N>.bin, which hold only the chunks
// that are not all zero, each as {u64 offset, u64 length, bytes}. The target
// loads by guest physical address: what overlaps its own RAM is copied, and
// the rest of its RAM starts zeroed, so a target may have less RAM, more,
// or RAM elsewhere.

#define HANDOFF_CHUNK 0x4000ull

static NSString *handoff_image(id spec) {
    NSString *name = [spec isKindOfClass:[NSDictionary class]] && spec[@"image"] ? [spec[@"image"] description] : @"0";
    if (!g_handoff) fatal_scenario(@"a handoff needs --handoff");
    if (!name.length || [name containsString:@"/"] || [name hasPrefix:@"."]) fatal_scenario(@"bad handoff image name");
    return [g_handoff stringByAppendingPathComponent:name];
}

static void handoff_write_chunks(NSString *path, const uint8_t *p, uint64_t size) {
    NSMutableData *d = [NSMutableData data];
    static const uint8_t zero[HANDOFF_CHUNK];
    for (uint64_t off = 0; off < size; off += HANDOFF_CHUNK) {
        uint64_t n = MIN(HANDOFF_CHUNK, size - off);
        if (!memcmp(p + off, zero, n)) continue;
        uint64_t hdr[2] = {off, n};
        [d appendBytes:hdr length:sizeof(hdr)];
        [d appendBytes:p + off length:n];
    }
    if (![d writeToFile:path atomically:YES]) fatal_scenario([@"cannot write " stringByAppendingString:path]);
}

// Copies a saved range's chunks into whatever part of the target's RAM
// (host bytes, or shadow when shadow is set) has the same guest addresses.
static void handoff_read_chunks(NSString *path, uint64_t gpa, BOOL shadow) {
    NSData *d = [NSData dataWithContentsOfFile:path];
    if (!d) fatal_scenario([@"no handoff file " stringByAppendingString:path]);
    const uint8_t *b = d.bytes;
    for (NSUInteger at = 0; at + 16 <= d.length;) {
        uint64_t hdr[2];
        memcpy(hdr, b + at, sizeof(hdr));
        at += 16;
        if (at + hdr[1] > d.length) fatal_scenario([@"truncated handoff file " stringByAppendingString:path]);
        uint64_t a = gpa + hdr[0], n = hdr[1];
        for (int i = 0; i < g_nram; i++) {
            ram_range *rr = &g_ram[i];
            uint64_t lo = MAX(a, rr->gpa), hi = MIN(a + n, rr->gpa + rr->size);
            if (lo >= hi) continue;
            memcpy((shadow ? rr->shadow : rr->host) + (lo - rr->gpa), b + at + (lo - a), hi - lo);
        }
        at += n;
    }
}

static NSArray *ram_json(void) {
    NSMutableArray *a = [NSMutableArray array];
    for (int i = 0; i < g_nram; i++) [a addObject:@[ hex64(g_ram[i].gpa), hex64(g_ram[i].size) ]];
    return a;
}

static void op_handoff_save(NSDictionary *op) {
    NSString *dir = handoff_image(op);
    [[NSFileManager defaultManager] removeItemAtPath:dir error:nil];
    if (![[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil])
        fatal_scenario([@"cannot create " stringByAppendingString:dir]);
    for (int i = 0; i < g_nram; i++) {
        handoff_write_chunks([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"host%d.bin", i]], g_ram[i].host,
                             g_ram[i].size);
        handoff_write_chunks([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"shadow%d.bin", i]],
                             g_ram[i].shadow, g_ram[i].size);
    }
    NSDictionary *index = @{@"ram" : ram_json(), @"blobs" : g_blobs ?: @{}};
    if (![index writeToFile:[dir stringByAppendingPathComponent:@"index.plist"] atomically:YES])
        fatal_scenario(@"cannot write the handoff index");
    emit(@{
        @"ev" : @"handoff_save",
        @"image" : dir.lastPathComponent,
        @"ram" : ram_json(),
        @"slots" : [[g_blobs allKeys] sortedArrayUsingSelector:@selector(compare:)] ?: @[],
    });
}

// config handoff_load: true, or {"image": name, "ram": bool, "blobs": bool}
// (both default true; "ram": false is a restore whose memory image was lost).
static void handoff_load(id spec) {
    NSString *dir = handoff_image(spec);
    NSDictionary *o = [spec isKindOfClass:[NSDictionary class]] ? spec : @{};
    BOOL ram = o[@"ram"] ? [o[@"ram"] boolValue] : YES, blobs = o[@"blobs"] ? [o[@"blobs"] boolValue] : YES;
    NSDictionary *index = [NSDictionary dictionaryWithContentsOfFile:[dir stringByAppendingPathComponent:@"index.plist"]];
    if (!index) fatal_scenario(@"no handoff image: the phase before saved none");
    NSArray *src = index[@"ram"];
    if (ram)
        for (NSUInteger i = 0; i < src.count; i++) {
            uint64_t gpa = num(src[i][0]);
            handoff_read_chunks([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"host%lu.bin", (unsigned long)i]],
                                gpa, NO);
            handoff_read_chunks(
                [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"shadow%lu.bin", (unsigned long)i]], gpa, YES);
        }
    NSDictionary *slots = blobs ? index[@"blobs"] : @{};
    [g_blobs addEntriesFromDictionary:slots];
    emit(@{
        @"ev" : @"handoff_load",
        @"image" : dir.lastPathComponent,
        @"from_ram" : src ?: @[],
        @"ram" : ram_json(),
        @"loaded_ram" : @(ram),
        @"slots" : [[slots allKeys] sortedArrayUsingSelector:@selector(compare:)],
    });
}

static void op_config(NSDictionary *op) {
    g_config = op;
    NSArray *ranges = op[@"ram"] ?: @[ @[ @"0x40000000", @"0x4000000" ] ];
    if (g_ram) fatal_scenario(@"config twice");
    g_ram = calloc(ranges.count, sizeof(ram_range));
    for (NSArray *r in ranges) {
        ram_range *rr = &g_ram[g_nram++];
        rr->gpa = num(r[0]);
        rr->size = num(r[1]);
        mach_vm_address_t host = 0;
        if (mach_vm_allocate(mach_task_self(), &host, rr->size, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
            fatal_scenario(@"mach_vm_allocate failed");
        rr->host = (uint8_t *)host;
        rr->shadow = calloc(1, rr->size);
    }
    // Before any device exists, as a restoring host fills guest RAM first.
    if (op[@"handoff_load"]) handoff_load(op[@"handoff_load"]);
    g_stampGPA = op[@"stamp_gpa"] ? num(op[@"stamp_gpa"]) : 0;
    if (op[@"stamp_slots"]) g_stampSlots = (uint32_t)num(op[@"stamp_slots"]);
    // For the metal queries and the diagnostic GPU name; the library makes
    // its own frame queue at the device's creation.
    g_mtl = MTLCreateSystemDefaultDevice();
    emit(@{
        @"ev" : @"binding",
        @"library" : image_of((const void *)rvg_device_create),
        @"setup" : g_setup,
        @"d_gpu" : g_mtl.name ?: @"",
        @"d_pid" : @(getpid()),
    });
    if ([op[@"iosfc"] boolValue] && ![op[@"iosfc_after"] boolValue]) create_iosfc();
    create_device(op);
    if ([op[@"iosfc"] boolValue] && [op[@"iosfc_after"] boolValue]) create_iosfc();
}

// ---- MMIO, as QEMU issues it: on a pool thread, the caller waiting ----

typedef struct {
    void *target;
    BOOL iosfc, write, record;
    uint64_t off;
    uint32_t val, result;
    dispatch_semaphore_t done;
} mmio_job;

static void mmio_do(void *p) {
    mmio_job *j = p;
    t_runner = 1;
    if (j->record) {
        atomic_store(&g_callThread, pthread_self());
        atomic_store(&g_recordPending, 1);
    }
    // QEMU declares the IOSurface host device's MMIO 32 bits wide: a write
    // is zero-extended, and a read truncated (docs/testing.md, section 3.1).
    if (j->iosfc) {
        if (j->write) LIBV(rvg_iosurface_device_mmio_write(j->target, j->off, (uint64_t)j->val));
        else j->result = (uint32_t)LIB(rvg_iosurface_device_mmio_read(j->target, j->off));
    } else {
        if (j->write) LIBV(rvg_device_mmio_write(j->target, j->off, j->val));
        else j->result = LIB(rvg_device_mmio_read(j->target, j->off));
    }
    if (j->record) atomic_store(&g_callThread, (pthread_t)NULL);
    t_runner = 0;
    dispatch_semaphore_signal(j->done);
}

static uint32_t mmio_call(BOOL iosfc, BOOL write, uint64_t off, uint32_t val, BOOL record) {
    void *target = iosfc ? (void *)g_iosfc : (void *)g_dev;
    if (!target) fatal_scenario(iosfc ? @"no IOSurface host device" : @"no device");
    mmio_job j = {target, iosfc, write, record, off, val, 0, dispatch_semaphore_create(0)};
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), &j, mmio_do);
    dispatch_semaphore_wait(j.done, DISPATCH_TIME_FOREVER);
    return j.result;
}

static uint32_t mmio(BOOL iosfc, BOOL write, uint64_t off, uint32_t val) {
    return mmio_call(iosfc, write, off, val, NO);
}

static void op_mmio_w(NSDictionary *op, BOOL iosfc) {
    uint64_t t = now_ns();
    BOOL quiet = [op[@"quiet"] boolValue];
    mmio_call(iosfc, YES, num(op[@"off"]), (uint32_t)num(op[@"val"]), !quiet);
    if (quiet) return;
    emit(@{@"ev" : iosfc ? @"iosfc_w" : @"mmio_w", @"off" : hex64(num(op[@"off"])), @"val" : hex64((uint32_t)num(op[@"val"])),
           @"t_us" : @((now_ns() - t) / 1000)});
}

static void op_mmio_r(NSDictionary *op, BOOL iosfc) {
    uint32_t v = mmio_call(iosfc, NO, num(op[@"off"]), 0, YES);
    NSMutableDictionary *o = [@{@"ev" : iosfc ? @"iosfc_r" : @"mmio_r", @"off" : hex64(num(op[@"off"])), @"val" : hex64(v)} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

static void op_mmio_sweep(NSDictionary *op, BOOL iosfc) {
    uint64_t from = num(op[@"from"]), to = num(op[@"to"]), step = op[@"step"] ? num(op[@"step"]) : 4;
    NSMutableDictionary *nz = [NSMutableDictionary dictionary];
    for (uint64_t off = from; off < to; off += step) {
        uint32_t v = mmio(iosfc, NO, off, 0);
        if (v) nz[hex64(off)] = hex64(v);
    }
    NSMutableDictionary *o = [@{@"ev" : iosfc ? @"iosfc_sweep" : @"mmio_sweep", @"from" : hex64(from), @"to" : hex64(to),
                                @"nonzero" : nz} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

// ---- waits: on guest-visible conditions, with a deadline ----
//
// A wait polls guest memory, event counts, or a register that reading does
// not change (never a read-to-clear one), until the condition holds or the
// deadline passes.

// display: count only that display's callbacks (nil: every display's).
static uint32_t count_cb(NSString *kind, NSNumber *display) {
    uint32_t n = 0;
    os_unfair_lock_lock(&g_evLock);
    for (NSDictionary *e in g_events)
        if ([e[@"kind"] isEqual:@"cb"] && [e[@"cb"] isEqual:kind] && (!display || [e[@"display"] isEqual:display])) n++;
    os_unfair_lock_unlock(&g_evLock);
    return n;
}

static BOOL wait_satisfied(NSDictionary *op) {
    if (op[@"gpa"]) {
        uint32_t v = *(volatile uint32_t *)gpa_ptr(num(op[@"gpa"]), 4);
        if (op[@"mask"]) v &= (uint32_t)num(op[@"mask"]);
        if (op[@"eq"] && v != (uint32_t)num(op[@"eq"])) return NO;
        if (op[@"ge"] && (int32_t)(v - (uint32_t)num(op[@"ge"])) < 0) return NO;
        if (op[@"ne"] && v == (uint32_t)num(op[@"ne"])) return NO;
    }
    if (op[@"irqs"] && atomic_load(&g_nirq) < num(op[@"irqs"])) return NO;
    if (op[@"iosfc_irqs"] && atomic_load(&g_niosfcIrq) < num(op[@"iosfc_irqs"])) return NO;
    if (op[@"callbacks"] && atomic_load(&g_ncb) < num(op[@"callbacks"])) return NO;
    if (op[@"blocks"] && atomic_load(&g_nblk) < num(op[@"blocks"])) return NO;
    if (op[@"cb"] && count_cb(op[@"cb"], op[@"cb_display"]) < num(op[@"cb_count"] ?: @1)) return NO;
    if (op[@"mmio"]) {
        uint32_t v = mmio(NO, NO, num(op[@"mmio"]), 0);
        if (op[@"mmio_eq"] && v != (uint32_t)num(op[@"mmio_eq"])) return NO;
        if (op[@"mmio_ne"] && v == (uint32_t)num(op[@"mmio_ne"])) return NO;
    }
    if (op[@"present_count"]) {
        RvgDisplay *d = display_at(op);
        uint64_t guest = 0;
        if (d) LIBV(rvg_display_present_counts(d, &guest, NULL));
        if (!d || guest < num(op[@"present_count"])) return NO;
    }
    return YES;
}

static int wait_which(NSDictionary *op) {
    // "any": a list of conditions, satisfied when one of them is; which one
    // is reported (the first in list order that holds).
    if (!op[@"any"]) return wait_satisfied(op) ? 0 : -1;
    NSArray *conds = op[@"any"];
    for (NSUInteger i = 0; i < conds.count; i++)
        if (wait_satisfied(conds[i])) return (int)i;
    return -1;
}

static void op_wait(NSDictionary *op) {
    uint64_t timeout_ms = op[@"timeout_ms"] ? num(op[@"timeout_ms"]) : 5000;
    uint64_t start = now_ns(), deadline = start + timeout_ms * 1000000ull;
    BOOL ok;
    int which = -1;
    for (;;) {
        if ((ok = (which = wait_which(op)) >= 0)) break;
        if (now_ns() >= deadline) break;
        // Events kick the condition; guest memory has no doorbell of its own,
        // so each poll waits at most 1 ms.
        pthread_mutex_lock(&g_waitMutex);
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&g_waitCond, &g_waitMutex, &ts);
        pthread_mutex_unlock(&g_waitMutex);
    }
    NSMutableDictionary *o = [@{@"ev" : @"wait", @"ok" : @(ok), @"t_us" : @((now_ns() - start) / 1000)} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    if (op[@"any"] && ok) o[@"which"] = @(which);
    if (!ok && op[@"gpa"]) o[@"val"] = hex64(*(volatile uint32_t *)gpa_ptr(num(op[@"gpa"]), 4));
    emit(o);
}

// ---- guest memory ops ----

static void op_diff(NSDictionary *op) {
    NSMutableArray *runs = [NSMutableArray array];
    uint64_t lo = op[@"gpa"] ? num(op[@"gpa"]) : 0, hi = op[@"gpa"] ? lo + num(op[@"len"]) : UINT64_MAX;
    for (int ri = 0; ri < g_nram; ri++) {
        ram_range *rr = &g_ram[ri];
        uint64_t a = MAX(lo, rr->gpa), b = MIN(hi, rr->gpa + rr->size);
        if (a >= b) continue;
        uint8_t *m = rr->host + (a - rr->gpa), *s = rr->shadow + (a - rr->gpa);
        uint64_t n = b - a, i = 0;
        while (i < n) {
            if ((i & 7) == 0 && i + 8 <= n && *(uint64_t *)(m + i) == *(uint64_t *)(s + i)) {
                i += 8;
                continue;
            }
            if (m[i] == s[i]) {
                i++;
                continue;
            }
            // A run ends after 16 equal bytes, so nearby changes group.
            uint64_t j = i, last = i;
            while (j < n && j - last < 16) {
                if (m[j] != s[j]) last = j;
                j++;
            }
            uint64_t len = last + 1 - i;
            NSMutableDictionary *r = [@{@"gpa" : hex64(a + i), @"len" : hex64(len)} mutableCopy];
            if (len <= 1024) {
                r[@"old"] = to_hex(s + i, len);
                r[@"new"] = to_hex(m + i, len);
            } else {
                r[@"sha256_new"] = sha256(m + i, len);
            }
            [runs addObject:r];
            i = last + 1;
        }
        memcpy(s, m, n);
    }
    NSMutableDictionary *o = [@{@"ev" : @"diff", @"runs" : runs} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

static void write_guest(uint64_t gpa, const void *p, size_t n) {
    memcpy(gpa_ptr(gpa, n), p, n);
    ram_range *rr = ram_for(gpa, n);
    memcpy(rr->shadow + (gpa - rr->gpa), p, n);
}

static void op_mem(NSDictionary *op) {
    NSData *d = from_hex(op[@"hex"]);
    write_guest(num(op[@"gpa"]), d.bytes, d.length);
}

// N copies of a template at gpa + k * stride; each field [off, size, start,
// step] stores start + k * step, little-endian, size bytes (4 or 8), at off
// in copy k. For guest structures too large to spell out (thousands of
// commands, page tables of thousands of entries).
static void op_mem_seq(NSDictionary *op) {
    NSData *t = from_hex(op[@"hex"]);
    uint64_t gpa = num(op[@"gpa"]), count = num(op[@"count"]), stride = num(op[@"stride"]);
    if (stride < t.length) fatal_scenario(@"mem_seq: stride shorter than the template");
    NSMutableData *d = [NSMutableData dataWithLength:(count ? (count - 1) * stride + t.length : 0)];
    uint8_t *b = d.mutableBytes;
    for (uint64_t k = 0; k < count; k++) {
        memcpy(b + k * stride, t.bytes, t.length);
        for (NSArray *f in op[@"fields"] ?: @[]) {
            uint64_t off = num(f[0]), size = num(f[1]), v = num(f[2]) + k * num(f[3]);
            if ((size != 4 && size != 8) || off + size > t.length) fatal_scenario(@"mem_seq: bad field");
            memcpy(b + k * stride + off, &v, size);
        }
    }
    if (d.length) write_guest(gpa, d.bytes, d.length);
}

static void op_fill(NSDictionary *op) {
    uint64_t len = num(op[@"len"]);
    NSMutableData *d = [NSMutableData dataWithLength:len];
    memset(d.mutableBytes, (int)num(op[@"byte"]), len);
    write_guest(num(op[@"gpa"]), d.bytes, len);
}

static void op_mem_file(NSDictionary *op) {
    NSString *path = [g_fixtures stringByAppendingPathComponent:op[@"file"]];
    NSData *d = [NSData dataWithContentsOfFile:path];
    if (!d) fatal_scenario([@"no fixture " stringByAppendingString:path]);
    if (op[@"sha256"] && ![sha256(d.bytes, d.length) isEqual:op[@"sha256"]])
        fatal_scenario([@"fixture changed: " stringByAppendingString:op[@"file"]]);
    write_guest(num(op[@"gpa"]), d.bytes, d.length);
}

// Compares two guest ranges: a differential check, where two encodings of one
// workload must leave identical memory. Reports equality and, if not, the
// first differing offset.
static void op_cmp(NSDictionary *op) {
    uint64_t a = num(op[@"a"]), b = num(op[@"b"]), len = num(op[@"len"]);
    uint8_t *pa = gpa_ptr(a, len), *pb = gpa_ptr(b, len);
    NSMutableDictionary *o = [@{@"ev" : @"cmp", @"a" : hex64(a), @"b" : hex64(b), @"len" : hex64(len)} mutableCopy];
    // skip: [[offset, length]...] left out of the comparison (padding a
    // reply leaves uninitialised); the observation lists them.
    NSArray *skip = op[@"skip"] ?: @[];
    BOOL (^skipped)(uint64_t) = ^BOOL(uint64_t off) {
      for (NSArray *r in skip)
          if (off >= num(r[0]) && off < num(r[0]) + num(r[1])) return YES;
      return NO;
    };
    uint64_t i = 0;
    while (i < len && (pa[i] == pb[i] || skipped(i))) i++;
    o[@"equal"] = @(i == len);
    if (i < len) o[@"first_diff"] = hex64(i);
    if (skip.count) o[@"skip"] = skip;
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

static void op_dump(NSDictionary *op) {
    uint64_t gpa = num(op[@"gpa"]), len = num(op[@"len"]);
    uint8_t *p = gpa_ptr(gpa, len);
    NSMutableDictionary *o = [@{@"ev" : @"dump", @"gpa" : hex64(gpa), @"len" : hex64(len)} mutableCopy];
    if (len <= 4096) o[@"hex"] = to_hex(p, len);
    else o[@"sha256"] = sha256(p, len);
    if (op[@"label"]) o[@"label"] = op[@"label"];
    emit(o);
}

// ---- the host API ----

// A mutation of a blob, for willResume: "patch" [[offset, u32], ...], then
// "truncate" to that many bytes. A blob the device wrote (a slot) is first
// made canonical: every byte the library never wrote is zeroed, since a
// mutation may make the library read what it never wrote. Written are the
// header and the 22 device-state pairs (+0 to +192, from an initialised
// array), and the device-info pairs write_device_info wrote, up to and
// including their {0, 0}; the rest of the device-info slots and the last 8
// bytes are not. Those are the bytes the normalizer's blob_pgsd reports
// (pgtest/normalize.py).
static NSData *mutate_blob(NSData *blob, NSDictionary *op, BOOL canonical) {
    NSMutableData *m = [blob mutableCopy] ?: [NSMutableData data];
    uint8_t *b = m.mutableBytes;
    NSUInteger len = m.length;
    if (canonical && len >= 16 && ((uint32_t *)b)[1] == 2) {
        NSUInteger off = 192;
        while (off + 16 <= len) {
            uint32_t k;
            memcpy(&k, b + off, 4);
            off += 8;
            if (!k) break;
        }
        if (off < len) memset(b + off, 0, len - off);
    }
    for (NSArray *pt in op[@"patch"] ?: @[]) {
        uint64_t at = num(pt[0]);
        uint32_t v = (uint32_t)num(pt[1]);
        if (at + 4 > len) fatal_scenario(@"willResume: patch past the blob");
        memcpy(b + at, &v, 4);
    }
    if (op[@"truncate"]) {
        if (num(op[@"truncate"]) > len) fatal_scenario(@"willResume: truncate past the blob");
        m.length = (NSUInteger)num(op[@"truncate"]);
    }
    return m;
}

static void op_api(NSDictionary *op) {
    NSString *name = op[@"name"];
    NSMutableDictionary *o = [@{@"ev" : @"api", @"name" : name} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    uint64_t t = now_ns();
    // The device calls need a device: a scenario never makes them after a
    // release or a refusal.
    BOOL devCall = [@[ @"reset", @"pause", @"unpause", @"stop", @"willSuspend", @"finishSuspend", @"willResume", @"didResume" ]
        containsObject:name];
    if (devCall && !g_dev) fatal_scenario([@"no device for api " stringByAppendingString:name]);
    if ([name isEqual:@"reset"]) LIBV(rvg_device_reset(g_dev));
    else if ([name isEqual:@"pause"]) LIBV(rvg_device_pause(g_dev));
    else if ([name isEqual:@"unpause"]) LIBV(rvg_device_unpause(g_dev));
    else if ([name isEqual:@"stop"]) LIBV(rvg_device_stop(g_dev));
    else if ([name isEqual:@"willSuspend"]) LIBV(rvg_device_will_suspend(g_dev));
    else if ([name isEqual:@"finishSuspend"]) {
        RvgBytes b = {NULL, 0};
        RvgStatus st = LIB(rvg_device_finish_suspend(g_dev, &b));
        NSData *blob = nil;
        // The slot keeps the library's bytes, which are freed with it.
        if (st == RVG_OK)
            blob = [[NSData alloc] initWithBytesNoCopy:b.data
                                                length:b.length
                                           deallocator:^(void *bytes, NSUInteger length) {
                                             rvg_bytes_free((RvgBytes){bytes, length});
                                           }];
        if (blob) g_blobs[op[@"slot"] ?: @"s"] = blob;
        o[@"ok"] = @(blob != nil);
        if (blob) o[@"blob_pgsd"] = to_hex(blob.bytes, blob.length);
    } else if ([name isEqual:@"willResume"]) {
        NSData *blob = op[@"hex"] ? from_hex(op[@"hex"]) : g_blobs[op[@"slot"] ?: @"s"];
        if (op[@"patch"] || op[@"truncate"] || [op[@"canonical"] boolValue]) blob = mutate_blob(blob, op, !op[@"hex"]);
        // nil: a host that passes no state at all (NULL); an empty state is a
        // pointer the length 0 leaves unread.
        static const uint8_t empty[1];
        BOOL nilState = [op[@"nil"] boolValue];
        const uint8_t *state = nilState ? NULL : (blob.length ? blob.bytes : empty);
        size_t length = nilState ? 0 : blob.length;
        // nil_error: a host that passes no error pointer.
        RvgResumeError err = RVG_RESUME_ERROR_NONE;
        BOOL withError = ![op[@"nil_error"] boolValue];
        RvgStatus st = LIB(rvg_device_will_resume(g_dev, state, length, withError ? &err : NULL));
        // A bool, as the SDK's willResume answered: JSON's true or false.
        o[@"ok"] = @((bool)(st == RVG_OK));
        if (withError && err != RVG_RESUME_ERROR_NONE) o[@"error"] = @{@"code" : @(err), @"d_desc" : status_message(st)};
    } else if ([name isEqual:@"didResume"]) LIBV(rvg_device_did_resume(g_dev));
    else if ([name isEqual:@"release"]) release_all();
    else if ([name isEqual:@"recreate"]) {
        // A migration target: a fresh device over the same guest RAM, with
        // the scenario's displays, as QEMU creates one.
        release_all();
        create_device(op[@"config"] ?: g_config);
    } else if ([name isEqual:@"new_display"]) {
        new_display(op[@"spec"]);
        return;
    } else if ([name isEqual:@"release_display"]) {
        NSUInteger i = (NSUInteger)num(op[@"display"]);
        if (i < g_ndisplays && g_displays[i]) {
            LIBV(rvg_display_release(g_displays[i]));
            g_displays[i] = NULL;
        }
    } else if ([name isEqual:@"set_modes"]) {
        // nil: no modes at all, which changes nothing, as an empty list.
        RvgDisplay *d = display_at(op);
        NSArray *modes = [op[@"nil"] boolValue] ? @[] : (op[@"modes"] ?: @[]);
        if (d) set_modes(d, modes);
        o[@"modes_valid"] = @([(op[@"modes"] ?: @[]) count]);
    } else if ([name isEqual:@"display_props"]) {
        // What the host can read of a display. A released or refused display
        // answers nothing.
        RvgDisplay *d = display_at(op);
        if (!d) o[@"ok"] = @NO;
        else {
            RvgDisplayCoord c = LIB(rvg_display_cursor_position(d));
            uint64_t guest = 0, host = 0;
            LIBV(rvg_display_present_counts(d, &guest, &host));
            [o addEntriesFromDictionary:@{
                @"guestPresentCount" : @(guest),
                @"hostPresentCount" : @(host),
                @"cursorPosition" : @[ @(c.x), @(c.y) ],
            }];
        }
    } else if ([name isEqual:@"cursorPosition"]) {
        // A released display answers {0, 0}.
        RvgDisplay *d = display_at(op);
        RvgDisplayCoord c = {0, 0};
        if (d) c = LIB(rvg_display_cursor_position(d));
        o[@"x"] = @(c.x);
        o[@"y"] = @(c.y);
    } else if ([name isEqual:@"present_counts"]) {
        // A released display answers 0 and 0.
        RvgDisplay *d = display_at(op);
        uint64_t guest = 0, host = 0;
        if (d) LIBV(rvg_display_present_counts(d, &guest, &host));
        o[@"guest"] = @(guest);
        o[@"host"] = @(host);
    } else if ([name isEqual:@"maxPorts"]) {
        o[@"value"] = @(RVG_MAX_DISPLAY_PORTS);
    } else if ([name isEqual:@"iosfc_again"]) {
        // A second IOSurface host device in the same process, as QEMU's
        // apple-gfx-mmio makes one, released at once; no_map: without a
        // memory map, which it refuses.
        RvgIOSurfaceDevice *d = NULL;
        new_iosfc([op[@"no_map"] boolValue], &d);
        o[@"ok"] = @(d != NULL);
        if (d) LIBV(rvg_iosurface_device_release(d));
    } else {
        fatal_scenario([@"unknown api " stringByAppendingString:name]);
    }
    o[@"t_us"] = @((now_ns() - t) / 1000);
    emit(o);
}

// ---- encoding a frame, as QEMU does ----

typedef struct {
    __unsafe_unretained NSDictionary *op;
    __unsafe_unretained NSMutableDictionary *out;
    dispatch_semaphore_t done;
} encode_job;

// A frame encoded with hold, uncommitted, with its target and op, until an
// encode_commit commits it. Holding a frame in flight is how a host makes
// the display's presents wait for it deterministically.
@interface PGRunHeld : NSObject
@property RvgFrame *frame;
@property RvgFrameTarget *target;
@property uint32_t w, h;
@property(strong) NSDictionary *op;
@end
@implementation PGRunHeld
@end

static NSMutableArray<PGRunHeld *> *g_held;

// Commits the frame, waits for it, and reads the whole target back, w * 4
// bytes a row; then releases the frame and the target.
static void encode_finish(RvgFrame *frame, RvgFrameTarget *target, uint32_t w, uint32_t h, NSDictionary *op,
                          NSMutableDictionary *out) {
    RvgStatus st = LIB(rvg_frame_commit(frame));
    if (st == RVG_OK) st = LIB(rvg_frame_wait(frame));
    out[@"status"] = status_name(st);
    if (st != RVG_OK) out[@"d_error"] = status_message(st);
    size_t bpp = 4, bpr = (size_t)w * bpp, n = bpr * h;
    NSMutableData *px = [NSMutableData dataWithLength:n];
    RvgStatus rs = LIB(rvg_frame_target_read_pixels(target, px.mutableBytes, bpr, n));
    if (rs != RVG_OK) fatal_scenario([@"encode: read_pixels refused: " stringByAppendingString:status_name(rs)]);
    LIBV(rvg_frame_release(frame));
    LIBV(rvg_frame_target_release(target));
    out[@"sha256"] = sha256(px.bytes, n);
    if (n <= 4096) out[@"hex"] = to_hex(px.bytes, n);
    NSMutableArray *samples = [NSMutableArray array];
    for (NSArray *pt in op[@"samples"] ?: @[]) {
        size_t x = num(pt[0]), y = num(pt[1]);
        if (x < w && y < h) [samples addObject:to_hex((uint8_t *)px.bytes + y * bpr + x * bpp, bpp)];
    }
    if (samples.count) out[@"samples"] = samples;
}

static void encode_do(void *p) {
    encode_job *j = p;
    t_runner = 1;
    atomic_store(&g_callThread, pthread_self());
    @autoreleasepool {
        NSDictionary *op = j->op;
        RvgDisplay *d = display_at(op);
        uint32_t w = (uint32_t)num(op[@"w"] ?: @64), h = (uint32_t)num(op[@"h"] ?: @32);
        RvgFrameTargetConfig tc = {
            .struct_size = sizeof(tc),
            .width = w,
            .height = h,
            .format = (RvgPixelFormat)num(op[@"format"] ?: @(RVG_PIXEL_FORMAT_BGRA8_UNORM)),
        };
        RvgFrameTarget *target = NULL;
        RvgFrame *frame = NULL;
        uint64_t t = now_ns();
        RvgStatus st = LIB(rvg_frame_target_create(g_dev, &tc, &target));
        if (st == RVG_OK) {
            NSArray *rg = op[@"region"] ?: @[ @0, @0, @(w), @(h) ];
            RvgRegion region = {num(rg[0]), num(rg[1]), num(rg[2]), num(rg[3])};
            st = LIB(rvg_display_encode_frame(d, target, &region, &frame));
        }
        BOOL ok = st == RVG_OK;
        j->out[@"ok"] = @(ok);
        j->out[@"t_encode_us"] = @((now_ns() - t) / 1000);
        if (!ok) {
            j->out[@"status"] = status_name(st);
            j->out[@"d_error"] = status_message(st);
            LIBV(rvg_frame_target_release(target));
        } else if ([op[@"hold"] boolValue]) {
            if (!g_held) g_held = [NSMutableArray array];
            j->out[@"held"] = @(g_held.count);
            PGRunHeld *hf = [PGRunHeld new];
            hf.frame = frame;
            hf.target = target;
            hf.w = w;
            hf.h = h;
            hf.op = op;
            [g_held addObject:hf];
        } else if ([op[@"no_commit"] boolValue]) {
            // Dropped uncommitted: the display counts it in flight for its
            // life.
            LIBV(rvg_frame_release(frame));
            LIBV(rvg_frame_target_release(target));
        } else {
            encode_finish(frame, target, w, h, op, j->out);
        }
    }
    atomic_store(&g_callThread, (pthread_t)NULL);
    t_runner = 0;
    dispatch_semaphore_signal(j->done);
}

static void encode_commit_do(void *p) {
    encode_job *j = p;
    t_runner = 1;
    atomic_store(&g_callThread, pthread_self());
    @autoreleasepool {
        PGRunHeld *hf = g_held[num(j->op[@"held"])];
        RvgFrame *frame = hf.frame;
        RvgFrameTarget *target = hf.target;
        hf.frame = NULL;
        hf.target = NULL;
        encode_finish(frame, target, hf.w, hf.h, hf.op, j->out);
    }
    atomic_store(&g_callThread, (pthread_t)NULL);
    t_runner = 0;
    dispatch_semaphore_signal(j->done);
}

// Commits a frame encode held, on a pool thread as QEMU would, and reports it.
static void op_encode_commit(NSDictionary *op) {
    NSMutableDictionary *o = [@{@"ev" : @"encode_commit", @"held" : op[@"held"]} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    if (!g_held || num(op[@"held"]) >= g_held.count || !g_held[num(op[@"held"])].frame)
        fatal_scenario(@"encode_commit: nothing held there");
    encode_job j = {op, o, dispatch_semaphore_create(0)};
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), &j, encode_commit_do);
    dispatch_semaphore_wait(j.done, DISPATCH_TIME_FOREVER);
    emit(o);
}

static void op_encode(NSDictionary *op) {
    NSMutableDictionary *o = [@{@"ev" : @"encode"} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    if (op[@"usage"] || op[@"other_device"]) fatal_scenario(@"encode: the library makes the frame target");
    if (!display_at(op)) fatal_scenario(@"encode: no such display");
    encode_job j = {op, o, dispatch_semaphore_create(0)};
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), &j, encode_do);
    dispatch_semaphore_wait(j.done, DISPATCH_TIME_FOREVER);
    emit(o);
}

// ---- values the host assigns, computed from Metal (docs/testing.md 5.8) ----

static void op_metal(NSDictionary *op) {
    NSString *q = op[@"query"];
    NSMutableDictionary *o = [@{@"ev" : @"metal", @"query" : q} mutableCopy];
    if (op[@"label"]) o[@"label"] = op[@"label"];
    if ([q isEqual:@"heapTextureSizeAndAlign"]) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor new];
        td.textureType = (MTLTextureType)num(op[@"type"] ?: @(MTLTextureType2D));
        td.pixelFormat = (MTLPixelFormat)num(op[@"format"]);
        td.width = num(op[@"w"]);
        td.height = num(op[@"h"] ?: @1);
        td.depth = num(op[@"d"] ?: @1);
        td.mipmapLevelCount = num(op[@"mips"] ?: @1);
        td.sampleCount = num(op[@"samples"] ?: @1);
        td.arrayLength = num(op[@"array"] ?: @1);
        td.usage = (MTLTextureUsage)num(op[@"usage"] ?: @1);
        td.resourceOptions = (MTLResourceOptions)num(op[@"options"] ?: @(MTLResourceStorageModePrivate));
        if (op[@"allow_gpu_optimized"]) td.allowGPUOptimizedContents = [op[@"allow_gpu_optimized"] boolValue];
        MTLSizeAndAlign sa = [g_mtl heapTextureSizeAndAlignWithDescriptor:td];
        o[@"size"] = hex64(sa.size);
        o[@"align"] = hex64(sa.align);
    } else if ([q isEqual:@"minimumLinearTextureAlignment"]) {
        o[@"value"] = @([g_mtl minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)num(op[@"format"])]);
    } else {
        fatal_scenario([@"unknown metal query " stringByAppendingString:q]);
    }
    emit(o);
}

// ---- the scenario thread ----

static NSArray *g_ops;

static void teardown_and_exit(int64_t index) {
    atomic_store(&g_op, index);
    op_events(@{@"label" : @"end"});
    uint64_t t = now_ns();
    // Tear down as a host does on exit: the displays, then the device, then
    // the IOSurface host device.
    release_all();
    if (g_iosfc) LIBV(rvg_iosurface_device_release(g_iosfc));
    g_iosfc = NULL;
    emit(@{@"ev" : @"teardown", @"t_us" : @((now_ns() - t) / 1000)});
    emit(@{@"ev" : @"end"});
    _exit(0);
}

static void *scenario_main(void *arg) {
    t_runner = 1;
    @autoreleasepool {
        int64_t index = 0;
        unsigned watchdog = 30;
        for (NSDictionary *op in g_ops) {
            atomic_store(&g_op, index++);
            NSString *kind = op[@"op"];
            BOOL guarded = [kind isEqual:@"api"] || [kind isEqual:@"encode"] || [kind isEqual:@"encode_commit"] ||
                           [kind isEqual:@"trace_fire"];
            // willSuspend drops any interrupt not yet raised: the device raises
            // one only while it runs, and a stamp's signal writes the stamp
            // before it raises the interrupt (reims-vgpu-core's
            // DeviceCore::signal_stamp), so a guest that has seen its stamp
            // can meet a willSuspend before the stamp's interrupt. Whether it
            // did is timing: the call first settles, as a checkpoint does, so
            // that the interrupts of what the guest has seen are raised
            // (no_settle: true skips it).
            if ([kind isEqual:@"api"] && [op[@"name"] isEqual:@"willSuspend"] && ![op[@"no_settle"] boolValue])
                quiesce(40, 3000);
            if (guarded) atomic_store(&g_recordPending, 1);
            @autoreleasepool {
                if ([kind isEqual:@"config"]) {
                    if (op[@"watchdog_s"]) watchdog = (unsigned)num(op[@"watchdog_s"]);
                    alarm(watchdog);  // covers teardown too: CmdDelay hangs there
                    op_config(op);
                } else if ([kind isEqual:@"mem"]) op_mem(op);
                else if ([kind isEqual:@"mem_seq"]) op_mem_seq(op);
                else if ([kind isEqual:@"fill"]) op_fill(op);
                else if ([kind isEqual:@"mem_file"]) op_mem_file(op);
                else if ([kind isEqual:@"mmio_w"]) op_mmio_w(op, NO);
                else if ([kind isEqual:@"mmio_r"]) op_mmio_r(op, NO);
                else if ([kind isEqual:@"mmio_sweep"]) op_mmio_sweep(op, NO);
                else if ([kind isEqual:@"iosfc_w"]) op_mmio_w(op, YES);
                else if ([kind isEqual:@"iosfc_r"]) op_mmio_r(op, YES);
                else if ([kind isEqual:@"iosfc_sweep"]) op_mmio_sweep(op, YES);
                else if ([kind isEqual:@"wait"]) op_wait(op);
                else if ([kind isEqual:@"diff"]) op_diff(op);
                else if ([kind isEqual:@"dump"]) op_dump(op);
                else if ([kind isEqual:@"cmp"]) op_cmp(op);
                else if ([kind isEqual:@"events"]) op_events(op);
                else if ([kind isEqual:@"quiesce"]) quiesce(op[@"quiet_ms"] ? num(op[@"quiet_ms"]) : 40, num(op[@"timeout_ms"] ?: @3000));
                else if ([kind isEqual:@"api"]) op_api(op);
                else if ([kind isEqual:@"encode"]) op_encode(op);
                else if ([kind isEqual:@"encode_commit"]) op_encode_commit(op);
                else if ([kind isEqual:@"metal"]) op_metal(op);
                else if ([kind isEqual:@"trace_fire"]) op_trace_fire(op);
                else if ([kind isEqual:@"handoff_save"]) op_handoff_save(op);
                else if ([kind isEqual:@"mark"]) emit(@{@"ev" : @"mark", @"label" : op[@"label"] ?: @""});
                else fatal_scenario([@"unknown op " stringByAppendingString:kind ?: @"(none)"]);
            }
            if (guarded) atomic_store(&g_recordPending, 0);
            if ([kind isEqual:@"config"] && !g_dev) {
                // A refused device ends the scenario: nothing else can run.
                emit(@{@"ev" : @"end"});
                _exit(0);
            }
        }
        teardown_and_exit(index);
    }
    return NULL;
}

int main(int argc, const char *argv[]) {
    t_runner = 1;
    @autoreleasepool {
        const char *scenario = NULL, *out = NULL, *setup = "map";
        long phase = -1;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--setup") && i + 1 < argc) setup = argv[++i];
            else if (!strcmp(argv[i], "--fixtures") && i + 1 < argc) g_fixtures = @(argv[++i]);
            else if (!strcmp(argv[i], "--phase") && i + 1 < argc) phase = atol(argv[++i]);
            else if (!strcmp(argv[i], "--handoff") && i + 1 < argc) g_handoff = @(argv[++i]);
            else if (!strcmp(argv[i], "--build") && i + 1 < argc) {
                char real[PATH_MAX];
                const char *b = argv[++i];
                g_build = @(realpath(b, real) ? real : b);
            }
            else if (!scenario) scenario = argv[i];
            else if (!out) out = argv[i];
        }
        if (!scenario || !out) {
            fprintf(stderr, "usage: pgrun SCENARIO.jsonl OUT.jsonl --setup blocks|map|isolate [--fixtures DIR]"
                            " [--phase K --handoff DIR] [--build DIR]\n");
            return 64;
        }
        g_setup = @(setup);
        if (![g_setup isEqual:@"blocks"] && ![g_setup isEqual:@"map"] && ![g_setup isEqual:@"isolate"]) return 64;
        g_out_fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (g_out_fd < 0) {
            perror(out);
            return 66;
        }
        g_t0 = now_ns();
        g_events = [NSMutableArray array];
        g_traces = [NSMutableArray array];
        g_blobs = [NSMutableDictionary dictionary];
        const int sigs[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGFPE};
        for (size_t i = 0; i < sizeof(sigs) / sizeof(sigs[0]); i++) signal(sigs[i], on_signal);
        signal(SIGALRM, on_alarm);

        // Hidden inputs are pinned (docs/testing.md, section 3.2): the
        // preferences domain must not exist.
        CFArrayRef keys = CFPreferencesCopyKeyList(CFSTR("com.apple.gpusw.ParavirtualizedGraphics"), kCFPreferencesCurrentUser,
                                                   kCFPreferencesAnyHost);
        CFArrayRef keys2 = CFPreferencesCopyKeyList(CFSTR("com.apple.gpusw.ParavirtualizedGraphics"), kCFPreferencesAnyUser,
                                                    kCFPreferencesCurrentHost);
        if ((keys && CFArrayGetCount(keys)) || (keys2 && CFArrayGetCount(keys2))) {
            fprintf(stderr, "pgrun: the com.apple.gpusw.ParavirtualizedGraphics preferences domain is set; the suite needs it absent\n");
            return 67;
        }
        if (getenv("MTL_DEBUG_LAYER")) {
            fprintf(stderr, "pgrun: MTL_DEBUG_LAYER is set; the suite pins it unset\n");
            return 67;
        }

        NSString *text = [NSString stringWithContentsOfFile:@(scenario) encoding:NSUTF8StringEncoding error:nil];
        NSMutableArray *ops = [NSMutableArray array];
        for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
            if (!line.length || [line hasPrefix:@"#"]) continue;
            NSDictionary *op = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
            if (!op) {
                fprintf(stderr, "pgrun: bad line: %s\n", line.UTF8String);
                return 65;
            }
            [ops addObject:op];
        }
        // The phases: the ops between {"op": "phase"} markers. A scenario
        // with more than one runs a phase a process, named by --phase.
        NSMutableArray *phases = [NSMutableArray arrayWithObject:[NSMutableArray array]];
        for (NSDictionary *op in ops) {
            if ([op[@"op"] isEqual:@"phase"]) [phases addObject:[NSMutableArray array]];
            else [phases.lastObject addObject:op];
        }
        if (phases.count > 1 && phase < 0) {
            fprintf(stderr, "pgrun: the scenario has %lu phases: name one with --phase\n", (unsigned long)phases.count);
            return 64;
        }
        if (phase >= (long)phases.count) {
            fprintf(stderr, "pgrun: no phase %ld\n", phase);
            return 64;
        }
        g_ops = phases[phase < 0 ? 0 : phase];
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 << 20);
        pthread_t th;
        pthread_create(&th, &attr, scenario_main, NULL);
        g_scenarioThread = th;
    }
    // The main thread only parks: every callback arrives on the library's
    // threads, and one on this thread breaks a thread rule.
    dispatch_main();
}

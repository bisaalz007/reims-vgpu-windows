// The step kernel of the adapter's command predication (gpu/predication.rs), which build.rs
// compiles into predication.metallib. One thread runs one step: it reads and writes the command
// buffer's skip counter, and writes the arguments of the indirect form a predicated dispatch or
// indirect command buffer execution runs as.
//
// The counter is 0 while commands run and above 0 while they are skipped; a skipped block's
// nested blocks count on from it, so each end steps it back one.
#include <metal_stdlib>
using namespace metal;

// One step, written by the CPU when the adapter encodes it. Its layout is `Step` in
// predication.rs.
struct Step {
    // What the step does (below), and RESET.
    uint op;
    // The comparison of a condition: 0 ==, 1 !=, 2 >, 3 >=, 4 <, 5 <=.
    uint comparison;
    // The value the guest's word is compared with, on the right.
    uint reference;
    // The word of the state buffer an overrun sets, past the counter.
    uint flag;
    // The words of `args` a skipped step writes as 0.
    uint mask;
    // How many words of arguments it writes.
    uint count;
    uint unused[2];
    // The arguments of a direct dispatch or execution.
    uint args[8];
    // The arguments the indirect form reads.
    uint out[8];
};

// The step's counter is 0 before it: the first step of a unit of compute encoders.
constant uint RESET = 0x100;

// An If: inside a skipped block one more level, else skipped from here when the condition fails.
constant uint IF = 1;
// An Else: what ran is skipped, and what was skipped at this level runs.
constant uint ELSE = 2;
// The end of a block: one level less.
constant uint END = 3;
// A loop's start inside a skipped block: one more level, as an If's.
constant uint ENTER = 4;
// A loop's condition before (While) or after (DoWhile) an iteration: while running, a condition
// that fails skips the loop's iterations from here.
constant uint CHECK = 5;
// After a loop's last laid-down iteration, a loop still running sets its flag.
constant uint OVERRUN = 6;
// The arguments of `args`, their masked words 0 while skipped.
constant uint ARGS = 7;
// The guest's arguments, their masked words 0 while skipped.
constant uint GUEST_ARGS = 8;

// Whether `word OP reference` holds, as Metal compares a predication word.
static bool holds(uint word, uint comparison, uint reference) {
    switch (comparison) {
    case 1:
        return word != reference;
    case 2:
        return word > reference;
    case 3:
        return word >= reference;
    case 4:
        return word < reference;
    case 5:
        return word <= reference;
    default:
        return word == reference;
    }
}

kernel void predication_step(device uint *state [[buffer(0)]],
                             device Step &step [[buffer(1)]],
                             device const uint *guest [[buffer(2)]]) {
    uint c = (step.op & RESET) != 0 ? 0 : state[0];
    switch (step.op & 0xff) {
    case IF:
        c = c != 0 ? c + 1 : (holds(guest[0], step.comparison, step.reference) ? 0 : 1);
        break;
    case ELSE:
        c = c == 0 ? 1 : (c == 1 ? 0 : c);
        break;
    case END:
        c = c != 0 ? c - 1 : 0;
        break;
    case ENTER:
        c = c != 0 ? c + 1 : 0;
        break;
    case CHECK:
        if (c == 0 && !holds(guest[0], step.comparison, step.reference)) {
            c = 1;
        }
        break;
    case OVERRUN:
        if (c == 0) {
            state[1 + step.flag] = 1;
        }
        break;
    case ARGS:
    case GUEST_ARGS:
        for (uint i = 0; i < step.count && i < 8; i++) {
            uint v = (step.op & 0xff) == ARGS ? step.args[i] : guest[i];
            step.out[i] = (c != 0 && ((step.mask >> i) & 1) != 0) ? 0 : v;
        }
        break;
    }
    state[0] = c;
}

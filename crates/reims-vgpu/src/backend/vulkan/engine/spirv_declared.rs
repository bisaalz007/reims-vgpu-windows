//! Descriptor slots a SPIR-V module declares.
//!
//! The engine builds its descriptor-set layout from the resources a request
//! provides. A guest draw may name fewer than the shader reads — Metal answers
//! such a slot with zeros, so the guest issues it freely — and the shader then
//! indexes a descriptor the set never declared. That is undefined in Vulkan and
//! the ICDs disagree loudly: on the NVIDIA rail this device runs it returns
//! garbage pixels. Scanning the module tells the layout builder which slots
//! must exist regardless of what the request carried.
//!
//! # Why the module is scanned rather than the reflection consulted
//!
//! `metal2vulkan`'s reflection is the *Metal* side of the crossing: it names
//! the argument indices the AIR declares. What the layout needs is the *Vulkan*
//! side — the `set`/`binding` decorations the translated module actually
//! carries — and those are produced by the translator's own relocation, after
//! reflection. A slot the reflection knows about but the relocation placed
//! elsewhere would be declared at the wrong binding. The module is the only
//! statement of what the shader will index at run time, so it is what is read.
//! Reflection still decides *meaning* everywhere else; this answers one narrow
//! question the module alone can answer.

use std::collections::HashMap;

const OP_DECORATE: u16 = 71;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;

const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

/// `OpTypeImage` `Dim` operand for an input attachment (SPIR-V spec, `Dim`
/// enum). Such an image is *not* a sampled texture: the engine binds it from
/// the request's own colour-input state as `INPUT_ATTACHMENT`, and declaring it
/// here as well would put two different descriptor types on one binding.
const DIM_SUBPASS_DATA: u32 = 6;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_HEADER_WORDS: usize = 5;

/// What a declared slot needs from the layout.
///
/// Storage buffers are deliberately absent: every path that has them derives
/// them from the request, and a *writable* resource must never be satisfied
/// with a shared dummy — a shader storing into one would corrupt whatever else
/// was pointed at it. Only the read-only kinds a draw legitimately omits are
/// reported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeclaredKind {
    SampledImage,
    Sampler,
}

/// `(set, binding, kind)` for every image or sampler variable the module
/// decorates with both a set and a binding. Returns nothing for a module that
/// is not valid SPIR-V rather than guessing — a caller that cannot parse the
/// shader has no business editing the layout it implies.
pub(crate) fn declared_descriptors(words: &[u32]) -> Vec<(u32, u32, DeclaredKind)> {
    if words.len() < SPIRV_HEADER_WORDS || words[0] != SPIRV_MAGIC {
        return Vec::new();
    }
    let mut binding_of: HashMap<u32, u32> = HashMap::new();
    let mut set_of: HashMap<u32, u32> = HashMap::new();
    let mut image_types: Vec<u32> = Vec::new();
    let mut sampler_types: Vec<u32> = Vec::new();
    // pointer result id -> pointee type id
    let mut pointee_of: HashMap<u32, u32> = HashMap::new();
    // variable result id -> its pointer type id
    let mut variables: Vec<(u32, u32)> = Vec::new();

    let mut i = SPIRV_HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        // A zero word count cannot advance the cursor: stop rather than spin.
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        let operands = &words[i + 1..i + word_count];
        match opcode {
            OP_DECORATE if operands.len() >= 3 => match operands[1] {
                DECORATION_BINDING => {
                    binding_of.insert(operands[0], operands[2]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    set_of.insert(operands[0], operands[2]);
                }
                _ => {}
            },
            // `Dim` is operand 2 of `OpTypeImage` (result id, sampled type,
            // Dim, ...). `SubpassData` is an input attachment and belongs to
            // the colour-input rail, not this one.
            OP_TYPE_IMAGE if operands.len() >= 3 && operands[2] != DIM_SUBPASS_DATA => {
                image_types.push(operands[0]);
            }
            OP_TYPE_SAMPLED_IMAGE if !operands.is_empty() => {
                image_types.push(operands[0]);
            }
            OP_TYPE_SAMPLER if !operands.is_empty() => {
                sampler_types.push(operands[0]);
            }
            OP_TYPE_POINTER if operands.len() >= 3 => {
                pointee_of.insert(operands[0], operands[2]);
            }
            OP_VARIABLE if operands.len() >= 2 => {
                variables.push((operands[1], operands[0]));
            }
            _ => {}
        }
        i += word_count;
    }

    let mut out = Vec::new();
    for (var_id, ptr_type) in variables {
        let (Some(binding), Some(set)) = (binding_of.get(&var_id), set_of.get(&var_id)) else {
            continue;
        };
        let Some(pointee) = pointee_of.get(&ptr_type) else {
            continue;
        };
        let kind = if image_types.contains(pointee) {
            DeclaredKind::SampledImage
        } else if sampler_types.contains(pointee) {
            DeclaredKind::Sampler
        } else {
            continue;
        };
        out.push((*set, *binding, kind));
    }
    out
}

/// [`declared_descriptors`], memoised on the module's content digest.
///
/// Liquid Glass is a chain of many small passes, so the same handful of modules
/// is scanned on every draw of every frame; the scan is O(module words) and the
/// answer is a pure function of the words. The digest is the key rather than a
/// pipeline id because a guest pipeline ref is reused across content — keying
/// on the ref would serve the previous shader's slots for the new one.
pub(crate) fn declared_descriptors_cached(
    words: &[u32],
) -> std::sync::Arc<Vec<(u32, u32, DeclaredKind)>> {
    use std::collections::HashMap as Map;
    use std::sync::{Mutex, OnceLock};
    type Cache = Map<super::digest::Digest128, std::sync::Arc<Vec<(u32, u32, DeclaredKind)>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    /// Distinct translated modules a session sees is in the low hundreds; the
    /// cap is a runaway guard, not a working-set bound. On overflow the cache
    /// is cleared rather than evicted one entry at a time — the scan is cheap
    /// enough that a rare full refill costs less than tracking LRU here.
    const CACHE_CAP: usize = 4096;

    let key = super::digest::Digest128::of_u32_words(words);
    let cache = CACHE.get_or_init(|| Mutex::new(Map::new()));
    let Ok(mut map) = cache.lock() else {
        return std::sync::Arc::new(declared_descriptors(words));
    };
    if let Some(hit) = map.get(&key) {
        note_scan(true);
        return hit.clone();
    }
    note_scan(false);
    let scanned = std::sync::Arc::new(declared_descriptors(words));
    if map.len() >= CACHE_CAP {
        map.clear();
    }
    map.insert(key, scanned.clone());
    scanned
}

/// Cumulative `(hits, misses)` for the declared-slot scan cache.
static SCAN_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCAN_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn note_scan(hit: bool) {
    let counter = if hit { &SCAN_HITS } else { &SCAN_MISSES };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// `(hits, misses)` of the declared-slot scan cache, for the per-second census.
pub(crate) fn scan_stats() -> (u64, u64) {
    (
        SCAN_HITS.load(std::sync::atomic::Ordering::Relaxed),
        SCAN_MISSES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal module: header, then the words given.
    fn module(body: &[u32]) -> Vec<u32> {
        let mut w = vec![SPIRV_MAGIC, 0x0001_0000, 0, 1, 0];
        w.extend_from_slice(body);
        w
    }

    fn op(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let count = (operands.len() + 1) as u32;
        let mut w = vec![(count << 16) | u32::from(opcode)];
        w.extend_from_slice(operands);
        w
    }

    /// Ids used by the fixtures: 10 = image type, 11 = pointer to it,
    /// 12 = the variable, 20/21/22 the sampler equivalents.
    fn sampled_image_at(binding: u32) -> Vec<u32> {
        let mut b = Vec::new();
        // OpTypeImage %10 <sampled_type> Dim2D ...
        b.extend(op(OP_TYPE_IMAGE, &[10, 1, 1, 0, 0, 0, 1, 0]));
        b.extend(op(OP_TYPE_POINTER, &[11, 0, 10]));
        b.extend(op(OP_VARIABLE, &[11, 12, 0]));
        b.extend(op(OP_DECORATE, &[12, DECORATION_DESCRIPTOR_SET, 0]));
        b.extend(op(OP_DECORATE, &[12, DECORATION_BINDING, binding]));
        b
    }

    #[test]
    fn a_sampled_image_the_guest_never_bound_is_still_declared() {
        let m = module(&sampled_image_at(35));
        assert_eq!(
            declared_descriptors(&m),
            vec![(0, 35, DeclaredKind::SampledImage)]
        );
    }

    #[test]
    fn a_separate_sampler_is_declared_with_its_own_kind() {
        let mut b = Vec::new();
        b.extend(op(OP_TYPE_SAMPLER, &[20]));
        b.extend(op(OP_TYPE_POINTER, &[21, 0, 20]));
        b.extend(op(OP_VARIABLE, &[21, 22, 0]));
        b.extend(op(OP_DECORATE, &[22, DECORATION_DESCRIPTOR_SET, 0]));
        b.extend(op(OP_DECORATE, &[22, DECORATION_BINDING, 64]));
        assert_eq!(
            declared_descriptors(&module(&b)),
            vec![(0, 64, DeclaredKind::Sampler)]
        );
    }

    /// An input attachment is `OpTypeImage` with `Dim = SubpassData`. The
    /// engine binds it as `INPUT_ATTACHMENT` from the request's colour-input
    /// state; declaring it here too would put two descriptor types on one
    /// binding and fail pipeline creation.
    #[test]
    fn a_subpass_input_is_not_a_sampled_image() {
        let mut b = Vec::new();
        b.extend(op(OP_TYPE_IMAGE, &[10, 1, DIM_SUBPASS_DATA, 0, 0, 0, 2, 0]));
        b.extend(op(OP_TYPE_POINTER, &[11, 0, 10]));
        b.extend(op(OP_VARIABLE, &[11, 12, 0]));
        b.extend(op(OP_DECORATE, &[12, DECORATION_DESCRIPTOR_SET, 0]));
        b.extend(op(OP_DECORATE, &[12, DECORATION_BINDING, 96]));
        assert!(declared_descriptors(&module(&b)).is_empty());
    }

    /// A variable with only one of the two decorations names no slot: the
    /// layout is `(set, binding)` and half of that is not a location.
    #[test]
    fn a_variable_missing_either_decoration_declares_nothing() {
        let mut b = Vec::new();
        b.extend(op(OP_TYPE_IMAGE, &[10, 1, 1, 0, 0, 0, 1, 0]));
        b.extend(op(OP_TYPE_POINTER, &[11, 0, 10]));
        b.extend(op(OP_VARIABLE, &[11, 12, 0]));
        b.extend(op(OP_DECORATE, &[12, DECORATION_BINDING, 35]));
        assert!(declared_descriptors(&module(&b)).is_empty());
    }

    /// Not SPIR-V, truncated, or a malformed instruction stream: answer
    /// nothing rather than a guess, and above all do not loop forever on a
    /// zero word count.
    #[test]
    fn a_module_that_is_not_spirv_declares_nothing_and_terminates() {
        assert!(declared_descriptors(&[]).is_empty());
        assert!(declared_descriptors(&[0xdead_beef, 0, 0, 0, 0]).is_empty());
        // Zero word count mid-stream.
        let mut m = module(&[]);
        m.push(0);
        m.push(0x1234);
        assert!(declared_descriptors(&m).is_empty());
        // Word count that runs past the end.
        let mut m = module(&[]);
        m.push((99u32 << 16) | u32::from(OP_DECORATE));
        assert!(declared_descriptors(&m).is_empty());
    }

    /// The cache answers the same slots as a direct scan and reports a hit the
    /// second time the same module is asked about.
    #[test]
    fn the_scan_cache_is_keyed_on_content_and_reports_its_hits() {
        let m = module(&sampled_image_at(37));
        let direct = declared_descriptors(&m);
        let (h0, m0) = scan_stats();
        let first = declared_descriptors_cached(&m);
        let (h1, m1) = scan_stats();
        assert_eq!(first.as_ref(), &direct);
        assert_eq!((h1 - h0, m1 - m0), (0, 1), "first ask is a miss");
        let second = declared_descriptors_cached(&m);
        let (h2, m2) = scan_stats();
        assert_eq!(second.as_ref(), &direct);
        assert_eq!((h2 - h1, m2 - m1), (1, 0), "same content is a hit");
        // Different content is a different key, not a stale hit.
        let other = module(&sampled_image_at(38));
        let third = declared_descriptors_cached(&other);
        assert_eq!(third.as_ref(), &vec![(0, 38, DeclaredKind::SampledImage)]);
    }
}

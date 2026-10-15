//! The suspend blob: the device state a suspend hands the host, and a resume reads back.
//!
//! The format is this library's own. Every word is a little-endian `u32`:
//!
//! | offset | what |
//! |---|---|
//! | 0 | the magic, the bytes `RVGS` ([`SUSPEND_STATE_MAGIC`]) |
//! | 4 | the format version, 1 ([`SUSPEND_STATE_VERSION`]) |
//! | 8 | the sections, up to the end of the blob |
//!
//! A section is its tag, its payload's length in bytes, then the payload. Version 1 has two
//! sections, each exactly once, in either order:
//! - tag 1, the registers: [`SuspendDeviceState`]'s 20 words in field order (80 bytes);
//! - tag 2, the device info: the max key the guest last asked device info for (an `i32`), then
//!   the pairs the host wrote for that key, without their terminator: at most
//!   [`MAX_INFO_PAIRS`], each key one a resume reads (1-42 or 44).
//!
//! A resume refuses, with one log line and a resume error code (the C API's `RVG_RESUME_ERROR_*`):
//! - a blob shorter than its header, or whose magic is not ours: InvalidContent;
//! - a version other than 1: InvalidSuspendStateVersion;
//! - malformed sections: a section running past the end, an unknown tag, a section repeated or
//!   missing, a payload of the wrong length, or a device-info key a resume does not read:
//!   InvalidContent.
//!
//! Nothing in a blob can end the process, and every blob [`SuspendState::serialize`] writes
//! parses back to the same state (`tests::serialize::round_trip`).

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::num::ToUsize;
use crate::platform::types::NsErrorText;

/// The domain of the error a refused blob answers: a diagnostic, which no host reads (a host gets
/// the code alone).
pub const RESUME_ERROR_DOMAIN: &str = "PGResumeErrorDomain";
/// A version other than 1: the C API's `RVG_RESUME_ERROR_INVALID_SUSPEND_STATE_VERSION`.
pub const RESUME_ERROR_INVALID_SUSPEND_STATE_VERSION: i64 = 1;
/// A blob that is malformed, or absent: `RVG_RESUME_ERROR_INVALID_CONTENT`.
pub const RESUME_ERROR_INVALID_CONTENT: i64 = 2;

/// `APVKeyValuePair` (8 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct KeyValuePair {
    /// `key`.
    pub key: u32,
    /// `value`.
    pub value: u32,
}

const _: () = assert!(size_of::<KeyValuePair>() == 8);

/// The device's registers a suspend saves, in the order the registers section carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(missing_docs)] // the registers' names
pub struct SuspendDeviceState {
    pub root_page_number: u32,
    pub binary_version: u32,
    pub fifo_base_page: u32,
    pub fifo_enabled: u32,
    pub fifo_length: u32,
    pub fifo_written: u32,
    pub fifo_read: u32,
    pub fifo_start: u32,
    pub interrupt_display: u32,
    pub interrupt_stamp: u32,
    /// Whether the fault mask was nonzero. A resume clears the fault mask whatever it says.
    pub interrupt_fault: u32,
    pub efi_enabled: u32,
    pub efi_start: u32,
    pub efi_length: u32,
    pub efi_depth: u32,
    pub efi_mode: u32,
    pub efi_stride: u32,
    pub display_port_count: u32,
    pub fifo_fault_offset: u32,
    pub display_port_connection_types: u32,
}

/// The registers section's words.
const REGISTER_WORDS: usize = 20;

impl SuspendDeviceState {
    /// The registers in the section's order.
    fn to_words(self) -> [u32; REGISTER_WORDS] {
        [
            self.root_page_number,
            self.binary_version,
            self.fifo_base_page,
            self.fifo_enabled,
            self.fifo_length,
            self.fifo_written,
            self.fifo_read,
            self.fifo_start,
            self.interrupt_display,
            self.interrupt_stamp,
            self.interrupt_fault,
            self.efi_enabled,
            self.efi_start,
            self.efi_length,
            self.efi_depth,
            self.efi_mode,
            self.efi_stride,
            self.display_port_count,
            self.fifo_fault_offset,
            self.display_port_connection_types,
        ]
    }

    /// The registers from the section's words.
    fn from_words(w: [u32; REGISTER_WORDS]) -> Self {
        let [
            root_page_number,
            binary_version,
            fifo_base_page,
            fifo_enabled,
            fifo_length,
            fifo_written,
            fifo_read,
            fifo_start,
            interrupt_display,
            interrupt_stamp,
            interrupt_fault,
            efi_enabled,
            efi_start,
            efi_length,
            efi_depth,
            efi_mode,
            efi_stride,
            display_port_count,
            fifo_fault_offset,
            display_port_connection_types,
        ] = w;
        SuspendDeviceState {
            root_page_number,
            binary_version,
            fifo_base_page,
            fifo_enabled,
            fifo_length,
            fifo_written,
            fifo_read,
            fifo_start,
            interrupt_display,
            interrupt_stamp,
            interrupt_fault,
            efi_enabled,
            efi_start,
            efi_length,
            efi_depth,
            efi_mode,
            efi_stride,
            display_port_count,
            fifo_fault_offset,
            display_port_connection_types,
        }
    }
}

/// The blob's first word: the bytes `RVGS`. Name is ours.
pub const SUSPEND_STATE_MAGIC: u32 = u32::from_le_bytes(*b"RVGS");
/// The format version this library writes, and the only one it reads.
pub const SUSPEND_STATE_VERSION: u32 = 1;
/// The device info pairs the serializer's caller asks the writer for, the terminator included.
pub const SERIALIZED_INFO_PAIRS: usize = 46;
/// The most device info pairs a blob carries: the writer's pairs without their terminator.
pub const MAX_INFO_PAIRS: usize = 45;

/// The header's size: the magic and the version.
const HEADER_SIZE: usize = 8;
/// A section header's size: the tag and the payload's length.
const SECTION_HEADER_SIZE: usize = 8;

/// A section's tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    /// The registers.
    Registers,
    /// The device info max key and pairs.
    DeviceInfo,
}

impl TryFrom<u32> for Section {
    type Error = u32;

    fn try_from(tag: u32) -> Result<Self, u32> {
        match tag {
            1 => Ok(Section::Registers),
            2 => Ok(Section::DeviceInfo),
            other => Err(other),
        }
    }
}

impl Section {
    /// The tag as the blob carries it.
    fn tag(self) -> u32 {
        match self {
            Section::Registers => 1,
            Section::DeviceInfo => 2,
        }
    }
}

/// Whether a resume reads device info key `key`: 1-42 and 44 (43 has no reader, and 0 would end
/// the list).
fn readable_info_key(key: u32) -> bool {
    matches!(key, 1..=42 | 44)
}

/// The resume error `code` in [`RESUME_ERROR_DOMAIN`], with no user info. Only the code reaches
/// the host (`DeviceCore::will_resume` stores it); the domain and `description` are diagnostics.
fn resume_error(code: i64) -> NsErrorText {
    NsErrorText {
        domain: RESUME_ERROR_DOMAIN.to_owned(),
        code,
        description: format!("suspend state refused ({RESUME_ERROR_DOMAIN} code {code})"),
    }
}

/// The little-endian word at `offset` of `bytes`, if it holds one.
fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let b: [u8; 4] = bytes.get(offset..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(b))
}

/// What a suspend saves: the registers and the device info the guest last asked for.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SuspendState {
    /// The device's registers.
    pub registers: SuspendDeviceState,
    /// The max key the guest last asked device info for, as the device stores it.
    pub device_info_max_key: i32,
    /// The pairs the host wrote for that max key, without their terminator.
    pub device_info: Vec<KeyValuePair>,
}

impl SuspendState {
    /// The state with the device info pairs `written`, as the writer filled them: the pairs up to
    /// its terminator, at most [`MAX_INFO_PAIRS`].
    pub fn new(
        registers: SuspendDeviceState,
        device_info_max_key: i32,
        written: &[KeyValuePair],
    ) -> Self {
        let device_info = written
            .iter()
            .take_while(|p| p.key != 0)
            .take(MAX_INFO_PAIRS)
            .copied()
            .collect();
        SuspendState {
            registers,
            device_info_max_key,
            device_info,
        }
    }

    /// The blob: the header, the registers section, then the device info section, in one
    /// allocation of the blob's length.
    pub fn serialize(&self) -> Vec<u8> {
        let pairs = self
            .device_info
            .get(..MAX_INFO_PAIRS)
            .unwrap_or(&self.device_info);
        let registers = self.registers.to_words();
        let registers_len = size_of_val(&registers);
        let info_len = 4usize.saturating_add(pairs.len().saturating_mul(8));
        let header = [
            SUSPEND_STATE_MAGIC,
            SUSPEND_STATE_VERSION,
            Section::Registers.tag(),
            u32::try_from(registers_len).unwrap_or(u32::MAX),
        ];
        let info_header = [
            Section::DeviceInfo.tag(),
            u32::try_from(info_len).unwrap_or(u32::MAX),
            self.device_info_max_key.cast_unsigned(),
        ];
        let len = HEADER_SIZE
            .saturating_add(SECTION_HEADER_SIZE.saturating_mul(2))
            .saturating_add(registers_len)
            .saturating_add(info_len);
        let mut blob = Vec::with_capacity(len);
        let words = header
            .into_iter()
            .chain(registers)
            .chain(info_header)
            .chain(pairs.iter().flat_map(|p| [p.key, p.value]));
        for word in words {
            blob.extend_from_slice(&word.to_le_bytes());
        }
        blob
    }

    /// The state a blob carries, or the refusal a resume answers with. Each refusal logs one
    /// warning.
    pub fn parse(bytes: &[u8]) -> Result<Self, NsErrorText> {
        let invalid = |why: std::fmt::Arguments<'_>| {
            log::warn!("suspend state refused: {why}");
            Err(resume_error(RESUME_ERROR_INVALID_CONTENT))
        };
        let (Some(magic), Some(version)) = (word(bytes, 0), word(bytes, 4)) else {
            return invalid(format_args!(
                "{} bytes are shorter than its header",
                bytes.len()
            ));
        };
        if magic != SUSPEND_STATE_MAGIC {
            return invalid(format_args!("magic {magic:#010x} is not ours"));
        }
        if version != SUSPEND_STATE_VERSION {
            log::warn!("suspend state refused: version {version} is not {SUSPEND_STATE_VERSION}");
            return Err(resume_error(RESUME_ERROR_INVALID_SUSPEND_STATE_VERSION));
        }
        let mut registers = None;
        let mut info = None;
        let mut at = HEADER_SIZE;
        while at < bytes.len() {
            let (Some(tag), Some(length)) = (word(bytes, at), word(bytes, at.saturating_add(4)))
            else {
                return invalid(format_args!("the section header at {at} runs past the end"));
            };
            let start = at.saturating_add(SECTION_HEADER_SIZE);
            let end = start.checked_add(length.to_usize());
            let Some((payload, end)) = end.and_then(|end| Some((bytes.get(start..end)?, end)))
            else {
                return invalid(format_args!(
                    "section {tag} at {at} runs past the end ({length} bytes)"
                ));
            };
            let section = match Section::try_from(tag) {
                Ok(section) => section,
                Err(tag) => return invalid(format_args!("section {tag} at {at} is unknown")),
            };
            let repeated = match section {
                Section::Registers => registers.is_some(),
                Section::DeviceInfo => info.is_some(),
            };
            if repeated {
                return invalid(format_args!("section {tag} at {at} is repeated"));
            }
            match section {
                Section::Registers => {
                    let Some(r) = parse_registers(payload) else {
                        return invalid(format_args!(
                            "section {tag} is {length} bytes, not {}",
                            REGISTER_WORDS.saturating_mul(4)
                        ));
                    };
                    registers = Some(r);
                }
                Section::DeviceInfo => {
                    let Some((max_key, pairs)) = parse_device_info(payload) else {
                        return invalid(format_args!(
                            "section {tag} is {length} bytes: not a max key and at most {} pairs",
                            MAX_INFO_PAIRS
                        ));
                    };
                    if let Some(p) = pairs.iter().find(|p| !readable_info_key(p.key)) {
                        return invalid(format_args!(
                            "device info key {} is not one a resume reads",
                            p.key
                        ));
                    }
                    info = Some((max_key, pairs));
                }
            }
            at = end;
        }
        let Some(registers) = registers else {
            return invalid(format_args!(
                "section {} is missing",
                Section::Registers.tag()
            ));
        };
        let Some((device_info_max_key, device_info)) = info else {
            return invalid(format_args!(
                "section {} is missing",
                Section::DeviceInfo.tag()
            ));
        };
        Ok(SuspendState {
            registers,
            device_info_max_key,
            device_info,
        })
    }
}

/// The registers section's payload: exactly its 20 words.
fn parse_registers(payload: &[u8]) -> Option<SuspendDeviceState> {
    let (words, rest) = payload.as_chunks::<4>();
    if !rest.is_empty() {
        return None;
    }
    let words: [[u8; 4]; REGISTER_WORDS] = words.try_into().ok()?;
    Some(SuspendDeviceState::from_words(
        words.map(u32::from_le_bytes),
    ))
}

/// The device info section's payload: the max key, then at most [`MAX_INFO_PAIRS`] whole pairs.
fn parse_device_info(payload: &[u8]) -> Option<(i32, Vec<KeyValuePair>)> {
    let max_key = word(payload, 0)?.cast_signed();
    let (pairs, rest) = payload.get(4..)?.as_chunks::<8>();
    if !rest.is_empty() || pairs.len() > MAX_INFO_PAIRS {
        return None;
    }
    let pairs = pairs
        .iter()
        .map(|p| {
            let (key, value) = p.split_at(4);
            KeyValuePair {
                key: word(key, 0).unwrap_or(0),
                value: word(value, 0).unwrap_or(0),
            }
        })
        .collect();
    Some((max_key, pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    type Lines = Vec<(log::Level, String)>;

    /// What parsing `blob` gives, and the lines it logged, each as its level and message.
    fn parsed(blob: &[u8]) -> (Result<SuspendState, NsErrorText>, Lines) {
        let logs = crate::testkit::logs::capture();
        let r = SuspendState::parse(blob);
        let lines = logs
            .take()
            .into_iter()
            .map(|(level, _, message)| (level, message))
            .collect();
        (r, lines)
    }

    /// The refusal's code, after checking it logged exactly `line`, as a warning.
    fn refused(blob: &[u8], line: &str) -> i64 {
        let (r, lines) = parsed(blob);
        assert_eq!(lines, vec![(log::Level::Warn, line.to_owned())]);
        let e = r.unwrap_err();
        assert_eq!(e.domain, "PGResumeErrorDomain");
        e.code
    }

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn registers() -> SuspendDeviceState {
        SuspendDeviceState::from_words(std::array::from_fn(|i| {
            0x100_u32.wrapping_add(u32::try_from(i).unwrap())
        }))
    }

    fn state(max_key: i32, info: &[(u32, u32)]) -> SuspendState {
        SuspendState {
            registers: registers(),
            device_info_max_key: max_key,
            device_info: info
                .iter()
                .map(|&(key, value)| KeyValuePair { key, value })
                .collect(),
        }
    }

    /// A header, then the sections given as (tag, payload words).
    fn blob(sections: &[(u32, &[u32])]) -> Vec<u8> {
        let mut w = vec![SUSPEND_STATE_MAGIC, SUSPEND_STATE_VERSION];
        for &(tag, payload) in sections {
            w.extend([tag, u32::try_from(payload.len() * 4).unwrap()]);
            w.extend(payload);
        }
        words(&w)
    }

    fn register_words() -> Vec<u32> {
        registers().to_words().to_vec()
    }

    mod serialize {
        use super::*;

        /// The header, the registers section, then the device info section, allocated once at
        /// the blob's length
        #[test]
        fn layout() {
            let b = state(13, &[(12, 1), (1, 4)]).serialize();
            let mut want = vec![SUSPEND_STATE_MAGIC, 1, 1, 80];
            want.extend(register_words());
            want.extend([2, 20, 13, 12, 1, 1, 4]);
            assert_eq!(b, words(&want));
            assert_eq!(&b[..4], b"RVGS");
            assert_eq!(b.capacity(), b.len());
        }

        #[test]
        fn round_trip() {
            for (max_key, info) in [
                (1, vec![]),
                (0, vec![]),
                (-1, vec![(1, 2)]),
                (i32::MAX, vec![(44, 7), (42, 0)]),
            ] {
                let s = state(max_key, &info);
                let (r, lines) = parsed(&s.serialize());
                assert_eq!(r.unwrap(), s);
                assert!(lines.is_empty());
            }
        }

        /// `new` keeps the writer's pairs up to its terminator, at most 45
        #[test]
        fn new_stops_at_the_terminator() {
            let mut written = [KeyValuePair::default(); SERIALIZED_INFO_PAIRS];
            written[0] = KeyValuePair { key: 3, value: 9 };
            written[2] = KeyValuePair { key: 4, value: 9 };
            let s = SuspendState::new(registers(), 5, &written);
            assert_eq!(s.device_info, [KeyValuePair { key: 3, value: 9 }]);
            let full = [KeyValuePair { key: 1, value: 1 }; SERIALIZED_INFO_PAIRS];
            let s = SuspendState::new(registers(), 5, &full);
            assert_eq!(s.device_info.len(), MAX_INFO_PAIRS);
            assert!(parsed(&s.serialize()).0.is_ok());
        }

        /// A max key of any value serializes: negative and huge ones included
        #[test]
        fn any_max_key() {
            for max_key in [i32::MIN, -25, -1, 0, 47, 0x10_0001, i32::MAX] {
                let b = state(max_key, &[]).serialize();
                assert_eq!(b.len(), 8 + 88 + 12);
                let r = parsed(&b).0.unwrap();
                assert_eq!(r.device_info_max_key, max_key);
            }
        }
    }

    mod parse {
        use super::*;

        /// Sections in either order
        #[test]
        fn either_order() {
            let b = blob(&[(2, &[45, 1, 4]), (1, &register_words())]);
            let s = parsed(&b).0.unwrap();
            assert_eq!(s, state(45, &[(1, 4)]));
        }

        #[test]
        fn short_header() {
            let header = words(&[SUSPEND_STATE_MAGIC, 1]);
            for n in 0..8 {
                let line = format!("suspend state refused: {n} bytes are shorter than its header");
                assert_eq!(refused(&header[..n], &line), 2);
            }
        }

        /// Any other magic (here `PGSD`) is not ours
        #[test]
        fn foreign_magic() {
            let b = words(&[0x5047_5344, 2, 22, 45, 0, 0, 0, 0]);
            assert_eq!(
                refused(&b, "suspend state refused: magic 0x50475344 is not ours"),
                2
            );
        }

        #[test]
        fn other_versions() {
            for v in [0, 2, u32::MAX] {
                let mut b = state(1, &[]).serialize();
                b[4..8].copy_from_slice(&v.to_le_bytes());
                let line = format!("suspend state refused: version {v} is not 1");
                assert_eq!(refused(&b, &line), 1);
            }
        }

        #[test]
        fn missing_sections() {
            assert_eq!(
                refused(&blob(&[]), "suspend state refused: section 1 is missing"),
                2
            );
            assert_eq!(
                refused(
                    &blob(&[(1, &register_words())]),
                    "suspend state refused: section 2 is missing"
                ),
                2
            );
            assert_eq!(
                refused(
                    &blob(&[(2, &[1])]),
                    "suspend state refused: section 1 is missing"
                ),
                2
            );
        }

        #[test]
        fn repeated_section() {
            let b = blob(&[(2, &[1]), (1, &register_words()), (2, &[1])]);
            assert_eq!(
                refused(&b, "suspend state refused: section 2 at 108 is repeated"),
                2
            );
        }

        #[test]
        fn unknown_section() {
            let b = blob(&[(1, &register_words()), (3, &[]), (2, &[1])]);
            assert_eq!(
                refused(&b, "suspend state refused: section 3 at 96 is unknown"),
                2
            );
        }

        /// Every truncation of a good blob is refused
        #[test]
        fn truncation() {
            let b = state(13, &[(12, 1)]).serialize();
            for n in 0..b.len() {
                let (r, lines) = parsed(&b[..n]);
                assert!(r.is_err(), "{n}");
                assert_eq!(lines.len(), 1, "{n}");
            }
        }

        #[test]
        fn section_past_the_end() {
            let mut b = blob(&[(1, &register_words()), (2, &[1])]);
            // The device info section claims 8 bytes it does not have.
            b[100..104].copy_from_slice(&12u32.to_le_bytes());
            assert_eq!(
                refused(
                    &b,
                    "suspend state refused: section 2 at 96 runs past the end (12 bytes)"
                ),
                2
            );
            // A length that wraps the offset.
            b[100..104].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(parsed(&b).0.is_err());
            // A section header cut short.
            let b = [blob(&[(1, &register_words()), (2, &[1])]), vec![2, 0, 0]].concat();
            assert_eq!(
                refused(
                    &b,
                    "suspend state refused: the section header at 108 runs past the end"
                ),
                2
            );
        }

        #[test]
        fn wrong_lengths() {
            let short = &register_words()[..19];
            assert_eq!(
                refused(
                    &blob(&[(1, short), (2, &[1])]),
                    "suspend state refused: section 1 is 76 bytes, not 80"
                ),
                2
            );
            // Half a pair.
            let b = blob(&[(1, &register_words()), (2, &[1, 4])]);
            assert_eq!(
                refused(
                    &b,
                    "suspend state refused: section 2 is 8 bytes: not a max key and at most 45 \
                     pairs"
                ),
                2
            );
            // No max key.
            assert!(
                parsed(&blob(&[(1, &register_words()), (2, &[])]))
                    .0
                    .is_err()
            );
            // 46 pairs.
            let mut info = vec![45];
            info.extend([1, 0].repeat(46));
            assert!(
                parsed(&blob(&[(1, &register_words()), (2, &info)]))
                    .0
                    .is_err()
            );
            info.truncate(1 + 45 * 2);
            assert!(
                parsed(&blob(&[(1, &register_words()), (2, &info)]))
                    .0
                    .is_ok()
            );
        }

        /// Keys 0, 43 and 45 up are refused before any check reads them
        #[test]
        fn device_info_keys() {
            for key in [0, 43, 45, u32::MAX] {
                let b = state(i32::MAX, &[(1, 0), (key, 0)]).serialize();
                let line = format!(
                    "suspend state refused: device info key {key} is not one a resume reads"
                );
                assert_eq!(refused(&b, &line), 2);
            }
            for key in (1..=42).chain([44]) {
                assert!(parsed(&state(i32::MAX, &[(key, 0)]).serialize()).0.is_ok());
            }
        }
    }
}

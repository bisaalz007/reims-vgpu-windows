//! The binary versions a guest may ask for, and the features each gives.
//!
//! [`features_for_version`] is the features of any version: the 51 valid ones are the
//! labels of a cumulative switch. [`set_binary_version`] is a guest's request, negotiated
//! against the host's highest version and the device's feature level.
//!
//! What the guest sees (testing.md 7.1): a negative request is refused; 0 is accepted and logged
//! unknown; 1 and 236 are stored with bit 31; above 0xf5 clamps; other unknown values step down
//! to a valid one; the feature level clamps after the unknown-version log, and a level of 5 or
//! more dies there.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::guest::GuestDeath;
use crate::guest::records::{BinaryVersion, GuestFeatures};

/// The first feature level no device has: a version write on a device at or above it dies.
pub const DEVICE_FEATURE_LEVEL_MAX: u64 = 5;

/// The highest binary version the host knows (0xf5): an unknown request above it clamps to it.
/// Name is ours: the constant the negotiation compares against.
pub const HIGHEST_BINARY_VERSION: i32 = 0xf5;

/// One step of the cumulative switch: the case labels that enter at it and what it sets.
type Step = (&'static [i32], fn(&mut GuestFeatures));

/// The features' switch, highest label first: a version enters at the step that lists it and
/// falls through every later step.
const STEPS: &[Step] = &[
    (&[245], |f| f.supports_multiple_displays = true),
    (&[238], |f| f.supports_display_port_connection_type = true),
    (&[237], |f| f.supports_texture_descriptor2 = true),
    (&[235], |f| f.supports_info_indirect = true),
    (&[233], |f| f.supports_dispatch_threads_indirect = true),
    (&[232], |f| f.supports_protected_texture_options = true),
    (&[231], |f| f.supports_indirect_command_buffers = true),
    (&[230], |f| f.supports_cpu_buffers = true),
    (&[209], |f| f.supports_correct_2p_xr10_a8 = true),
    (&[208], |f| f.supports_display_set_guest_icc_profile = true),
    (&[207], |f| f.supports_argument_buffers = true),
    (&[206], |f| {
        f.supports_insert_compressed_texture_reinterpretation_flush = true
    }),
    (&[205], |f| f.supports_protection_options_envelope = true),
    (&[204], |f| f.supports_heap_hazard_tracking = true),
    (&[203], |f| f.supports_shared_storage_heaps = true),
    (&[202], |f| f.supports_command_buffer_jump = true),
    (&[201], |f| f.supports_object_unique_identifier = true),
    (&[200], |f| {
        f.supports_compute_pass_descriptor_dispatch_type = true
    }),
    (&[113], |f| {
        f.supports_multiplane_mipped_blit_textures = true
    }),
    (&[112], |f| f.supports_blit_encoder_spi = true),
    (&[111], |f| f.supports_dynamic_attribute_stride = true),
    (&[110], |f| f.supports_swizzled_textures = true),
    (&[109], |f| f.supports_cmd_exec_indirect3 = true),
    (&[108], |f| f.supports_display_compositor_parameters = true),
    (&[107], |f| f.supports_tile_shaders_image_blocks = true),
    (&[106], |f| {
        f.supports_variable_rasterization_rate_maps = true
    }),
    (&[105], |f| f.supports_programmable_sample_positions = true),
    (&[104], |f| f.supports_sunburst_display = true),
    (&[103], |f| f.supports_vertex_amplification = true),
    (&[102], |f| f.supports_default_raster_sample_count = true),
    (&[101], |f| f.supports_info_encoder = true),
    (&[100], |f| f.supports_shared_textures = true),
    (&[62], |f| f.s8_byte_count_for_d32s8 = 1),
    (&[61], |f| f.max_fifo_count = 64),
    (&[60], |f| f.buffer_from_io_surface = true),
    (&[43], |f| f.metal_heaps = true),
    (&[42], |f| f.display_dimension_floats = true),
    (&[41, 40], |f| f.display_set_properties = true),
    (&[31], |f| f.dual_plane_textures = true),
    (&[30], |f| f.synchronize_discard_child_resources = true),
    (&[22], |f| f.supports_rgha_presents = true),
    (&[21], |f| f.display_sleep = true),
    (&[20, 11], |f| f.discard_resources = true),
    (&[10], |f| {
        f.mapper_io_surfaces = true;
        f.synchronize_and_discard_resources = true;
        f.display_mapper_surface = true;
    }),
    (&[6], |f| f.supports_sunburst_display = true),
    (&[5], |f| f.supports_shared_textures = true),
    (&[4], |f| f.display_doorbell = true),
    (&[3], |f| f.efi_display = true),
    (&[2], |f| f.object_tables = true), // Then `break`
];

/// The whole 60-byte features of any `version`, valid or not. A version the host does not know
/// still gets a struct: `valid` false, `maxFIFOCount` 32, `s8ByteCountForD32S8` 4, and
/// `unsupported` for 1 and 236; a version write stores exactly that for a version it marked with
/// bit 31. The 51 valid versions are the labels of [`STEPS`].
pub fn features_for_version(version: BinaryVersion) -> GuestFeatures {
    // Zeroed, then valid, 32 FIFOs, 4 bytes of stencil.
    let mut f = GuestFeatures {
        valid: true,
        max_fifo_count: 32,
        s8_byte_count_for_d32s8: 4,
        ..GuestFeatures::default()
    };
    match STEPS
        .iter()
        .position(|(labels, _)| labels.contains(&version))
    {
        // Enter at the version's case and fall through to case 2.
        Some(entry) => {
            for (_, set) in STEPS.iter().skip(entry) {
                set(&mut f);
            }
        }
        None => {
            // Cases 1 and 236 mark it unsupported, then default.
            if version == 1 || version == 236 {
                f.unsupported = true;
            }
            // Default: not valid.
            f.valid = false;
        }
    }
    // Signed comparisons on the version.
    if version == 40 {
        f.dual_plane_textures = false;
    }
    if (7..100).contains(&version) {
        f.supports_shared_textures = false;
    }
    if (7..104).contains(&version) {
        f.supports_sunburst_display = false;
    }
    f
}

/// A new device's line for its feature level: 0 logs nothing, 1-4 name their year, and 5 or more
/// is an error (the device is still created; its first version write dies).
pub fn log_device_feature_level(feature_level: u64) {
    let year = match feature_level {
        0 => return,
        1 => 2022,
        2 => 2023,
        3 => 2024,
        4 => 2025,
        _ => {
            log::error!(
                "device feature level {feature_level} is above 4: the guest's first binary \
                 version write will end the process"
            );
            return;
        }
    };
    log::info!("device feature level {feature_level}: guest features as of {year}");
}

/// The version a device feature level allows a request: 1 → at most 62, 2 → 113, 3 → 209,
/// 0 and 4 → 245. A level of [`DEVICE_FEATURE_LEVEL_MAX`] or more dies.
pub fn binary_version_from_feature_level(feature_level: u64, version: i32) -> i32 {
    // A level ≥ 5 is accepted at creation and dies here, at the first version write (and at every
    // `reset`).
    crate::fatal_unless!(
        feature_level < DEVICE_FEATURE_LEVEL_MAX,
        GuestDeath::FeatureLevelTooHigh
    );
    match feature_level {
        1 => version.min(62),
        2 => version.min(113),
        3 => version.min(209),
        _ => version.min(245),
    }
}

/// The version a version write stores for a request, and whether it logged the request as
/// unknown. Name is ours: the two values the negotiation computes before its stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegotiatedVersion {
    /// What `_binaryVersion` becomes (bit 31 set: unsupported). From [`step_binary_version`]:
    /// not yet clamped to the device's feature level.
    pub version: i32,
    /// The request is logged as unknown.
    pub unknown: bool,
}

/// A version write's arithmetic: [`step_binary_version`], then [`clamp_binary_version`]. `None`
/// is the refusal of a request with bit 31 set (logged, nothing stored). The caller then stores
/// `version`, logs, sets the device's features from [`features_for_version`]`(version)`
/// and turns EFI off when that has no `efiDisplay`.
/// The unknown request is logged between the two halves, so [`set_binary_version`] calls them
/// apart.
///
/// Negative ignored; 0 accepted (logged unknown); 1 and 236 stored with bit 31; above 0xf5
/// clamped; other unknown values step down to a valid one; the feature-level clamp; the dead
/// `version == 1` check.
#[cfg(test)]
pub fn negotiate_binary_version(requested: u32, feature_level: u64) -> Option<NegotiatedVersion> {
    let n = step_binary_version(requested)?;
    Some(NegotiatedVersion {
        version: clamp_binary_version(n.version, feature_level),
        unknown: n.unknown,
    })
}

/// The first half of a version write's arithmetic, before its unknown-version log: the refusal
/// of a request with bit 31 set (`None`), bit 31 for an unsupported request, and the step down of
/// an unknown one. `version` is not yet clamped to the device's feature level
/// ([`clamp_binary_version`], after the log). Name is ours.
pub fn step_binary_version(requested: u32) -> Option<NegotiatedVersion> {
    let mut version = requested.cast_signed();
    if version < 0 {
        return None;
    }
    let mut unknown = false;
    let features = features_for_version(version);
    if features.unsupported {
        version |= i32::MIN;
        unknown = true;
    } else if !features.valid {
        if version > HIGHEST_BINARY_VERSION {
            version = HIGHEST_BINARY_VERSION;
        } else if version != 0 {
            // Step down to the next valid version (2 at worst).
            loop {
                version = version.wrapping_sub(1);
                if version == 0 || features_for_version(version).valid {
                    break;
                }
            }
        }
        unknown = true;
    }
    Some(NegotiatedVersion { version, unknown })
}

/// The second half of a version write's arithmetic, after its unknown-version log: the
/// feature-level clamp of a non-negative version (which dies for a level of 5 or more,
/// [`binary_version_from_feature_level`]) and the dead `version == 1` check. Name is ours.
pub fn clamp_binary_version(version: i32, feature_level: u64) -> i32 {
    let mut version = version;
    if version >= 0 {
        version = binary_version_from_feature_level(feature_level, version);
    }
    // Dead: 1 already has bit 31, and no clamp or step lands on it.
    if version == 1 {
        version = 0x8000_0001_u32.cast_signed();
    }
    version
}

/// A guest's binary version write, through the unsupported-version log.
///
/// Negotiates ([`step_binary_version`], then [`clamp_binary_version`]), logs its lines, and calls
/// `store_binary_version` to store the version (between the unknown-version log and the setting
/// log). Returns the stored version, or `None` for a refused request (nothing stored). The caller
/// then sets the device's features from [`features_for_version`] of the stored version and,
/// when that has no `efiDisplay`, turns EFI off. Dies where [`binary_version_from_feature_level`]
/// does (feature level 5 or more), after the unknown-version log.
pub fn set_binary_version(
    requested: u32,
    feature_level: u64,
    store_binary_version: impl FnOnce(i32),
) -> Option<i32> {
    let Some(stepped) = step_binary_version(requested) else {
        log::warn!("binary version request {requested:#x} refused: bit 31 is set");
        return None;
    };
    if stepped.unknown {
        log::debug!("guest asked for unknown binary version {requested}");
    }
    // After the line: a level of 5 or more dies here with it written.
    let version = clamp_binary_version(stepped.version, feature_level);
    store_binary_version(version);
    log::info!("guest asked for binary version {requested}; version {version:#x} is stored");
    if version < 0 {
        log::warn!("binary version {requested} is not supported: it is stored with bit 31 set");
    }
    Some(version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::capture::catch_death;

    mod set_version {
        use super::*;
        use crate::sync::Mutex;
        use crate::testkit::logs::{Capture, capture};

        /// The lines `logs` took since its last take, each as its level and message.
        fn lines(logs: &Capture) -> Vec<String> {
            logs.take()
                .into_iter()
                .map(|(level, _, message)| format!("{level} {message}"))
                .collect()
        }

        /// (returned, the store and the lines in the order they happened).
        fn run(requested: u32, level: u64) -> (Option<i32>, Vec<String>) {
            let logs = capture();
            let seen = Mutex::new(Vec::new());
            let out = set_binary_version(requested, level, |v| {
                let mut seen = seen.lock();
                seen.extend(lines(&logs));
                seen.push(format!("store {v:#x}"));
            });
            let mut v = seen.lock().clone();
            v.extend(lines(&logs));
            (out, v)
        }

        #[test]
        fn negative_is_refused_with_a_warning() {
            let (out, v) = run(0x8000_0000, 4);
            assert_eq!(out, None);
            assert_eq!(
                v,
                ["WARN binary version request 0x80000000 refused: bit 31 is set"]
            );
        }

        #[test]
        fn known_version_logs_the_setting_only() {
            let (out, v) = run(245, 4);
            assert_eq!(out, Some(245));
            assert_eq!(
                v,
                [
                    "store 0xf5",
                    "INFO guest asked for binary version 245; version 0xf5 is stored",
                ]
            );
        }

        /// 0 is unknown (logged) and stays 0; the feature level clamps.
        #[test]
        fn unknown_and_clamped() {
            let (out, v) = run(0, 4);
            assert_eq!(out, Some(0));
            assert_eq!(
                v,
                [
                    "DEBUG guest asked for unknown binary version 0",
                    "store 0x0",
                    "INFO guest asked for binary version 0; version 0x0 is stored",
                ]
            );
            let (out, _) = run(245, 1);
            assert_eq!(out, Some(62));
        }

        /// 1 and 236 are unsupported: stored with bit 31, and the error follows.
        #[test]
        fn unsupported_logs_three_lines() {
            let (out, v) = run(236, 4);
            assert_eq!(out, Some(0x8000_00ec_u32.cast_signed()));
            assert_eq!(
                v,
                [
                    "DEBUG guest asked for unknown binary version 236",
                    "store 0x800000ec",
                    "INFO guest asked for binary version 236; version 0x800000ec is stored",
                    "WARN binary version 236 is not supported: it is stored with bit 31 set",
                ]
            );
        }

        /// The unknown-version line comes before the feature-level clamp, so a device feature
        /// level of 5 or more writes that line and then dies at the clamp, with nothing stored and
        /// no setting line. A known request dies with nothing logged.
        /// The level ≥ 5 death comes after the unknown-version log.
        #[test]
        fn level_5_logs_an_unknown_request_before_it_dies() {
            let logs = capture();
            for (requested, unknown) in [(0, true), (7, true), (300, true), (245, false)] {
                let stored: Mutex<Option<i32>> = Mutex::new(None);
                let site =
                    catch_death(|| set_binary_version(requested, 5, |v| *stored.lock() = Some(v)));
                assert_eq!(
                    site,
                    Err(GuestDeath::FeatureLevelTooHigh.site()),
                    "{requested}"
                );
                let want: Vec<String> = if unknown {
                    vec![format!(
                        "DEBUG guest asked for unknown binary version {requested}"
                    )]
                } else {
                    Vec::new()
                };
                assert_eq!(lines(&logs), want, "{requested}");
                assert_eq!(*stored.lock(), None, "{requested}");
            }
        }

        /// An unsupported request has bit 31 set after the step, so the clamp (non-negative
        /// versions only) skips it: a level of 5 or more does not die, and all three lines log.
        #[test]
        fn level_5_passes_an_unsupported_request() {
            let (out, v) = run(236, 5);
            assert_eq!(out, Some(0x8000_00ec_u32.cast_signed()));
            assert_eq!(
                v,
                [
                    "DEBUG guest asked for unknown binary version 236",
                    "store 0x800000ec",
                    "INFO guest asked for binary version 236; version 0x800000ec is stored",
                    "WARN binary version 236 is not supported: it is stored with bit 31 set",
                ]
            );
        }
    }

    mod feature_level_log {
        use super::*;
        use log::Level;

        fn lines(level: u64) -> Vec<(Level, String)> {
            let logs = crate::testkit::logs::capture();
            log_device_feature_level(level);
            logs.take()
                .into_iter()
                .map(|(level, _, message)| (level, message))
                .collect()
        }

        #[test]
        fn zero_logs_nothing() {
            assert!(lines(0).is_empty());
        }

        #[test]
        fn one_to_four_name_the_year() {
            for (level, year) in [(1, 2022), (2, 2023), (3, 2024), (4, 2025)] {
                assert_eq!(
                    lines(level),
                    vec![(
                        Level::Info,
                        format!("device feature level {level}: guest features as of {year}")
                    )]
                );
            }
        }

        /// Printed unsigned.
        #[test]
        fn five_and_up_are_errors() {
            for level in [5, 7, u64::MAX] {
                assert_eq!(
                    lines(level),
                    vec![(
                        Level::Error,
                        format!(
                            "device feature level {level} is above 4: the guest's first binary \
                             version write will end the process"
                        )
                    )]
                );
            }
        }
    }

    /// The 51 valid versions.
    const VALID: [i32; 51] = [
        2, 3, 4, 5, 6, 10, 11, 20, 21, 22, 30, 31, 40, 41, 42, 43, 60, 61, 62, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 230, 231, 232, 233, 235, 237, 238, 245,
    ];

    /// The flags a feature set has, by name, for readable comparisons.
    fn flags(f: &GuestFeatures) -> Vec<&'static str> {
        let all = [
            (f.object_tables, "objectTables"),
            (f.efi_display, "efiDisplay"),
            (f.display_doorbell, "displayDoorbell"),
            (f.mapper_io_surfaces, "mapperIOSurfaces"),
            (
                f.synchronize_and_discard_resources,
                "synchronizeAndDiscardResources",
            ),
            (f.display_mapper_surface, "displayMapperSurface"),
            (f.discard_resources, "discardResources"),
            (f.display_sleep, "displaySleep"),
            (f.supports_rgha_presents, "supportsRGhAPresents"),
            (
                f.synchronize_discard_child_resources,
                "synchronizeDiscardChildResources",
            ),
            (f.dual_plane_textures, "dualPlaneTextures"),
            (f.display_set_properties, "displaySetProperties"),
            (f.display_dimension_floats, "displayDimensionFloats"),
            (f.metal_heaps, "metalHeaps"),
            (f.buffer_from_io_surface, "bufferFromIOSurface"),
            (f.supports_shared_textures, "supportsSharedTextures"),
            (f.supports_sunburst_display, "supportsSunburstDisplay"),
            (f.supports_multiple_displays, "supportsMultipleDisplays"),
        ];
        all.into_iter()
            .filter(|(b, _)| *b)
            .map(|(_, n)| n)
            .collect()
    }

    mod features_for_version {
        use super::*;

        /// Exactly the 51 labels are valid; nothing else in a wide range is.
        #[test]
        fn valid_versions_are_the_51_labels() {
            let valid: Vec<i32> = (-1024..=4096)
                .filter(|&v| features_for_version(v).valid)
                .collect();
            assert_eq!(valid, VALID.to_vec());
            for v in [i32::MIN, -1, i32::MAX, 0x8000_00f5_u32 as i32] {
                assert!(!features_for_version(v).valid, "{v}");
            }
        }

        /// An unknown version still gets a struct: 32 FIFOs, 4 stencil bytes, and
        /// `unsupported` only for 1 and 236.
        #[test]
        fn unknown_versions_keep_the_defaults() {
            for v in [i32::MIN, -1, 0, 1, 7, 99, 236, 246, i32::MAX] {
                let f = features_for_version(v);
                assert!(!f.valid, "{v}");
                assert_eq!(f.unsupported, v == 1 || v == 236, "{v}");
                assert_eq!(f.max_fifo_count, 32, "{v}");
                assert_eq!(f.s8_byte_count_for_d32s8, 4, "{v}");
                assert!(flags(&f).is_empty(), "{v}: {:?}", flags(&f));
            }
        }

        /// Case 2 sets objectTables only; each lower case adds its flag.
        #[test]
        fn fallthrough_is_cumulative() {
            assert_eq!(flags(&features_for_version(2)), ["objectTables"]);
            assert_eq!(
                flags(&features_for_version(4)),
                ["objectTables", "efiDisplay", "displayDoorbell"]
            );
            assert_eq!(
                flags(&features_for_version(6)),
                [
                    "objectTables",
                    "efiDisplay",
                    "displayDoorbell",
                    "supportsSharedTextures",
                    "supportsSunburstDisplay"
                ]
            );
            let f = features_for_version(245);
            assert!(f.valid && !f.unsupported);
            assert_eq!(f.max_fifo_count, 64);
            assert_eq!(f.s8_byte_count_for_d32s8, 1);
            assert!(f.supports_multiple_displays && f.supports_display_port_connection_type);
            assert!(f.object_tables && f.efi_display && f.supports_sunburst_display);
            let g = features_for_version(238);
            assert!(!g.supports_multiple_displays && g.supports_display_port_connection_type);
        }

        /// maxFIFOCount 64 from 61 up; s8ByteCountForD32S8 1 from 62 up.
        #[test]
        fn fifo_count_and_stencil_bytes() {
            assert_eq!(features_for_version(60).max_fifo_count, 32);
            assert_eq!(features_for_version(61).max_fifo_count, 64);
            assert_eq!(features_for_version(61).s8_byte_count_for_d32s8, 4);
            assert_eq!(features_for_version(62).s8_byte_count_for_d32s8, 1);
        }

        /// 41 and 40 share a case; 40 then clears dualPlaneTextures.
        #[test]
        fn version_40_clears_dual_plane() {
            let f40 = features_for_version(40);
            let f41 = features_for_version(41);
            assert!(f40.display_set_properties && f41.display_set_properties);
            assert!(!f40.dual_plane_textures);
            assert!(f41.dual_plane_textures);
        }

        /// 20 and 11 share a case.
        #[test]
        fn versions_20_and_11_share_a_case() {
            assert_eq!(features_for_version(20), features_for_version(11));
            assert!(features_for_version(11).discard_resources);
            assert!(!features_for_version(10).discard_resources);
        }

        /// 7 ≤ v < 100 clears supportsSharedTextures; 7 ≤ v < 104 clears the sunburst flag.
        #[test]
        fn post_switch_ranges() {
            for v in [10, 62] {
                let f = features_for_version(v);
                assert!(
                    !f.supports_shared_textures && !f.supports_sunburst_display,
                    "{v}"
                );
            }
            for v in [100, 103] {
                let f = features_for_version(v);
                assert!(
                    f.supports_shared_textures && !f.supports_sunburst_display,
                    "{v}"
                );
            }
            let f = features_for_version(104);
            assert!(f.supports_shared_textures && f.supports_sunburst_display);
            let f = features_for_version(6);
            assert!(f.supports_shared_textures && f.supports_sunburst_display);
        }
    }

    mod feature_level_clamp {
        use super::*;

        #[test]
        fn clamps_per_level() {
            assert_eq!(binary_version_from_feature_level(0, 245), 245);
            assert_eq!(binary_version_from_feature_level(1, 245), 62);
            assert_eq!(binary_version_from_feature_level(1, 10), 10);
            assert_eq!(binary_version_from_feature_level(2, 245), 113);
            assert_eq!(binary_version_from_feature_level(3, 245), 209);
            assert_eq!(binary_version_from_feature_level(4, 245), 245);
            assert_eq!(binary_version_from_feature_level(4, 0), 0);
        }

        /// A level of 5 or more dies.
        #[test]
        fn level_5_dies() {
            for level in [5, u64::MAX] {
                let site = catch_death(|| binary_version_from_feature_level(level, 2));
                assert_eq!(site, Err(GuestDeath::FeatureLevelTooHigh.site()));
            }
        }
    }

    mod negotiate {
        use super::*;

        fn nv(version: i32, unknown: bool) -> Option<NegotiatedVersion> {
            Some(NegotiatedVersion { version, unknown })
        }

        /// Every valid version is kept as it is at level 0, and not logged as unknown.
        #[test]
        fn valid_versions_pass() {
            for v in VALID {
                assert_eq!(negotiate_binary_version(v as u32, 0), nv(v, false));
            }
        }

        /// Bit 31 set is refused.
        #[test]
        fn negative_is_ignored() {
            assert_eq!(negotiate_binary_version(0x8000_0000, 0), None);
            assert_eq!(negotiate_binary_version(u32::MAX, 0), None);
            // A refused request never reaches the feature-level assert.
            assert_eq!(negotiate_binary_version(0x8000_0002, 7), None);
        }

        /// 0 is accepted and logged unknown.
        #[test]
        fn zero_is_accepted() {
            assert_eq!(negotiate_binary_version(0, 0), nv(0, true));
        }

        /// 1 and 236 get bit 31, skip the level clamp (so no assert).
        #[test]
        fn unsupported_get_bit_31() {
            assert_eq!(
                negotiate_binary_version(1, 0),
                nv(0x8000_0001_u32 as i32, true)
            );
            assert_eq!(
                negotiate_binary_version(236, 0),
                nv(0x8000_00ec_u32 as i32, true)
            );
            assert_eq!(
                negotiate_binary_version(236, 9),
                nv(0x8000_00ec_u32 as i32, true)
            );
            // What the device then stores: an invalid set with 32 FIFOs.
            let f = features_for_version(0x8000_00ec_u32 as i32);
            assert!(!f.valid && !f.unsupported);
            assert_eq!(f.max_fifo_count, 32);
        }

        /// Above 0xf5 clamps to 0xf5; unknown values step down.
        #[test]
        fn clamp_and_step_down() {
            assert_eq!(negotiate_binary_version(246, 0), nv(245, true));
            assert_eq!(negotiate_binary_version(0x7fff_ffff, 0), nv(245, true));
            assert_eq!(negotiate_binary_version(244, 0), nv(238, true));
            assert_eq!(negotiate_binary_version(234, 0), nv(233, true));
            assert_eq!(negotiate_binary_version(199, 0), nv(113, true));
            assert_eq!(negotiate_binary_version(99, 0), nv(62, true));
            assert_eq!(negotiate_binary_version(9, 0), nv(6, true));
            assert_eq!(negotiate_binary_version(7, 0), nv(6, true));
        }

        /// The level clamp applies after the step down.
        #[test]
        fn feature_level_clamp() {
            assert_eq!(negotiate_binary_version(245, 1), nv(62, false));
            assert_eq!(negotiate_binary_version(300, 2), nv(113, true));
            assert_eq!(negotiate_binary_version(210, 3), nv(209, true));
            assert_eq!(negotiate_binary_version(245, 4), nv(245, false));
        }

        /// The two halves the unknown-version log sits between: the step takes no level (so it
        /// never reaches the death) and the clamp applies the level to what it returns.
        #[test]
        fn step_then_clamp() {
            let step = step_binary_version;
            assert_eq!(step(0x8000_0000), None);
            assert_eq!(step(245), nv(245, false));
            assert_eq!(step(300), nv(245, true));
            assert_eq!(step(0), nv(0, true));
            assert_eq!(step(236), nv(0x8000_00ec_u32 as i32, true));
            assert_eq!(clamp_binary_version(245, 1), 62);
            assert_eq!(clamp_binary_version(0, 4), 0);
            // Bit 31 skips the clamp, so a level of 5 or more passes it.
            assert_eq!(
                clamp_binary_version(0x8000_00ec_u32 as i32, 5),
                0x8000_00ec_u32 as i32
            );
            let site = catch_death(|| clamp_binary_version(0, 5));
            assert_eq!(site, Err(GuestDeath::FeatureLevelTooHigh.site()));
        }

        /// A level ≥ 5 dies at the first non-negative version write.
        #[test]
        fn level_5_dies_on_write() {
            let site = catch_death(|| negotiate_binary_version(2, 5));
            assert_eq!(site, Err(GuestDeath::FeatureLevelTooHigh.site()));
        }
    }
}

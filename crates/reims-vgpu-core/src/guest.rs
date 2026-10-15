//! Guest memory and the guest ABI.
//!
//! - [`ptr`]: `GuestPtr`, the only raw access to guest RAM;
//! - [`page`]: `GuestPage`, one mapped guest page;
//! - [`records`]: `APVFeatures` and the records a guest writes, read from private copies;
//! - [`abi`]: the codecs: the suspend blob (our own format), the device info pairs, and
//!   the features of each of the 51 binary versions.
//!
//! The area's own deaths are [`GuestDeath`]; a NULL guest access dies the death its caller names.

pub mod abi;
pub mod page;
pub mod ptr;
pub mod records;

pub use page::GuestPage;
pub use ptr::{GuestPtr, ZERO_REGION_END};

use std::fmt;

use crate::raise::death::DeathId;
use crate::raise::death::DeathKind;
use crate::raise::death::DeathSite;

/// Where the guest ABI's codecs end the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuestDeath {
    /// A device info list holds a key no reader takes (43, or 45 and above).
    UnknownInfoKey,
    /// A binary version write on a device whose feature level is 5 or more.
    FeatureLevelTooHigh,
}

impl GuestDeath {
    /// The dotted id the death line prints.
    pub fn id(self) -> &'static str {
        match self {
            GuestDeath::UnknownInfoKey => "guest.device_info.unknown_key",
            GuestDeath::FeatureLevelTooHigh => "guest.features.feature_level_too_high",
        }
    }

    /// The site the death's line and `testkit::capture` report: an assert, the dotted id, line 0.
    pub(crate) fn site(self) -> DeathSite {
        DeathSite {
            kind: DeathKind::Assert,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for GuestDeath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<GuestDeath> for DeathId {
    fn from(d: GuestDeath) -> DeathId {
        DeathId::Guest(d)
    }
}

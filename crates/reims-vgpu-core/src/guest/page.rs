//! `GuestPage`: one mapped guest page, a [`GuestPtr`] whose offsets stay inside the page.

use crate::guest::ptr::GuestPtr;

/// One guest page mapped on the host (the root node and a device-info page, `APV_PAGE_SIZE` bytes;
/// the display's shared page, `display::page::PAGE_LENGTH`): a `GuestPtr` at the page's first
/// byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GuestPage {
    base: GuestPtr,
}

impl GuestPage {
    /// The page at `base`.
    pub const fn new(base: GuestPtr) -> GuestPage {
        GuestPage { base }
    }

    /// The page's first byte.
    pub const fn base(self) -> GuestPtr {
        self.base
    }

    /// The byte at `off` in the page (no caller indexes past the length it mapped). Unchecked
    /// (`page + off`).
    pub const fn at(self, off: u64) -> GuestPtr {
        self.base.offset(off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::guest_ram::RawRegion;

    const SITE: DeathSite = DeathSite {
        kind: DeathKind::NullDeref,
        file: "test.guest_page",
        line: 0,
    };

    /// `GuestPage::new`, `base`, `at` (unchecked offset)
    #[test]
    fn page_offsets() {
        let r = RawRegion::new(0x4000).expect("region");
        let p = GuestPtr::new(r.addr());
        let page = GuestPage::new(p);
        assert_eq!(page.base(), p);
        assert_eq!(page.at(0x10), p.offset(0x10));
        assert_eq!(page.at(0x4000), p.offset(0x4000));
        page.at(0x3ffc).write_u32(0, 5, SITE);
        assert_eq!(r.read_u32(0x3ffc), Some(5));
        assert_eq!(GuestPage::default().base(), GuestPtr::default());
    }
}

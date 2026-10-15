//! The handlers' own tables and arithmetic. Each opcode's checks, its device or task call and
//! how its stamp completes are the suite's.

use super::*;

type P = crate::testkit::FakePlatform;

/// The opcode table: 53 cases and the default
#[test]
fn table() {
    let mut cases = 0;
    for id in 0..=u16::MAX {
        if Opcode::from_id(id).is_some() {
            cases += 1;
        }
    }
    assert_eq!(cases, 53);
    for id in (13..=29).chain([70, 0x100, u16::MAX]) {
        assert!(Opcode::from_id(id).is_none());
    }
    let deprecated = [3, 31, 33, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50];
    for id in 0..=69u16 {
        let op = Opcode::from_id(id);
        assert_eq!(
            op == Some(Opcode::Deprecated),
            deprecated.contains(&id),
            "{id}"
        );
        assert_eq!(
            op.and_then(Opcode::handler::<P>).is_some(),
            op.is_some() && !deprecated.contains(&id),
            "{id}"
        );
    }
}

/// os_mul_and_add_overflow and os_add3_overflow on a u32 result
#[test]
fn overflow_helpers() {
    assert_eq!(mul_add_overflows(3, 4, 8), (20, false));
    assert_eq!(mul_add_overflows(0x3fff_fffd, 4, 8), (0xffff_fffc, false));
    assert!(mul_add_overflows(0x3fff_fffe, 4, 8).1);
    assert!(mul_add_overflows(0x4000_0000, 4, 0).1);
    assert_eq!(add3_overflows(12, 0, u32::MAX - 12), (u32::MAX, false));
    assert!(add3_overflows(12, 1, u32::MAX - 12).1);
}

//! Fences (object type 7, command 13): no command follows the header, and no placement is
//! touched.

use super::STATE_OBJECT_TYPE;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::raise::RaiseResult;
use crate::resources::manager::create::CreateEnv;
use crate::resources::manager::records::OperationHeader;

/// The command type that makes a fence.
const NEW_FENCE: u32 = 13;

/// A fence from `entry` and its record.
pub(crate) fn create<P: Platform, E: CreateEnv<P>>(
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
) -> RaiseResult<Result<P::Fence, String>> {
    let object_type = entry.object_type();
    if object_type != STATE_OBJECT_TYPE {
        return Ok(Err(format!("object of type {object_type} is not a fence")));
    }
    let header = refuse!(
        cursor
            .read::<OperationHeader>()
            .map_err(|e| format!("command header not readable: {e}"))
    );
    if header.command_type != NEW_FENCE {
        let ct = header.command_type;
        return Ok(Err(format!("unknown command type {ct}")));
    }
    match env.device().map_or(Ok(None), |d| d.new_fence())? {
        Some(fence) => Ok(Ok(fence)),
        None => Ok(Err("Metal made no fence".to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::manager::create::test_env::*;
    use crate::testkit::Step;

    #[test]
    fn a_fence_takes_no_placement_and_refuses_as_the_other_states() {
        let e = TestEnv::new();
        let (_m, g) = placement(3, 4);
        let run = |t: u8, b: Bytes| create(&e, &entry(t), b.cursor()).unwrap();
        assert!(run(7, Bytes::new().header(13)).is_ok());
        assert_eq!(e.calls("newFence").len(), 1);
        assert_eq!(read_placement(g), (3, 4), "a fence writes no placement");
        assert_eq!(
            run(5, Bytes::new().header(13)).unwrap_err(),
            "object of type 5 is not a fence"
        );
        assert!(
            run(7, Bytes::new().u32(13))
                .unwrap_err()
                .starts_with("command header not readable: ")
        );
        assert_eq!(
            run(7, Bytes::new().header(3)).unwrap_err(),
            "unknown command type 3"
        );
        e.p.script().push("newFence", Step::Nil);
        assert_eq!(
            run(7, Bytes::new().header(13)).unwrap_err(),
            "Metal made no fence"
        );
        assert_eq!(e.calls("newFence").len(), 2);
        e.p.script()
            .push("newFence", Step::raise("NSGenericException", "x"));
        assert!(create(&e, &entry(7), Bytes::new().header(13).cursor()).is_err());
    }
}

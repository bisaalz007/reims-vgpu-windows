//! The other commands: the debug, no-op and delay commands, the child FIFO commands, the device
//! info query, and what an unknown or deprecated opcode does.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::Arc;

use crate::fifo::commands::{Command, dump};
use crate::fifo::wire::{FifoCommand, Payload};
use crate::fifo::{Fifo, RaiseKind};
use crate::platform::Platform;
use crate::raise;

/// The records' sizes.
const CHILD_FIFO: u32 = 4;
const GET_DEVICE_INFO: u32 = 12;
const DELAY: u32 = 8;

/// An opcode the table has no case for: logged, dumped and faulted, with no pool or holder of its
/// own.
pub(super) fn invalid<P: Platform>(
    fifo: &Arc<Fifo<P>>,
    header: &FifoCommand,
    payload: Payload<'_>,
) {
    log::warn!(
        "channel {}: unknown command {}",
        fifo.stamp_index(),
        header.command_id
    );
    dump(fifo, payload, payload.len());
    fifo.fault_at_stamp_value(header.stamp_value);
}

/// A deprecated opcode: logged, dumped and faulted, with no pool or holder of its own (what the
/// fault autoreleases lands in the thread's last-resort pool).
pub(super) fn deprecated<P: Platform>(
    fifo: &Arc<Fifo<P>>,
    header: &FifoCommand,
    payload: Payload<'_>,
) {
    log::warn!(
        "channel {}: command {} is no longer supported",
        fifo.stamp_index(),
        header.command_id
    );
    dump(fifo, payload, payload.len());
    fifo.fault_at_stamp_value(header.stamp_value);
}

/// 0: a debug string; nothing is done with it.
pub(super) fn debug<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(1)?;
        c.signal();
        Ok(())
    });
}

/// 30.
pub(super) fn nop<P: Platform>(c: &mut Command<'_, P>) {
    c.signal();
}

/// 61: a short payload faults; otherwise the stamp is never completed.
pub(super) fn delay<P: Platform>(c: &mut Command<'_, P>) {
    // No signal on success; every later stamp waits behind this one.
    c.f_class(|c, _| c.check_size(DELAY));
}

/// 48: defines a child FIFO. Root-ness is checked before the size; the child's start result is
/// ignored.
pub(super) fn define_child_fifo<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        if !c.fifo.is_root_fifo() {
            log::warn!(
                "channel {}: only the root FIFO defines child FIFOs",
                c.fifo.stamp_index()
            );
            raise!(RaiseKind::NotRoot, "only the root FIFO defines child FIFOs");
        }
        c.check_size(CHILD_FIFO)?;
        if !c.device().create_fifo(c.payload.u32(0))? {
            raise!(RaiseKind::CreateChild, "child FIFO not created");
        }
        c.signal();
        Ok(())
    });
}

/// 49: deletes a child FIFO, draining it on this thread (a delay there hangs it).
pub(super) fn delete_child_fifo<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        if !c.fifo.is_root_fifo() {
            log::warn!(
                "channel {}: only the root FIFO deletes child FIFOs",
                c.fifo.stamp_index()
            );
            raise!(RaiseKind::NotRoot, "only the root FIFO deletes child FIFOs");
        }
        c.check_size(CHILD_FIFO)?;
        if !c.device().delete_fifo(c.payload.u32(0))? {
            raise!(RaiseKind::DeleteChild, "child FIFO not deleted");
        }
        c.signal();
        Ok(())
    });
}

/// 58: writes the device's info (synchronously), then signals.
pub(super) fn get_device_info<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(GET_DEVICE_INFO)?;
        let p = c.payload;
        c.device().get_device_info(p.u32(0), p.u32(4), p.u32(8))?;
        c.signal();
        Ok(())
    });
}

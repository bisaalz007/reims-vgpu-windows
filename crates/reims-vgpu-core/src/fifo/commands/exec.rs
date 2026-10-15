//! The exec-indirect commands (55, 67) and the info-indirect command (69): each checks its
//! counts, looks a user task up and hands it the command buffers and resources.
//!
//! The checks and the lookup are F class; the task call itself is outside any handler: a raise out
//! of it is not a fault but a death. Every raise the task call makes outside its own handlers is
//! uncaught and dies at its own site, so the call never returns one; were it to, the raise would
//! die there.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::fifo::RaiseKind;
use crate::fifo::commands::{Command, NOT_KERNEL, UNSUPPORTED, add3_overflows};
use crate::fifo::wire::HEADER_LEN;
use crate::guest::records::{
    CmdExecIndirectCmdBuffer, CmdExecIndirectResource, CmdExecIndirectResource3,
};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::raise;
use crate::raise::RaiseResult;
use crate::tasks::{ExecArgs, Task};

/// The record's size: the task, the resource count and the command buffer count.
const EXEC_INDIRECT: u32 = 12;
/// A resource's size in 55.
const RESOURCE2: u32 = 24;
/// A resource's size in 67 and 69.
const RESOURCE3: u32 = 8;
/// A command buffer's size.
const CMD_BUFFER: u32 = 16;

/// Which of the three.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// 55.
    Exec2,
    /// 67.
    Exec3,
    /// 69.
    Info,
}

/// The checks and the lookup: the task and the two counts.
fn prologue<P: Platform>(c: &mut Command<'_, P>, kind: Kind) -> RaiseResult<(Task<P>, u32, u32)> {
    let gated = match kind {
        Kind::Exec2 => true,
        Kind::Exec3 => c.features().supports_cmd_exec_indirect3,
        Kind::Info => c.features().supports_info_indirect,
    };
    if !gated {
        raise!(RaiseKind::Unsupported, UNSUPPORTED);
    }
    c.check_size(EXEC_INDIRECT)?;
    let p = c.payload;
    let task_id = p.u32(0);
    if task_id == 0 {
        raise!(RaiseKind::NotKernel, NOT_KERNEL);
    }
    let resource_count = p.u32(4);
    let cmd_buffer_count = p.u32(8);
    let record = if kind == Kind::Exec2 {
        RESOURCE2
    } else {
        RESOURCE3
    };
    // The command buffers' length stays 0 when the first multiply overflows (the tests
    // short-circuit)
    let mut cmd_buffers_length: u32 = 0;
    let (resources_length, mut overflow) = resource_count.overflowing_mul(record);
    let mut length = 0;
    if !overflow {
        let (l, o) = cmd_buffer_count.overflowing_mul(CMD_BUFFER);
        cmd_buffers_length = l;
        overflow = o;
        if !overflow {
            let (l, o) = add3_overflows(EXEC_INDIRECT, resources_length, cmd_buffers_length);
            length = l;
            overflow = o;
        }
    }
    if overflow {
        raise!(
            RaiseKind::Overflow,
            format!(
                "{resource_count} resources and {cmd_buffer_count} command buffers \
                 ({cmd_buffers_length:#x} bytes) overflow the payload length"
            )
        );
    }
    c.check_size(length)?;
    let task = c.user_task(task_id)?;
    Ok((task, resource_count, cmd_buffer_count))
}

/// The record lists: the resources after the record, the command buffers after them.
fn records<'a, P: Platform, R>(
    c: &Command<'a, P>,
    resource_count: u32,
    cmd_buffer_count: u32,
    record: u32,
) -> (
    crate::fifo::wire::Records<'a, R>,
    crate::fifo::wire::Records<'a, CmdExecIndirectCmdBuffer>,
)
where
    R: zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable + zerocopy::KnownLayout,
{
    let resources_at = HEADER_LEN.to_usize();
    let buffers_at = resources_at.wrapping_add(resource_count.wrapping_mul(record).to_usize());
    (
        c.payload.records(resources_at, resource_count),
        c.payload.records(buffers_at, cmd_buffer_count),
    )
}

/// 55.
pub(super) fn exec_indirect2<P: Platform>(c: &mut Command<'_, P>) {
    let hold = c.hold;
    let Some((task, rc, cbc)) = c.f_class(|c, _| prologue(c, Kind::Exec2)) else {
        return;
    };
    let (resources, cmd_buffers) = records::<P, CmdExecIndirectResource>(c, rc, cbc, RESOURCE2);
    let done = c.completion();
    // Outside any handler: a raise from the task is not a fault.
    if let Err(r) = task.exec_indirect2(
        hold,
        ExecArgs {
            cmd_buffers,
            resources,
            stamp_index: c.fifo.stamp_index(),
            fifo_event: c.fifo.event.as_ref(),
            fifo_event_value: &c.fifo.event_value,
        },
        done,
    ) {
        // Never reached: with no handler the raise died at its own site (module doc)
        r.fatal();
    }
}

/// 67.
pub(super) fn exec_indirect3<P: Platform>(c: &mut Command<'_, P>) {
    let hold = c.hold;
    let Some((task, rc, cbc)) = c.f_class(|c, _| prologue(c, Kind::Exec3)) else {
        return;
    };
    let (resources, cmd_buffers) = records::<P, CmdExecIndirectResource3>(c, rc, cbc, RESOURCE3);
    let done = c.completion();
    // Outside any handler.
    if let Err(r) = task.exec_indirect3(
        hold,
        ExecArgs {
            cmd_buffers,
            resources,
            stamp_index: c.fifo.stamp_index(),
            fifo_event: c.fifo.event.as_ref(),
            fifo_event_value: &c.fifo.event_value,
        },
        done,
    ) {
        // Never reached: with no handler the raise died at its own site (module doc)
        r.fatal();
    }
}

/// 69.
pub(super) fn info_indirect<P: Platform>(c: &mut Command<'_, P>) {
    let hold = c.hold;
    let Some((task, rc, cbc)) = c.f_class(|c, _| prologue(c, Kind::Info)) else {
        return;
    };
    let (resources, cmd_buffers) = records::<P, CmdExecIndirectResource3>(c, rc, cbc, RESOURCE3);
    let done = c.completion();
    // Outside any handler.
    if let Err(r) = task.info_indirect(
        hold,
        ExecArgs {
            cmd_buffers,
            resources,
            stamp_index: c.fifo.stamp_index(),
            fifo_event: c.fifo.event.as_ref(),
            fifo_event_value: &c.fifo.event_value,
        },
        done,
    ) {
        // Never reached: with no handler the raise died at its own site (module doc)
        r.fatal();
    }
}

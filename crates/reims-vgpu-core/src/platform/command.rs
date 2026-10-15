//! Platform trait: `CommandQueue`, `CommandBuffer`: queue and command-buffer selectors.

use super::event::EventRef;
use super::types::{CommandBufferStatus, NsErrorText};
use super::values::ComputePassDesc;
use super::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};

/// What a completed handler learns of its buffer: core's handlers read `status` and `error` of the
/// buffer they were added to, on Metal's completion thread. The platform reads both when the
/// handler runs; neither getter has an effect.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompletedInfo {
    /// `-[MTLCommandBuffer status]` when the handler runs.
    pub status: CommandBufferStatus,
    /// `-[MTLCommandBuffer error]` when the handler runs; `None` for nil.
    pub error: Option<NsErrorText>,
}

/// A completed handler (`addCompletedHandler:`'s block). Runs once, on a Metal thread (4.2 item 6).
pub type CompletedHandler = Box<dyn FnOnce(CompletedInfo) + Send>;

/// `id<MTLCommandQueue>`: the device's `reims-vgpu.blit-up`, `reims-vgpu.blit-down` and
/// `reims-vgpu.exec`, and the service's queues.
/// Core never nil-checks one.
pub trait CommandQueue<P: Platform>: Handle {
    /// sel: commandBuffer — core never checks for nil; later sends
    /// go to nil (`nil_send!`).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn command_buffer(&self, on: OnFault) -> RaiseResult<Option<P::CommandBuffer>>;
    /// sel: setLabel: — (`reims-vgpu.blit-up`, `reims-vgpu.blit-down`,
    /// `reims-vgpu.exec`). No path reaches it under a handler: a raise inside it dies at the send.
    fn set_label(&self, label: &str);
}

/// `id<MTLCommandBuffer>` (`MTLCommandBufferSPI` included).
pub trait CommandBuffer<P: Platform>: Handle {
    /// sel: addCompletedHandler:
    /// The handler runs on a Metal thread after completion or error; a thread with no handler
    /// (6.2).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn add_completed_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()>;
    /// sel: blitCommandEncoder — May return nil, which core never
    /// checks (the blit decoder's init); later sends to it are no-ops (`nil_send!`).
    ///
    /// Reached both under a handler and with none: `on` says which this call is.
    fn blit_encoder(&self, on: OnFault) -> RaiseResult<Option<P::BlitEncoder>>;
    /// sel: commit
    /// Reached both under a handler and with none: `on` says which this call is.
    fn commit(&self, on: OnFault) -> RaiseResult<()>;
    /// sel: computeCommandEncoder
    /// Reached both under a handler and with none: `on` says which this call is.
    fn compute_encoder(&self, on: OnFault) -> RaiseResult<Option<P::ComputeEncoder>>;
    /// sel: device
    fn device(&self) -> Option<P::Device>;
    /// sel: encodeSignalEvent:value:
    /// Signals `value` when the buffer completes.
    fn encode_signal(&self, event: Option<EventRef<'_, P>>, value: u64) -> RaiseResult<()>;
    /// sel: encodeSignalEventScheduled:value: SPI
    /// Undeclared anywhere; runtime encoding `v32@0:8@16Q24`, so `value` is `u64`
    /// (`metal/tests/spi_encodings.rs`). Signals `value` when the buffer is scheduled.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&P::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()>;
    /// sel: encodeWaitForEvent:value:
    fn encode_wait(&self, event: Option<EventRef<'_, P>>, value: u64) -> RaiseResult<()>;
    /// sel: error — `None` for nil.
    fn error(&self) -> Option<NsErrorText>;
    /// sel: renderCommandEncoderWithDescriptor:
    /// Core throws on nil. aborts: Metal's validation on some descriptors (render_pass_no_target,
    /// testing.md 7.4).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &P::RenderPassDescriptor,
    ) -> RaiseResult<Option<P::RenderEncoder>>;
    /// sel: setProtectionOptions: SPI — (the type-5 envelope),
    /// Declared on `MTLCommandBufferSPI` (`uint64_t`).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()>;
    /// sel: status — Raw `MTLCommandBufferStatus`.
    fn status(&self) -> CommandBufferStatus;
    /// sel: waitUntilCompleted
    /// Blocks.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()>;
    /// sel: waitUntilScheduled — Blocks.
    fn wait_until_scheduled(&self) -> RaiseResult<()>;
    /// sel: computeCommandEncoderWithDescriptor: — from a compute pass descriptor made from
    /// `desc`.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<P::ComputeEncoder>>;
}

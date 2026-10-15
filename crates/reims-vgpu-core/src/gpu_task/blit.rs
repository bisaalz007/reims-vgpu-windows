//! The XR10 blit pipelines: made once per task, when a resource first asks for one, from
//! the platform's shader library.
//!
//! No handler is live in the once's block, so a Metal raise inside it ends the task at the send
//! even under a caller's handler; a missing library or pipeline ends it too. The block holds under
//! [`Hold::NONE`](crate::rt::Hold::NONE), with no holder: it cannot take its caller's hold. The
//! library and the functions are released at the once's end; when the pipelines are made moves the
//! GPU's addresses, so they are never made early.

use super::service::GpuTask;
use super::{DeathKind, unraised};
use crate::fatal;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Library;
use crate::raise::OnFault;

/// The two XR10 blit pipelines.
pub(super) struct BlitPipelines<P: Platform> {
    /// `BlitInRGB_2P_XR10_A8`.
    pub(super) blit_in: P::ComputePipeline,
    /// `BlitOutRGB_2P_XR10_A8`.
    pub(super) blit_out: P::ComputePipeline,
}

impl<P: Platform> GpuTask<P> {
    /// The task's blit pipelines, made on the first call: the library, then the in
    /// function and its pipeline, then the out function and its pipeline.
    pub(super) fn blit_pipelines(&self) -> &BlitPipelines<P> {
        self.blit.get_or_init(|| {
            let device = self.device_handle();
            let library = match &device {
                Some(d) => d.new_shader_library().ok().flatten(),
                None => None,
            };
            let Some(library) = library else {
                fatal!(DeathKind::NoBlitLibrary)
            };
            let blit_in = unraised(library.new_function(OnFault::Die, "BlitInRGB_2P_XR10_A8"));
            let blit_in_pipeline = device
                .as_ref()
                .map(|d| d.new_compute_pipeline_state_for_function(blit_in.as_ref()))
                .and_then(Result::ok)
                .flatten();
            let Some(blit_in_pipeline) = blit_in_pipeline else {
                fatal!(DeathKind::NoBlitInPipeline)
            };
            let blit_out = unraised(library.new_function(OnFault::Die, "BlitOutRGB_2P_XR10_A8"));
            let blit_out_pipeline = device
                .as_ref()
                .map(|d| d.new_compute_pipeline_state_for_function(blit_out.as_ref()))
                .and_then(Result::ok)
                .flatten();
            let Some(blit_out_pipeline) = blit_out_pipeline else {
                fatal!(DeathKind::NoBlitOutPipeline)
            };
            let pipelines = BlitPipelines {
                blit_in: blit_in_pipeline,
                blit_out: blit_out_pipeline,
            };
            // The functions, the library and the device go
            drop(blit_out);
            drop(blit_in);
            drop(library);
            drop(device);
            pipelines
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::platform::Handle;
    use crate::raise::death::DeathKind as Fault;
    use crate::resources::host::ResourceHost;
    use crate::testkit::capture::catch_death;
    use crate::testkit::script::Step;

    /// One library per task; the in function comes first
    #[test]
    fn made_once_per_task() {
        let tx = ready(&rw_pages(4));
        let mark = tx.p.call_log().mark();
        assert!(tx.task.blit_in_rgb_2p_xr10_a8_pipeline().is_some());
        assert!(tx.task.blit_out_rgb_2p_xr10_a8_pipeline().is_some());
        let sels = tx.p.call_log().sels_since(mark);
        assert_eq!(
            sels.iter()
                .filter(|s| **s == "newLibraryWithSource:options:error:")
                .count(),
            1
        );
        let calls = tx.p.call_log().calls_to("newFunctionWithName:");
        let names: Vec<String> = calls.iter().map(|c| format!("{:?}", c.args)).collect();
        assert!(names[0].contains("BlitInRGB_2P_XR10_A8"));
        assert!(names[1].contains("BlitOutRGB_2P_XR10_A8"));
        let pos = |s: &str| sels.iter().position(|x| *x == s).unwrap();
        assert!(pos("newFunctionWithName:") < pos("newComputePipelineStateWithFunction:error:"));
    }

    /// No library, no in pipeline, no out pipeline: each ends the task
    #[test]
    fn a_missing_part_dies() {
        let tx = ready(&rw_pages(4));
        tx.p.script()
            .push("newLibraryWithSource:options:error:", Step::Nil);
        assert_eq!(
            catch_death(|| tx.task.blit_pipelines().blit_in.clone_ref()).map(|_| ()),
            Err(DeathKind::NoBlitLibrary.site())
        );
        let tx = ready(&rw_pages(4));
        tx.p.script()
            .push("newComputePipelineStateWithFunction:error:", Step::Nil);
        assert_eq!(
            catch_death(|| tx.task.blit_pipelines().blit_in.clone_ref()).map(|_| ()),
            Err(DeathKind::NoBlitInPipeline.site())
        );
        let tx = ready(&rw_pages(4));
        let sel = "newComputePipelineStateWithFunction:error:";
        tx.p.script().push(sel, Step::Default);
        tx.p.script().push(sel, Step::Nil);
        assert_eq!(
            catch_death(|| tx.task.blit_pipelines().blit_in.clone_ref()).map(|_| ()),
            Err(DeathKind::NoBlitOutPipeline.site())
        );
    }

    /// A Metal raise inside the once dies at the send even under a live handler, never
    /// reaching the caller's handler
    #[test]
    fn a_raise_inside_is_uncaught_under_a_handler() {
        for sel in [
            "newLibraryWithSource:options:error:",
            "newComputePipelineStateWithFunction:error:",
        ] {
            let tx = ready(&rw_pages(4));
            tx.p.script()
                .push(sel, Step::raise("NSInvalidArgumentException", "x"));
            let d = catch_death(|| tx.task.blit_pipelines().blit_in.clone_ref())
                .map(|_| ())
                .unwrap_err();
            assert_eq!((d.kind, d.file, d.line), (Fault::UncaughtObjc, sel, 0));
        }
    }
}

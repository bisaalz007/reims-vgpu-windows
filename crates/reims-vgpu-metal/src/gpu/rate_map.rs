//! Rasterization rate maps: core's `RateMap`, `RateMapDescriptor` and `RateMapLayerDescriptor` in
//! typed calls.
//!
//! Every method is one call through the message layer: a getter's or a descriptor property's raise ends
//! the process at the call; a raise inside a method with arguments is the caller's to catch.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::types::*;
use reims_vgpu_core::platform::values::RateMapDesc;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::msg::encode::{FromMtl, ToMtl};
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::{RateLayerDescriptorSpi, RateMapDescriptorSpi, RateMapSpi};

/// `id<MTLRasterizationRateMap>`.
pub type MetalRateMap = Obj<ProtocolObject<dyn MTLRasterizationRateMap>>;

impl reims_vgpu_core::platform::rate_map::RateMap<MetalPlatform> for MetalRateMap {
    /// Metal checks the buffer's room itself.
    fn copy_parameter_data(
        &self,
        buffer: &<MetalPlatform as Platform>::Buffer,
        offset: u64,
    ) -> RaiseResult<()> {
        let (m, buffer, offset) = (self.get(), buffer.get(), offset.to_usize());
        // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
        msg::send(
            "copyParameterDataToBuffer:offset:",
            Policy::Caught,
            || unsafe { m.copyParameterDataToBuffer_offset(buffer, offset) },
        )
    }

    fn layer_count(&self) -> u64 {
        let m = self.get();
        msg::get("layerCount", || m.layerCount()).to_u64()
    }

    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let (m, point, layer) = (
            self.get(),
            physical_coordinates.mtl(),
            layer_index.to_usize(),
        );
        // SAFETY: Metal checks the layer index itself.
        msg::send(
            "mapPhysicalToScreenCoordinates:forLayer:",
            Policy::Caught,
            || unsafe { m.mapPhysicalToScreenCoordinates_forLayer(point, layer) },
        )
        .map(MtlCoordinate2D::from_mtl)
    }

    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let (m, point, layer) = (self.get(), screen_coordinates.mtl(), layer_index.to_usize());
        // SAFETY: as above.
        msg::send(
            "mapScreenToPhysicalCoordinates:forLayer:",
            Policy::Caught,
            || unsafe { m.mapScreenToPhysicalCoordinates_forLayer(point, layer) },
        )
        .map(MtlCoordinate2D::from_mtl)
    }

    fn mutability(&self) -> u64 {
        let m = self.get();
        msg::get("mutability", || RateMapSpi::mutability(m))
    }

    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        let m = self.get();
        SizeAndAlign::from_mtl(msg::get("parameterBufferSizeAndAlign", || {
            m.parameterBufferSizeAndAlign()
        }))
    }

    fn physical_granularity(&self) -> MtlSize {
        let m = self.get();
        MtlSize::from_mtl(msg::get("physicalGranularity", || m.physicalGranularity()))
    }

    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize> {
        let (m, layer) = (self.get(), layer_index.to_usize());
        // SAFETY: as above.
        msg::send("physicalSizeForLayer:", Policy::Caught, || unsafe {
            m.physicalSizeForLayer(layer)
        })
        .map(MtlSize::from_mtl)
    }

    fn reset(&self, descriptor: &<MetalPlatform as Platform>::RateMapDescriptor) {
        let (m, descriptor) = (self.get(), descriptor.get());
        msg::get("resetUsingDescriptor:", || {
            m.resetUsingDescriptor(descriptor)
        });
    }
}

/// `MTLRasterizationRateMapDescriptor *`.
pub type MetalRateMapDescriptor = Obj<MTLRasterizationRateMapDescriptor>;

impl reims_vgpu_core::platform::rate_map::RateMapDescriptor<MetalPlatform>
    for MetalRateMapDescriptor
{
    fn set_layer(
        &self,
        on: OnFault,
        layer: &<MetalPlatform as Platform>::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let (d, layer, index) = (self.get(), layer.get(), layer_index.to_usize());
        // SAFETY: Metal checks the index itself.
        msg::send("setLayer:atIndex:", Policy::On(on), || unsafe {
            d.setLayer_atIndex(Some(layer), index)
        })
    }

    fn layer_count(&self) -> u64 {
        let d = self.get();
        msg::get("layerCount", || d.layerCount()).to_u64()
    }
    /// `layers` (`+0`, retained), then its `objectAtIndexedSubscript:`.
    fn layer(
        &self,
        on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::RateMapLayerDescriptor>> {
        let d = self.get();
        let layers: MetalRateMapLayerArray = Obj::from_retained(msg::get("layers", || d.layers()));
        layer_at(on, &layers, layer_index)
    }
}

/// A rasterization-rate-map descriptor holding `desc`: `+new`, then its screen size, its
/// mutability when set, and its minimum factor when set, in that order. Its layers are set on it
/// after.
pub fn rate_map_descriptor(desc: &RateMapDesc) -> MetalRateMapDescriptor {
    let d = crate::platform::class_new(MTLRasterizationRateMapDescriptor::new);
    let o = d.get();
    let size = desc.screen_size.mtl();
    // SAFETY: Metal checks the size when it makes the map.
    msg::get("setScreenSize:", || unsafe { o.setScreenSize(size) });
    if let Some(mutability) = desc.mutability {
        msg::get("setMutability:", || o.setMutability(mutability));
    }
    if let Some(factor) = desc.min_factor {
        msg::get("setMinFactor:", || o.setMinFactor(factor));
    }
    d
}

/// A layer array's `objectAtIndexedSubscript:`: the layer at `+0`, retained, nil for an index
/// past the layers.
fn layer_at(
    on: OnFault,
    layers: &MetalRateMapLayerArray,
    layer_index: u64,
) -> RaiseResult<Option<MetalRateMapLayerDescriptor>> {
    let (a, index) = (layers.get(), layer_index.to_usize());
    // SAFETY: Metal answers nil past the layers.
    msg::send("objectAtIndexedSubscript:", Policy::On(on), || unsafe {
        a.objectAtIndexedSubscript(index)
    })
    .map(|l| l.map(Obj::from_retained))
}

/// `MTLRasterizationRateLayerDescriptor *`.
pub type MetalRateMapLayerDescriptor = Obj<MTLRasterizationRateLayerDescriptor>;

impl reims_vgpu_core::platform::rate_map::RateMapLayerDescriptor for MetalRateMapLayerDescriptor {
    fn horizontal_sample_storage(&self) -> HostPtr {
        let l = self.get();
        HostPtr::from_ptr(msg::get("horizontalSampleStorage", || {
            l.horizontalSampleStorage_or_null()
        }))
    }

    fn sample_count(&self) -> MtlSize {
        let l = self.get();
        MtlSize::from_mtl(msg::get("sampleCount", || {
            RateLayerDescriptorSpi::sampleCount(l)
        }))
    }

    fn vertical_sample_storage(&self) -> HostPtr {
        let l = self.get();
        HostPtr::from_ptr(msg::get("verticalSampleStorage", || {
            l.verticalSampleStorage_or_null()
        }))
    }
}

/// `MTLRasterizationRateLayerArray *`.
pub type MetalRateMapLayerArray = Obj<MTLRasterizationRateLayerArray>;

#[cfg(test)]
mod tests {

    use objc2_metal::MTLRasterizationRateMapDescriptor;
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::rate_map::*;

    use super::*;
    use crate::gpu::resource::testkit::*;

    // BEGIN GENERATED TESTS

    /// Every RateMap method
    #[test]
    fn rate_map_sends() {
        let k = kit();
        let x = k.rate_map;
        let r = x.copy_parameter_data(&k.buffer, 0);
        assert!(r.is_ok(), "copy_parameter_data: {r:?}");
        assert_eq!(x.layer_count(), raw_u64(obj(&x), "layerCount"));
        let r = x.physical_to_screen(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0);
        assert!(r.is_ok(), "physical_to_screen: {r:?}");
        let r = x.screen_to_physical(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0);
        assert!(r.is_ok(), "screen_to_physical: {r:?}");
        assert_eq!(x.mutability(), raw_u64(obj(&x), "mutability"));
        let _ = x.parameter_buffer_size_and_align();
        let _ = x.physical_granularity();
        let r = x.physical_size(0);
        assert!(r.is_ok(), "physical_size: {r:?}");
        x.reset(&k.rate_map_descriptor);
    }

    /// Every RateMapDescriptor method, and its layer's
    #[test]
    fn rate_map_descriptor_sends() {
        let k = kit();
        let x = rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(1, 1, 1),
            ..RateMapDesc::default()
        });
        assert_eq!(x.layer_count(), raw_u64(obj(&x), "layerCount"));
        let r = x.set_layer(OnFault::Return, &k.rate_layer, 0);
        assert!(r.is_ok(), "set_layer: {r:?}");
        let l = x.layer(OnFault::Return, 0).unwrap().unwrap();
        assert_eq!(l.sample_count(), MtlSize::new(2, 2, 0));
        assert!(!l.horizontal_sample_storage().is_null());
        assert!(!l.vertical_sample_storage().is_null());
    }
    // END GENERATED TESTS

    /// The rate map's values: one layer of the 64x64 screen, a parameter buffer Metal sizes, a
    /// physical size no larger than the screen, the identity at the origin both ways, and nil for a
    /// layer index past the descriptor's layers
    #[test]
    fn rate_map_values() {
        let k = kit();
        let m = &k.rate_map;
        assert_eq!(m.layer_count(), 1);
        let sa = m.parameter_buffer_size_and_align();
        assert!(sa.size > 0 && sa.align > 0, "{sa:?}");
        let g = m.physical_granularity();
        assert!(g.width > 0 && g.height > 0, "{g:?}");
        let p = m.physical_size(0).unwrap();
        assert!(p.width <= 64 && p.height <= 64 && p.width > 0, "{p:?}");
        let origin = MtlCoordinate2D { x: 0.0, y: 0.0 };
        assert_eq!(m.screen_to_physical(origin, 0).unwrap(), origin);
        assert_eq!(m.physical_to_screen(origin, 0).unwrap(), origin);
        // A map made from a descriptor that set none is immutable (2, measured).
        assert_eq!(m.mutability(), 2);
    }

    /// A rate map descriptor built from a value holds its screen size, mutability and minimum
    /// factor; the typed reads count and find the layers set on it
    #[test]
    fn rate_map_descriptor_from_a_value() {
        let k = kit();
        let d = rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(32, 16, 0),
            mutability: Some(1),
            min_factor: Some(0.5),
        });
        assert_eq!(
            MtlSize::from_mtl(raw::<MTLSize>(obj(&d), "screenSize")),
            MtlSize::new(32, 16, 0)
        );
        assert_eq!(raw_u64(obj(&d), "mutability"), 1);
        assert_eq!(raw::<f32>(obj(&d), "minFactor"), 0.5);
        assert_eq!(d.layer_count(), 0);
        assert!(d.layer(OnFault::Return, 0).unwrap().is_none());
        d.set_layer(OnFault::Return, &k.rate_layer, 0).unwrap();
        assert_eq!(d.layer_count(), 1);
        let l = d.layer(OnFault::Return, 0).unwrap().unwrap();
        assert_eq!(l.id(), k.rate_layer.id());
        assert!(d.layer(OnFault::Return, 3).unwrap().is_none());
        // What is not set keeps Metal's own value.
        let plain = rate_map_descriptor(&RateMapDesc::default());
        let fresh: MetalRateMapDescriptor =
            Obj::from_retained(MTLRasterizationRateMapDescriptor::new());
        assert_eq!(
            raw_u64(obj(&plain), "mutability"),
            raw_u64(obj(&fresh), "mutability")
        );
        assert_eq!(
            raw::<f32>(obj(&plain), "minFactor"),
            raw::<f32>(obj(&fresh), "minFactor")
        );
    }
}

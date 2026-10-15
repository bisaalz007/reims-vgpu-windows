//! Reservations: the argument-buffer indices the guest placed its textures, samplers and ICBs
//! at, reserved on the device before the task runs.
//!
//! The walk reads every object entry with its placement slot (the count is read again each
//! step, index 0 included) and stops at the first either list refuses. A pinned buffer (a buffer or
//! heap whose slot's first word is set) is made and discarded; a texture, sampler or ICB with an id
//! in its slot has the slot's first word reserved. Then textures, samplers and ICBs are reserved,
//! in that order, each only if any. The whole walk is one landing frame ([`Held::raising`]) with
//! no handler: what its guest reads hold (the page-table read's device handle, the kernel task a
//! backing lookup reaches) is released at its end, and a raise carries it on to the enclosing
//! holder and reserves nothing.

use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::host::HostPtr;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::raise::RaiseResult;
use crate::resources::manager::create::Resources;
use crate::resources::manager::records::OperationHeader;
use crate::resources::manager::{DeathKind, ResourceManager};
use crate::rt::{Held, Hold};

/// The indices to reserve, by resource type: textures (type 0), samplers (1) and ICBs (5).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReservationTable {
    /// Type 0.
    pub texture_indices: Vec<u64>,
    /// Type 1.
    pub sampler_indices: Vec<u64>,
    /// Type 5.
    pub indirect_command_buffer_indices: Vec<u64>,
}

/// The resource types the device reserves indices of.
const TEXTURE_INDICES: u64 = 0;
const SAMPLER_INDICES: u64 = 1;
const ICB_INDICES: u64 = 5;

/// The sampler state's creation command type.
const SAMPLER_COMMAND: u32 = 3;

impl<P: Platform> ResourceManager<P> {
    /// Reserves the placed objects' indices (module doc). A raise leaves with nothing reserved.
    pub fn reserve_objects(&self) -> RaiseResult<Result<(), String>> {
        let mut discards: Vec<u32> = Vec::new();
        let mut table = ReservationTable::default();
        Held::raising(|held| -> RaiseResult<()> {
            let hold = Hold::of(held);
            let lists = &self.object_lists;
            let mut index: u32 = 0;
            while index < lists.object_list_count() {
                // Both read, the entry first
                let entry = lists.get_object_entry(index);
                let slot = lists.get_placement_list_entry(index);
                let (Ok(entry), Ok(slot)) = (entry, slot) else {
                    break;
                };
                let slot = slot.unwrap_or(GuestPtr::new(HostPtr::NULL));
                let null = DeathKind::NullPlacement;
                // Each case's test is a guard: a failed one falls to the arm that does nothing
                match entry.object_type() {
                    1 | 13 if slot.read_u64(0, null) != 0 => discards.push(index),
                    2 | 3 | 8 | 9 | 12 | 16 if slot.read_u64(8, null) != 0 => {
                        table.texture_indices.push(slot.read_u64(0, null));
                    }
                    7 if slot.read_u64(8, null) != 0 => {
                        self.reserve_object(hold, index, &entry, slot, &mut table)?;
                    }
                    20 if slot.read_u64(8, null) != 0 => {
                        table
                            .indirect_command_buffer_indices
                            .push(slot.read_u64(0, null));
                    }
                    // 11: mapper-ref textures are never reserved
                    _ => {}
                }
                index = index.wrapping_add(1);
            }
            for reference in discards {
                // A lookup error is ignored; a raise leaves
                if let Ok(resource) = self.get_object::<Resources>(hold, reference)? {
                    resource.as_deref().map_or(Ok(()), |r| r.discard())?;
                }
            }
            self.finalize_reservations(&mut table)?;
            Ok(())
        })?;
        Ok(Ok(()))
    }

    /// Reserves a sampler's index if the type-7 entry's record is a sampler state's creation;
    /// `slot` is read again at each use.
    pub(crate) fn reserve_object(
        &self,
        hold: Hold<'_>,
        _reference: u32,
        entry: &ObjectEntry,
        slot: GuestPtr,
        table: &mut ReservationTable,
    ) -> RaiseResult<bool> {
        if entry.object_type() != 7 {
            return Ok(false);
        }
        let offset = entry.offset;
        let Some(mut cursor) = self.new_memory_cursor(hold, offset, u64::from(entry.length()))?
        else {
            return Ok(false);
        };
        match cursor.read::<OperationHeader>() {
            Ok(header) if header.command_type == SAMPLER_COMMAND => {}
            _ => return Ok(false),
        }
        let null = DeathKind::NullPlacement;
        if slot.read_u64(8, null) != 0 {
            table.sampler_indices.push(slot.read_u64(0, null));
        }
        Ok(true)
    }

    /// Reserves the table's indices on the device: textures, samplers, then ICBs, each only if
    /// any. The first raise leaves, and the later types are not reserved.
    pub fn finalize_reservations(&self, table: &mut ReservationTable) -> RaiseResult<bool> {
        for (kind, indices) in [
            (TEXTURE_INDICES, &table.texture_indices),
            (SAMPLER_INDICES, &table.sampler_indices),
            (ICB_INDICES, &table.indirect_command_buffer_indices),
        ] {
            if !indices.is_empty() {
                self.device
                    .as_deref()
                    .map_or(Ok(()), |d| d.reserve_resource_indices(kind, indices))?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::*;
    use super::*;
    use crate::resources::texture::views::SerializerTextureResource;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Step, kind};
    use zerocopy::IntoBytes;

    const RESERVE: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    fn reserves(fx: &Fx) -> Vec<Vec<Arg>> {
        fx.log()
            .calls_to(RESERVE)
            .into_iter()
            .map(|c| c.args)
            .collect()
    }

    fn header(fx: &Fx, off: u64, command_type: u32) {
        fx.write(
            REC + off,
            OperationHeader {
                command_type,
                field4: 0,
            }
            .as_bytes(),
        );
    }

    /// Index 0 included, textures then samplers then ICBs.
    #[test]
    fn collects_textures_samplers_and_icbs_in_type_order() {
        let entries = vec![
            (2, 0, 0),          // 0: texture (index 0 included)
            (7, 8, REC),        // 1: sampler command
            (20, 0, 0),         // 2: ICB
            (3, 0, 0),          // 3: texture, no id: skipped
            (7, 8, REC + 0x10), // 4: compute pipeline command: not reserved
            (11, 0, 0),         // 5: mapper-ref texture: never
            (16, 0, 0),         // 6: shared texture
            (7, 4, REC + 0x20), // 7: short record: not reserved
            (5, 0, 0),          // 8: others: nothing
            (1, 0, 0),          // 9: buffer, first word 0: no discard
            (12, 0, 0),         // 10: texture, id only
        ];
        let fx = Fx::new(&entries, Placement::Mapped);
        header(&fx, 0, 3);
        header(&fx, 0x10, 11);
        fx.set_placement(0, 0x10, 1);
        fx.set_placement(1, 0x20, 2);
        fx.set_placement(2, 0x30, 3);
        fx.set_placement(3, 0x40, 0);
        fx.set_placement(4, 0x50, 5);
        fx.set_placement(5, 0x60, 6);
        fx.set_placement(6, 0x70, 7);
        fx.set_placement(7, 0x80, 8);
        fx.set_placement(8, 0x90, 9);
        fx.set_placement(10, 0, 11);
        fx.next_cursor(0, 8);
        fx.next_cursor(0x10, 8);
        fx.next_cursor(0x20, 4);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert_eq!(
            reserves(&fx),
            vec![
                vec![Arg::U64(0), Arg::U64s(vec![0x10, 0x70, 0])],
                vec![Arg::U64(1), Arg::U64s(vec![0x20])],
                vec![Arg::U64(5), Arg::U64s(vec![0x30])],
            ]
        );
        // Only type 7 with an id reads its record.
        assert_eq!(fx.log().calls_to("cursor_from_virtual_offset").len(), 3);
    }

    #[test]
    fn a_type_7_without_an_id_reads_nothing() {
        let fx = Fx::new(&[(7, 8, REC)], Placement::Mapped);
        header(&fx, 0, 3);
        fx.set_placement(0, 0x20, 0);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert!(fx.log().calls_to("cursor_from_virtual_offset").is_empty());
        assert!(reserves(&fx).is_empty());
    }

    #[test]
    fn a_null_cursor_reserves_nothing() {
        let fx = Fx::new(&[(7, 8, REC)], Placement::Mapped);
        fx.set_placement(0, 0x20, 1);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert!(reserves(&fx).is_empty());
    }

    #[test]
    fn an_unreadable_entry_stops_the_scan() {
        let fx = Fx::new(&[(2, 0, 0), (2, 0, 0)], Placement::Unmapped);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert!(reserves(&fx).is_empty());
        let fx = Fx::new(&[(2, 0, 0), (2, 0, 0)], Placement::Mapped);
        fx.set_placement(0, 1, 1);
        fx.olm.unmap_object_list();
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert!(reserves(&fx).is_empty());
    }

    #[test]
    fn a_null_placement_faults_at_the_load() {
        let fx = Fx::new(&[(1, 0, 0)], Placement::Off);
        let m = fx.m.clone();
        assert_eq!(
            catch_death(move || {
                let _ = m.reserve_objects();
            }),
            Err(DeathKind::NullPlacement.site())
        );
    }

    #[test]
    fn a_metal_throw_escapes() {
        let fx = Fx::new(&[(2, 0, 0)], Placement::Mapped);
        fx.set_placement(0, 4, 4);
        fx.platform
            .script()
            .push(RESERVE, Step::raise("NSInvalidArgumentException", "x"));
        assert!(fx.m.reserve_objects().is_err());
    }

    #[test]
    fn pinned_buffers_are_created_and_discarded_before_reserving() {
        let fx = Fx::new(&[(1, 0x10, REC), (2, 0, 0)], Placement::Mapped);
        fx.set_placement(0, 0x10_0000, 0);
        fx.set_placement(1, 3, 3);
        fx.next_cursor(0, 0x10);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert!(fx.m.get_existing_object::<Resources>(0).is_some());
    }

    /// A normal exit reserves; a raise (here the resource's discard) reserves nothing.
    #[test]
    fn a_raise_in_the_discards_reserves_nothing() {
        // Reference 1: reference 0 is nil.
        let setup = |discardable: bool| {
            let fx = Fx::new(&[(2, 0, 0), (1, 0x10, REC)], Placement::Mapped);
            fx.set_placement(0, 3, 3);
            fx.set_placement(1, 0x10_0000, 0);
            let t = fx
                .platform
                .world()
                .new_obj::<kind::Texture>(Default::default());
            let r = SerializerTextureResource::new(t, discardable);
            assert!(fx.m.register_object::<Resources>(1, r));
            fx
        };
        let fx = setup(true);
        assert_eq!(fx.m.reserve_objects(), Ok(Ok(())));
        assert_eq!(reserves(&fx).len(), 1);

        let fx = setup(false);
        {
            {
                assert!(fx.m.reserve_objects().is_err());
            }
        }
        assert!(reserves(&fx).is_empty());
    }

    #[test]
    fn finalize_reserves_in_type_order_and_skips_empty_tables() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let mut t = ReservationTable::default();
        assert!(fx.m.finalize_reservations(&mut t).unwrap());
        assert!(reserves(&fx).is_empty());
        t.indirect_command_buffer_indices = vec![9];
        t.texture_indices = vec![1, 2];
        assert!(fx.m.finalize_reservations(&mut t).unwrap());
        assert_eq!(
            reserves(&fx),
            vec![
                vec![Arg::U64(0), Arg::U64s(vec![1, 2])],
                vec![Arg::U64(5), Arg::U64s(vec![9])],
            ]
        );
    }

    /// The first raise leaves and the later types are not reserved.
    #[test]
    fn finalize_propagates_a_metal_raise() {
        let fx = Fx::new(&entries3(), Placement::Off);
        fx.platform
            .script()
            .push(RESERVE, Step::raise("NSInvalidArgumentException", "x"));
        let mut t = ReservationTable {
            texture_indices: vec![1],
            sampler_indices: vec![2],
            ..Default::default()
        };
        assert!(fx.m.finalize_reservations(&mut t).is_err());
        assert_eq!(reserves(&fx).len(), 1);
    }

    #[test]
    fn reserve_object_refuses_other_types() {
        let fx = Fx::new(&entries3(), Placement::Mapped);
        let mut t = ReservationTable::default();
        let e = ObjectEntry {
            type_and_length: 2 | (8 << 8),
            offset: REC,
        };
        let p = GuestPtr::new(fx.addr(PL));
        assert_eq!(fx.m.reserve_object(Hold::NONE, 0, &e, p, &mut t), Ok(false));
        assert!(fx.log().calls_to("cursor_from_virtual_offset").is_empty());
    }
}

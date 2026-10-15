//! The links between areas: one alias per cross-area handle, naming what an area holds on the
//! other side (architecture 9). The device's FIFOs, tasks and mappings hold it unretained, so a
//! link upgrades while the device's shutdown runs and answers nothing after.

use std::sync::{Arc, Weak};

use crate::device::{DeviceCore, MappingSink};
use crate::resources::manager::lookup::ManagerLookup;
use crate::tasks::local::LocalTask;

/// A FIFO's device (`Fifo::device`), unretained: it upgrades until the last reference to the
/// device's core goes (through the device's shutdown), and answers nothing after.
pub(crate) type FifoDeviceLink<P> = Weak<DeviceCore<P>>;

/// A mapping's device (`memory::mapping::Teardown::{RootTask, Task}`), unretained: a mapping
/// dropped during the device's teardown, its fields' drops included, still unmaps; one dropped
/// after it skips the unmap.
///
/// The device's [`MappingSink`]: the part of the device that outlives the core's other fields. A
/// `Weak<DeviceCore<P>>` would stop upgrading when the core's fields start to drop, so a mapping
/// those drops release would skip its unmap.
pub(crate) type MappingDeviceLink<P> = Weak<MappingSink<P>>;

/// A task's device (`LocalTask`, `RemoteTask` and the remote task's delegate), unretained, as
/// a FIFO's.
pub(crate) type TaskDeviceLink<P> = Weak<DeviceCore<P>>;

/// The kernel task (task 0), as the device's lookups hand it to the FIFO, the displays and the
/// tasks: always local (the lookup's remote arm is dead).
pub(crate) type KernelTaskRef<P> = Arc<LocalTask<P>>;

/// The deserializer's and the resource manager's object lookup: the manager's own lookup type.
pub(crate) type ObjectLookupRef<P> = Arc<ManagerLookup<P>>;

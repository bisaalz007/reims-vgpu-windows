//! [`BucketOrderMap`]: a `u32`-keyed map that iterates, drains and drops in ascending key order.
//!
//! The task table is one, keyed by task ID. Its order is the order of every fan-out across
//! tasks (the `unmapMemory`/`destroyTask` pairs of a teardown the host sees) and of the releases
//! when the table empties: ascending task ID.

use std::collections::BTreeMap;

/// A map from `u32` keys whose [`values`](Self::values) and [`drain`](Self::drain) go in
/// ascending key order. Dropping it drops the values in that order too.
#[derive(Debug)]
pub struct BucketOrderMap<V> {
    map: BTreeMap<u32, V>,
}

impl<V> Default for BucketOrderMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> BucketOrderMap<V> {
    /// An empty map.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// The value under `key`.
    pub fn get(&self, key: u32) -> Option<&V> {
        self.map.get(&key)
    }

    /// Stores `v` under `key` and returns the value it replaced.
    pub fn insert(&mut self, key: u32, v: V) -> Option<V> {
        self.map.insert(key, v)
    }

    /// Removes `key` and returns its value.
    pub fn remove(&mut self, key: u32) -> Option<V> {
        self.map.remove(&key)
    }

    /// Removes every key and returns the values in ascending key order, so that the caller
    /// decides where they drop.
    pub fn drain(&mut self) -> Vec<V> {
        std::mem::take(&mut self.map).into_values().collect()
    }

    /// The values in ascending key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the map holds no key.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<V> Drop for BucketOrderMap<V> {
    /// The values drop in ascending key order, one at a time.
    fn drop(&mut self) {
        while let Some((_, v)) = self.map.pop_first() {
            drop(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Pushes its key to a shared log when dropped.
    struct Tok(u32, Rc<RefCell<Vec<u32>>>);

    impl Drop for Tok {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    /// Keys whose insertion order is neither ascending nor descending.
    const KEYS: [u32; 9] = [17, 3, 40, 9, 0, 255, 25, 12, 1];

    fn ascending() -> Vec<u32> {
        let mut k = KEYS.to_vec();
        k.sort_unstable();
        k
    }

    fn filled(log: &Rc<RefCell<Vec<u32>>>) -> BucketOrderMap<Tok> {
        let mut m = BucketOrderMap::new();
        for k in KEYS {
            assert!(m.insert(k, Tok(k, log.clone())).is_none());
        }
        m
    }

    #[test]
    fn looks_up_inserts_replaces_and_removes() {
        let mut m: BucketOrderMap<&str> = BucketOrderMap::default();
        assert!(m.is_empty());
        assert_eq!(m.get(1), None);
        assert_eq!(m.insert(1, "a"), None);
        assert_eq!(m.insert(1, "b"), Some("a"));
        assert_eq!(m.insert(2, "c"), None);
        assert_eq!(m.get(1), Some(&"b"));
        assert_eq!(m.len(), 2);
        assert_eq!(m.remove(1), Some("b"));
        assert_eq!(m.remove(1), None);
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
    }

    #[test]
    fn values_go_in_ascending_key_order() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let m = filled(&log);
        let seen: Vec<u32> = m.values().map(|t| t.0).collect();
        assert_eq!(seen, ascending());
    }

    #[test]
    fn drain_yields_ascending_key_order_and_empties() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut m = filled(&log);
        let drained = m.drain();
        assert!(m.is_empty());
        assert!(
            log.borrow().is_empty(),
            "drain hands the values back undropped"
        );
        assert_eq!(drained.iter().map(|t| t.0).collect::<Vec<_>>(), ascending());
        drop(drained);
        assert_eq!(*log.borrow(), ascending());
        // The map is usable again.
        m.insert(5, Tok(5, log.clone()));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn drop_releases_in_ascending_key_order() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut m = filled(&log);
        drop(m.remove(40));
        assert_eq!(*log.borrow(), [40]);
        log.borrow_mut().clear();
        drop(m);
        let want: Vec<u32> = ascending().into_iter().filter(|&k| k != 40).collect();
        assert_eq!(*log.borrow(), want);
    }

    /// `std::mem::take`, as the device's drop empties its table, leaves an empty map and drops
    /// the taken one in ascending key order.
    #[test]
    fn a_taken_map_drops_in_ascending_key_order() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut m = filled(&log);
        drop(std::mem::take(&mut m));
        assert!(m.is_empty());
        assert_eq!(*log.borrow(), ascending());
    }
}

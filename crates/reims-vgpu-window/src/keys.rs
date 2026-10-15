//! The keys held down and the modifiers: what reaches the guest as downs and ups.
//!
//! The guest's USB keyboard repeats a held key itself, at the guest's own rate, so a key is one
//! down and one up: the host's repeats are dropped. [`KeyState`] tracks every key the guest was
//! told is down, so that each gets its up ([`KeyState::release_all`]) when the window loses the
//! keyboard, and none sticks.
//!
//! Modifiers arrive as `flagsChanged:` events, which carry the key that changed but not whether
//! it went down or up. The event's `modifierFlags` tells: its low, device-dependent bits
//! (IOKit's `NX_DEVICE*KEYMASK`) say which of left and right Shift, Control, Option and Command
//! are down. Every event's flags are the whole state, so a change missed while the window did
//! not have the keyboard is made good at the next event. Where an event carries no
//! device-dependent bits (a synthesized one), the side is taken from the key the event names.

/// A key went down or up: a USB HID keyboard usage (page 0x07).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    /// The key.
    pub usage: u16,
    /// Down (`true`) or up.
    pub down: bool,
}

impl KeyEvent {
    /// `usage` went down.
    pub fn down(usage: u16) -> KeyEvent {
        KeyEvent { usage, down: true }
    }

    /// `usage` went up.
    pub fn up(usage: u16) -> KeyEvent {
        KeyEvent { usage, down: false }
    }
}

/// Which modifiers are down, either side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    /// ⌃ Control.
    pub control: bool,
    /// ⌥ Option.
    pub option: bool,
    /// ⌘ Command.
    pub command: bool,
    /// ⇧ Shift.
    pub shift: bool,
}

/// `NSEventModifierFlagShift` (IOKit's `NX_SHIFTMASK`).
pub const FLAG_SHIFT: u64 = 1 << 17;
/// `NSEventModifierFlagControl` (`NX_CONTROLMASK`).
pub const FLAG_CONTROL: u64 = 1 << 18;
/// `NSEventModifierFlagOption` (`NX_ALTERNATEMASK`).
pub const FLAG_OPTION: u64 = 1 << 19;
/// `NSEventModifierFlagCommand` (`NX_COMMANDMASK`).
pub const FLAG_COMMAND: u64 = 1 << 20;

impl Modifiers {
    /// The modifiers `modifier_flags` (`NSEvent.modifierFlags`) says are down.
    pub fn from_flags(modifier_flags: u64) -> Modifiers {
        Modifiers {
            control: modifier_flags & FLAG_CONTROL != 0,
            option: modifier_flags & FLAG_OPTION != 0,
            command: modifier_flags & FLAG_COMMAND != 0,
            shift: modifier_flags & FLAG_SHIFT != 0,
        }
    }
}

/// `kVK_CapsLock`.
const KVK_CAPS_LOCK: u16 = 0x39;
/// Keyboard Caps Lock.
const USAGE_CAPS_LOCK: u16 = 0x39;

/// A modifier key.
struct Modifier {
    /// Its usage.
    usage: u16,
    /// Its device-dependent bit in `modifierFlags` (`NX_DEVICE*KEYMASK`).
    device: u64,
    /// Its kind's device-independent bit (`FLAG_*`).
    flag: u64,
    /// Its macOS key code.
    keycode: u16,
}

/// The modifiers, left then right of each kind, in the order their changes are sent.
const MODIFIERS: [(Modifier, Modifier); 4] = [
    (
        Modifier {
            usage: 0xE0,
            device: 0x0000_0001,
            flag: FLAG_CONTROL,
            keycode: 0x3B,
        },
        Modifier {
            usage: 0xE4,
            device: 0x0000_2000,
            flag: FLAG_CONTROL,
            keycode: 0x3E,
        },
    ),
    (
        Modifier {
            usage: 0xE1,
            device: 0x0000_0002,
            flag: FLAG_SHIFT,
            keycode: 0x38,
        },
        Modifier {
            usage: 0xE5,
            device: 0x0000_0004,
            flag: FLAG_SHIFT,
            keycode: 0x3C,
        },
    ),
    (
        Modifier {
            usage: 0xE2,
            device: 0x0000_0020,
            flag: FLAG_OPTION,
            keycode: 0x3A,
        },
        Modifier {
            usage: 0xE6,
            device: 0x0000_0040,
            flag: FLAG_OPTION,
            keycode: 0x3D,
        },
    ),
    (
        Modifier {
            usage: 0xE3,
            device: 0x0000_0008,
            flag: FLAG_COMMAND,
            keycode: 0x37,
        },
        Modifier {
            usage: 0xE7,
            device: 0x0000_0010,
            flag: FLAG_COMMAND,
            keycode: 0x36,
        },
    ),
];

/// The keys the guest was told are down.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyState {
    /// Each key down, in the order it went down.
    held: Vec<u16>,
}

impl KeyState {
    /// No key down.
    pub fn new() -> KeyState {
        KeyState::default()
    }

    /// Whether the guest was told `usage` is down.
    pub fn is_held(&self, usage: u16) -> bool {
        self.held.contains(&usage)
    }

    /// Which modifiers the guest was told are down.
    pub fn held_modifiers(&self) -> Modifiers {
        let either =
            |kind: &(Modifier, Modifier)| self.is_held(kind.0.usage) || self.is_held(kind.1.usage);
        let [control, shift, option, command] = &MODIFIERS;
        Modifiers {
            control: either(control),
            option: either(option),
            command: either(command),
            shift: either(shift),
        }
    }

    /// A key went down (`keyDown:`): its down, or `None` for the host's repeat of a held key
    /// (`is_repeat`, `NSEvent.isARepeat`) and for a key already down.
    pub fn key_down(&mut self, usage: u16, is_repeat: bool) -> Option<KeyEvent> {
        if is_repeat || self.is_held(usage) {
            return None;
        }
        self.held.push(usage);
        Some(KeyEvent::down(usage))
    }

    /// A key went up (`keyUp:`): its up, or `None` for a key the guest was not told is down.
    pub fn key_up(&mut self, usage: u16) -> Option<KeyEvent> {
        let at = self.held.iter().position(|&held| held == usage)?;
        self.held.remove(at);
        Some(KeyEvent::up(usage))
    }

    /// A modifier changed (`flagsChanged:`): `mac_keycode` is the key that changed
    /// (`NSEvent.keyCode`), `modifier_flags` the whole state (`NSEvent.modifierFlags`, with its
    /// device-dependent bits). Ups for the modifiers no longer down, then downs for those newly
    /// down, whichever key the event names: a change missed earlier is made good here.
    ///
    /// Caps Lock gives a down then an up on every change, its light on or off: the guest's
    /// keyboard toggles Caps Lock on each press, as the host's did.
    pub fn flags_changed(&mut self, mac_keycode: u16, modifier_flags: u64) -> Vec<KeyEvent> {
        let mut events = self.sync(Some(mac_keycode), modifier_flags);
        if mac_keycode == KVK_CAPS_LOCK {
            events.push(KeyEvent::down(USAGE_CAPS_LOCK));
            events.push(KeyEvent::up(USAGE_CAPS_LOCK));
        }
        events
    }

    /// Brings the modifiers held into line with any event's `modifierFlags` (a key or a click
    /// that arrives with a modifier the window did not see go down or up): as
    /// [`KeyState::flags_changed`], for an event that names no modifier key.
    pub fn sync_modifiers(&mut self, modifier_flags: u64) -> Vec<KeyEvent> {
        self.sync(None, modifier_flags)
    }

    /// Ups for every key down, the last down first. The guest then has no key down.
    pub fn release_all(&mut self) -> Vec<KeyEvent> {
        let events = self.held.iter().rev().map(|&usage| KeyEvent::up(usage));
        let events = events.collect();
        self.held.clear();
        events
    }

    /// The modifier changes `modifier_flags` makes, `keycode` the key the event names.
    fn sync(&mut self, keycode: Option<u16>, modifier_flags: u64) -> Vec<KeyEvent> {
        let mut wanted = Vec::with_capacity(8);
        for (left, right) in &MODIFIERS {
            let (l, r) = self.wanted(left, right, keycode, modifier_flags);
            wanted.push((left.usage, l));
            wanted.push((right.usage, r));
        }
        let mut events = Vec::new();
        for pass_down in [false, true] {
            for &(usage, down) in &wanted {
                if down == pass_down && down != self.is_held(usage) {
                    let event = if down {
                        self.key_down(usage, false)
                    } else {
                        self.key_up(usage)
                    };
                    events.extend(event);
                }
            }
        }
        events
    }

    /// Whether `left` and `right`, one kind of modifier, should be down.
    fn wanted(
        &self,
        left: &Modifier,
        right: &Modifier,
        keycode: Option<u16>,
        modifier_flags: u64,
    ) -> (bool, bool) {
        if modifier_flags & left.flag == 0 {
            return (false, false);
        }
        let (l, r) = (
            modifier_flags & left.device != 0,
            modifier_flags & right.device != 0,
        );
        if l || r {
            return (l, r);
        }
        // The kind is down but the event says not which side: the key it names changed, and
        // one side at least is down.
        let (mut l, mut r) = (self.is_held(left.usage), self.is_held(right.usage));
        if keycode == Some(left.keycode) {
            l = !l;
        } else if keycode == Some(right.keycode) {
            r = !r;
        }
        if !l && !r {
            if keycode == Some(right.keycode) {
                r = true;
            } else {
                l = true;
            }
        }
        (l, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L_SHIFT: u64 = FLAG_SHIFT | 0x02;
    const R_SHIFT: u64 = FLAG_SHIFT | 0x04;
    const L_CONTROL: u64 = FLAG_CONTROL | 0x01;
    const R_CONTROL: u64 = FLAG_CONTROL | 0x2000;
    const L_OPTION: u64 = FLAG_OPTION | 0x20;
    const R_OPTION: u64 = FLAG_OPTION | 0x40;
    const L_COMMAND: u64 = FLAG_COMMAND | 0x08;
    const R_COMMAND: u64 = FLAG_COMMAND | 0x10;
    /// A bit `modifierFlags` carries besides the modifiers (`NX_NONCOALSESCEDMASK`).
    const BASE: u64 = 0x100;

    fn down(usage: u16) -> KeyEvent {
        KeyEvent::down(usage)
    }

    fn up(usage: u16) -> KeyEvent {
        KeyEvent::up(usage)
    }

    /// Left and right Shift, Control, Option and Command are told apart
    #[test]
    fn left_and_right_modifiers_are_told_apart() {
        let cases = [
            (0x38, L_SHIFT, 0xE1),
            (0x3C, R_SHIFT, 0xE5),
            (0x3B, L_CONTROL, 0xE0),
            (0x3E, R_CONTROL, 0xE4),
            (0x3A, L_OPTION, 0xE2),
            (0x3D, R_OPTION, 0xE6),
            (0x37, L_COMMAND, 0xE3),
            (0x36, R_COMMAND, 0xE7),
        ];
        for (keycode, flags, usage) in cases {
            let mut keys = KeyState::new();
            assert_eq!(keys.flags_changed(keycode, BASE | flags), vec![down(usage)]);
            assert_eq!(keys.flags_changed(keycode, BASE), vec![up(usage)]);
        }
        // Left Shift, then right Shift with it, then left let go, then right.
        let mut keys = KeyState::new();
        assert_eq!(keys.flags_changed(0x38, L_SHIFT), vec![down(0xE1)]);
        assert_eq!(
            keys.flags_changed(0x3C, L_SHIFT | R_SHIFT),
            vec![down(0xE5)]
        );
        assert_eq!(keys.flags_changed(0x38, R_SHIFT), vec![up(0xE1)]);
        assert!(keys.held_modifiers().shift);
        assert_eq!(keys.flags_changed(0x3C, 0), vec![up(0xE5)]);
        assert_eq!(keys.held_modifiers(), Modifiers::default());
        // Right Command and left Option together.
        assert_eq!(keys.flags_changed(0x36, R_COMMAND), vec![down(0xE7)]);
        assert_eq!(
            keys.flags_changed(0x3A, R_COMMAND | L_OPTION),
            vec![down(0xE2)]
        );
        assert_eq!(
            keys.held_modifiers(),
            Modifiers {
                option: true,
                command: true,
                ..Modifiers::default()
            }
        );
    }

    /// With no device-dependent bits, the side is the key the event names
    #[test]
    fn a_synthesized_modifier_takes_its_side_from_the_key() {
        let mut keys = KeyState::new();
        assert_eq!(keys.flags_changed(0x3C, FLAG_SHIFT), vec![down(0xE5)]);
        assert_eq!(keys.flags_changed(0x38, FLAG_SHIFT), vec![down(0xE1)]);
        assert_eq!(keys.flags_changed(0x3C, FLAG_SHIFT), vec![up(0xE5)]);
        assert_eq!(keys.flags_changed(0x38, 0), vec![up(0xE1)]);
        // A flag with neither side seen: the left one.
        assert_eq!(keys.sync_modifiers(FLAG_COMMAND), vec![down(0xE3)]);
    }

    /// The host's repeats are dropped, and a key is down once
    #[test]
    fn host_repeats_are_dropped() {
        let mut keys = KeyState::new();
        assert_eq!(keys.key_down(0x04, false), Some(down(0x04)));
        assert_eq!(keys.key_down(0x04, true), None);
        assert_eq!(keys.key_down(0x04, true), None);
        assert_eq!(keys.key_down(0x04, false), None);
        assert_eq!(keys.key_up(0x04), Some(up(0x04)));
        assert_eq!(keys.key_up(0x04), None);
        // A repeat of a key not seen going down is dropped too.
        assert_eq!(keys.key_down(0x05, true), None);
        assert_eq!(keys.key_up(0x05), None);
    }

    /// Every key and modifier down gets its up, the last down first, and is then up
    #[test]
    fn release_all_lifts_every_held_key() {
        let mut keys = KeyState::new();
        keys.flags_changed(0x3B, L_CONTROL);
        keys.flags_changed(0x3D, L_CONTROL | R_OPTION);
        keys.key_down(0x04, false);
        keys.key_down(0x16, false);
        keys.key_up(0x04);
        assert_eq!(keys.release_all(), vec![up(0x16), up(0xE6), up(0xE0)]);
        assert_eq!(keys.release_all(), vec![]);
        assert_eq!(keys.held_modifiers(), Modifiers::default());
        assert!(!keys.is_held(0x16));
        // Released keys' later ups are dropped, and their flags' ups too.
        assert_eq!(keys.key_up(0x16), None);
        assert_eq!(keys.flags_changed(0x3B, 0), vec![]);
    }

    /// A flagsChanged missed while the window lacked the keyboard is made good at the next
    #[test]
    fn a_missed_flags_changed_is_made_good() {
        let mut keys = KeyState::new();
        keys.flags_changed(0x38, L_SHIFT);
        // Shift went up and right Control down unseen; then left Option goes down.
        assert_eq!(
            keys.flags_changed(0x3A, R_CONTROL | L_OPTION),
            vec![up(0xE1), down(0xE4), down(0xE2)]
        );
        // A key arrives with Command down, never seen going down.
        assert_eq!(
            keys.sync_modifiers(R_CONTROL | L_OPTION | L_COMMAND),
            vec![down(0xE3)]
        );
        // A click arrives with nothing down.
        assert_eq!(
            keys.sync_modifiers(BASE),
            vec![up(0xE4), up(0xE2), up(0xE3)]
        );
        assert_eq!(keys.release_all(), vec![]);
    }

    /// Caps Lock gives a down and an up at each change, and is never held
    #[test]
    fn caps_lock_toggles_on_each_change() {
        let mut keys = KeyState::new();
        let on = 1 << 16;
        assert_eq!(keys.flags_changed(0x39, on), vec![down(0x39), up(0x39)]);
        assert_eq!(keys.flags_changed(0x39, 0), vec![down(0x39), up(0x39)]);
        assert!(!keys.is_held(0x39));
        // With Shift down unseen, Shift is made good first.
        assert_eq!(
            keys.flags_changed(0x39, on | L_SHIFT),
            vec![down(0xE1), down(0x39), up(0x39)]
        );
        assert_eq!(keys.release_all(), vec![up(0xE1)]);
    }

    /// `Modifiers::from_flags` reads the device-independent bits
    #[test]
    fn modifiers_from_flags() {
        assert_eq!(
            Modifiers::from_flags(FLAG_CONTROL | FLAG_OPTION | FLAG_COMMAND | 0x2009),
            Modifiers {
                control: true,
                option: true,
                command: true,
                shift: false
            }
        );
        assert_eq!(Modifiers::from_flags(0x3F), Modifiers::default());
    }
}

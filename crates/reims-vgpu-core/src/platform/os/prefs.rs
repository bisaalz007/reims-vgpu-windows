//! The `Preferences` interface the public API's display constructors take
//! (`reims_vgpu::DisplayConfig::apply_preferences`, `DisplayMode::with_preferences`).
//!
//! No platform reads a preferences domain: the defaults are what an empty domain gives.

/// A reader of display settings a Rust host may hand to the public API's `with_preferences`
/// constructors. Nothing in this workspace implements it over a system domain.
pub trait Preferences: Send + Sync {
    /// A boolean setting: `(value, exists)`; `exists` is false for an absent key or one that does
    /// not convert to a boolean.
    fn app_bool(&self, key: &str) -> (bool, bool);
    /// A number setting, as f32; `None` for an absent key or one that is not a number.
    fn any_user_f32(&self, key: &str) -> Option<f32>;
}

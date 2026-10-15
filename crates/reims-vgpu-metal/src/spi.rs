//! What objc2-metal does not declare: the private classes Metal hands out and the methods the
//! platform sends them (`classes`), typed declarations of the private selectors of Metal's
//! public protocols and classes (`methods`), and Metal's C functions (`functions`). The one
//! place that writes `msg_send!`.

pub mod classes;
pub mod functions;
pub mod methods;

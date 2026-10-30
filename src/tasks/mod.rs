//! Tasks implementing repeating async events.
//!
//! # Tasks
//! - ``bad_input_task``: updates the `bad_input_likelihood` metric on every new frame
//! - ``solar_event_task``: temporarily disables kotekan RFI flagging around solar transit

mod bad_inputs;
mod solar_zeroing;

pub use bad_inputs::bad_input_task;
pub use solar_zeroing::solar_event_task;

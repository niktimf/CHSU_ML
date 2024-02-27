
mod data_processing;
mod model;

pub mod inference;
pub mod training;

pub use data_processing::{AgNewsDataset, TextClassificationDataset};
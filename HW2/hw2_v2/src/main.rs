mod data_processing;

use burn::backend::Wgpu;
use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};
use burn::tensor::Tensor;

use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn::tensor::backend::AutodiffBackend;
use hw2_v2::training;
use hw2_v2::training::TransformerConfig;
use hw2_v2::AgNewsDataset;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
    let config = TransformerConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new(),
    );

    training::train::<B, AgNewsDataset>(
        devices,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::{launch, ElemType};
    use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
    use burn::backend::Autodiff;

    pub fn run() {
        launch::<Autodiff<Wgpu<AutoGraphicsApi, ElemType, i32>>>(vec![WgpuDevice::default()]);
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run();
}

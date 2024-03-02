// Данный модуль обучает модель классификации текста с использованием предоставленных наборов данных
// для обучения и тестирования, а также предоставленной конфигурации. Сначала инициализируется токенизатор
// и батчеры для наборов данных, затем инициализируется модель и загрузчики данных для наборов данных.
// Затем функция инициализирует оптимизатор и планировщик скорости обучения, и использует их вместе с моделью
// и наборами данных для построения обучающего пайплайна, который используется для обучения модели. Обученная модель
// и конфигурация затем сохраняются в указанный каталог.

use crate::{
    data_processing::{
        BertCasedTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer,
    },
    model::{TextClassificationLstmModelConfig, TextClassificationTransformerModelConfig},
};
use std::fmt::{Debug, Display};

use crate::model::{LstmModelInitiation, TransformerModelInitiation};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, Dataset};
use burn::tensor::backend::Backend;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module,
    nn::lstm::LstmConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

#[derive(Config)]
pub struct TransformerConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 256)]
    pub max_seq_length: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub num_epochs: usize,
}
// Define train function
pub fn train<B, D>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_test: D,         // Testing dataset
    config: TransformerConfig, // Experiment configuration
    artifact_dir: &str,      // Directory to save model and config files
) where
    B: AutodiffBackend,
    D: TextClassificationDataset + 'static,
    // M: TransformerModelInitiation<B> + LstmModelInitiation<B> + 'static + Display
    //M: ModelInit<B> + 'static + Display
{
    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Initialize batchers for training and testing data
    let batcher_train = TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        devices[0].clone(),
        config.max_seq_length,
    );

    let batcher_test = TextClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        devices[0].clone(),
        config.max_seq_length,
    );

    // Initialize model
    let model = TextClassificationTransformerModelConfig::new(
        config.transformer.clone(),
        D::num_classes(),
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init(&devices[0]);

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_train, 50_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_test, 5_000));

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = NoamLrSchedulerConfig::new(1e-2)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .num_epochs(config.num_epochs)
        .build(model, optim, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}

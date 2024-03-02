// Это базовая модель классификации текста, реализованная на Rust с использованием фреймворка Burn.

use crate::data_processing::{TextClassificationInferenceBatch, TextClassificationTrainingBatch};
use burn::nn::{Lstm, LstmConfig};
use burn::tensor::{Bool, Device, Int, TensorKind};
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{AutodiffBackend, Backend},
    tensor::{activation::softmax, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use burn::data::dataloader::batcher::Batcher;
use derive_new::new;

#[derive(Module, Debug)]
pub struct TextClassificationTransformerModel<B: Backend> {
    model: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}

pub trait Embeddings<B: Backend> {
    fn embedding_token(&self) -> &Embedding<B>;
    fn embedding_pos(&self) -> &Embedding<B>;

    fn prepare_embeddings(
        &self,
        tokens: Tensor<B, 2, Int>,
        seq_length: usize,
        batch_size: usize,
        device: &Device<B>,
    ) -> Tensor<B, 3> {
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos().forward(index_positions);
        let embedding_tokens = self.embedding_token().forward(tokens);
        (embedding_positions + embedding_tokens) / 2
    }
}

pub trait TransformTensor<B: Backend> {
    fn n_classes(&self) -> usize;
    fn transform_output_tensor(&self, output: Tensor<B, 3>, batch_size: usize) -> Tensor<B, 2> {
        output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes()])
    }
}

pub trait TransformerEncode<B: Backend> {
    fn transformer_encode(
        &self,
        embedding: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3>;
}

impl<B: Backend> TransformerEncode<B> for TextClassificationTransformerModel<B> {
    fn transformer_encode(
        &self,
        embedding: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        self.model
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad))
    }
}

impl<B: Backend> Embeddings<B> for TextClassificationTransformerModel<B> {
    fn embedding_token(&self) -> &Embedding<B> {
        &self.embedding_token
    }

    fn embedding_pos(&self) -> &Embedding<B> {
        &self.embedding_pos
    }
}

impl<B: Backend> TransformTensor<B> for TextClassificationTransformerModel<B> {
    fn n_classes(&self) -> usize {
        self.n_classes
    }
}

pub trait LstmEncode<B: Backend> {
    fn lstm_encode(&self, embedding: Tensor<B, 3>) -> Tensor<B, 3>;
}

impl<B: Backend> LstmEncode<B> for TextClassificationLstmModel<B> {
    fn lstm_encode(&self, embedding: Tensor<B, 3>) -> Tensor<B, 3> {
        let (output, _) = self.model.forward(embedding, None);
        output
    }
}

impl<B: Backend> Embeddings<B> for TextClassificationLstmModel<B> {
    fn embedding_token(&self) -> &Embedding<B> {
        &self.embedding_token
    }

    fn embedding_pos(&self) -> &Embedding<B> {
        &self.embedding_pos
    }
}

impl<B: Backend> TransformTensor<B> for TextClassificationLstmModel<B> {
    fn n_classes(&self) -> usize {
        self.n_classes
    }
}

pub trait ModelOperations<B: Backend> {
    fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B>;
    fn inference(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2>;
}

impl<B: Backend> ModelOperations<B> for TextClassificationTransformerModel<B> {
    fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let embedding = self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded = self.transformer_encode(embedding, mask_pad);
        let output = self.output.forward(encoded);

        let output_classification = self.transform_output_tensor(output, batch_size);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        // Возвращаем loss и выход
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    fn inference(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2> {
        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let embedding: Tensor<B, 3> =
            self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded: Tensor<B, 3> = self.transformer_encode(embedding, mask_pad);
        let output: Tensor<B, 3> = self.output.forward(encoded);

        let output: Tensor<B, 2> = self.transform_output_tensor(output, batch_size);

        softmax(output, 1)
    }
}

#[derive(Module, Debug)]
pub struct TextClassificationLstmModel<B: Backend> {
    model: Lstm<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}

impl<B: Backend> ModelOperations<B> for TextClassificationLstmModel<B> {
    fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let embedding = self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded = self.lstm_encode(embedding);
        let output = self.output.forward(encoded);

        let output_classification = self.transform_output_tensor(output, batch_size);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        // Возвращаем loss и выход
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    fn inference(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2> {
        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let embedding: Tensor<B, 3> =
            self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded: Tensor<B, 3> = self.lstm_encode(embedding);
        let output: Tensor<B, 3> = self.output.forward(encoded);

        let output: Tensor<B, 2> = self.transform_output_tensor(output, batch_size);

        softmax(output, 1)
    }
}

// Конфигурация модели
#[derive(Config)]
pub struct TextClassificationTransformerModelConfig {
    model_config: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}
pub trait TransformerModelInitiation<B: Backend> {
    fn init(&self, device: &B::Device) -> TextClassificationTransformerModel<B>;
    fn init_with(
        &self,
        record: TextClassificationTransformerModelRecord<B>,
    ) -> TextClassificationTransformerModel<B>;
}

impl<B: Backend> TransformerModelInitiation<B> for TextClassificationTransformerModelConfig {
    fn init(&self, device: &B::Device) -> TextClassificationTransformerModel<B> {
        let output = LinearConfig::new(self.model_config.d_model, self.n_classes).init(device);
        let transformer_config = self.model_config.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.model_config.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.model_config.d_model).init(device);

        TextClassificationTransformerModel {
            model: transformer_config,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }

    fn init_with(
        &self,
        record: TextClassificationTransformerModelRecord<B>,
    ) -> TextClassificationTransformerModel<B> {
        let output =
            LinearConfig::new(self.model_config.d_model, self.n_classes).init_with(record.output);
        let transformer_config = self.model_config.init_with(record.model);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.model_config.d_model)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.model_config.d_model)
            .init_with(record.embedding_pos);

        TextClassificationTransformerModel {
            model: transformer_config,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}

#[derive(Config)]
pub struct TextClassificationLstmModelConfig {
    model_config: LstmConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

pub trait LstmModelInitiation<B: Backend> {
    fn init(&self, device: &B::Device) -> TextClassificationLstmModel<B>;
    fn init_with(
        &self,
        record: TextClassificationLstmModelRecord<B>,
    ) -> TextClassificationLstmModel<B>;
}

impl<B: Backend> LstmModelInitiation<B> for TextClassificationLstmModelConfig {
    fn init(&self, device: &B::Device) -> TextClassificationLstmModel<B> {
        let output = LinearConfig::new(self.model_config.d_hidden, self.n_classes).init(device);
        let model = self.model_config.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.model_config.d_input).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.model_config.d_input).init(device);

        TextClassificationLstmModel {
            model,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }

    fn init_with(
        &self,
        record: TextClassificationLstmModelRecord<B>,
    ) -> TextClassificationLstmModel<B> {
        let output =
            LinearConfig::new(self.model_config.d_hidden, self.n_classes).init_with(record.output);
        let model = self.model_config.init_with(record.model);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.model_config.d_input)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.model_config.d_input)
            .init_with(record.embedding_pos);

        TextClassificationLstmModel {
            model,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}


/// Определение шага обучения
impl<B: AutodiffBackend> TrainStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationTransformerModel<B> {
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Выполняем прямой проход, вычисляем градиенты и возвращаем их вместе с выходом
        let item = self.forward(item);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

/// Определение шага валидации
impl<B: Backend> ValidStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationTransformerModel<B> {
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Выполняем прямой проход и возвращаем выход
        self.forward(item)
    }
}

impl<B: AutodiffBackend> TrainStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationLstmModel<B> {
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Выполняем прямой проход, вычисляем градиенты и возвращаем их вместе с выходом
        let item = self.forward(item);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

/// Определение шага валидации
impl<B: Backend> ValidStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationLstmModel<B> {
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Выполняем прямой проход и возвращаем выход
        self.forward(item)
    }
}


/*
// Структура модели
#[derive(Module, Debug)]
pub struct TextClassificationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}


/// Поведение модели
impl<B: Backend> TextClassificationModel<B> {
    // Forward pass для обучения
    pub fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Производим кодирование трансформатором, вычисляем выход и loss
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        // Возвращаем loss и выход
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    /// Forward pass для инференса
    pub fn infer(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2> {

        // Получаем размер пакета и длину последовательности, а также устройство
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Перемещаем тензоры на устройство
        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Вычисляем токен и позиционные эмбеддинги, и объединяем их
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Производим кодирование трансформатором, вычисляем выход и применяем softmax для предсказания
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }
}

// Конфигурация модели
#[derive(Config)]
pub struct TextClassificationModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

// Функции инициализации модели
impl TextClassificationModelConfig {
    /// Инициализация модели с пустыми весами
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextClassificationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }

    /// Инициализация модели с загруженными весами
    pub fn init_with<B: Backend>(
        &self,
        record: TextClassificationModelRecord<B>,
    ) -> TextClassificationModel<B> {
        let output =
            LinearConfig::new(self.transformer.d_model, self.n_classes).init_with(record.output);
        let transformer = self.transformer.init_with(record.transformer);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.transformer.d_model)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model)
            .init_with(record.embedding_pos);

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}

/// Определение шага обучения
impl<B: AutodiffBackend> TrainStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationModel<B> {
    fn step(
        &self,
        item: TextClassificationTrainingBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
        // Выполняем прямой проход, вычисляем градиенты и возвращаем их вместе с выходом
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

/// Определение шага валидации
impl<B: Backend> ValidStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
for TextClassificationModel<B> {
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Выполняем прямой проход и возвращаем выход
        self.forward(item)
    }
}

 */

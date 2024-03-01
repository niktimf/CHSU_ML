
// Это базовая модель классификации текста, реализованная на Rust с использованием фреймворка Burn.

use crate::data_processing::{TextClassificationInferenceBatch, TextClassificationTrainingBatch};
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
use burn::nn::Lstm;
use burn::tensor::{Bool, Device, Int, TensorKind};
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

pub trait ModelOperations<B: Backend> {
    fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B>;
    fn inference(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2>;
    fn prepare_embeddings<const D: usize>(&self, tokens: Tensor<B, 2, Int>, seq_length: usize, batch_size: usize, device: &Device<B>) -> Tensor<B, 3>;
    fn encode(&self, embedding: Tensor<B, 3>, mask_pad: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3>;
    fn transform_output_tensor(&self, output: Tensor<B, 3>, batch_size: usize) -> Tensor<B, 2>;
}

impl <B: Backend> ModelOperations<B> for TextClassificationTransformerModel<B> {
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
        let encoded = self.encode(embedding, Some(mask_pad));
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
        let embedding: Tensor<B, 3> = self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded: Tensor<B, 3> = self.encode(embedding, Some(mask_pad));
        let output: Tensor<B, 3> = self.output.forward(encoded);
        let output: Tensor<B, 2> = self.transform_output_tensor(output, batch_size);

        softmax(output, 1)
    }

    fn prepare_embeddings<const D: usize>(&self, tokens: Tensor<B, 2, Int>, seq_length: usize, batch_size: usize, device: &Device<B>) -> Tensor<B, 3> {
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        (embedding_positions + embedding_tokens) / 2
    }

    // Кодирование с использованием трансформатора
    fn encode(&self, embedding: Tensor<B, 3>, mask_pad: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let Some(mask_pad) = mask_pad;
        self.model.forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad))
    }


    fn transform_output_tensor(&self, output: Tensor<B, 3>, batch_size: usize) -> Tensor<B, 2> {
        output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes])
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

impl <B: Backend> ModelOperations<B> for TextClassificationLstmModel<B> {
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
        let encoded = self.encode(embedding, None);
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
        let embedding: Tensor<B, 3> = self.prepare_embeddings(tokens, seq_length, batch_size, device);
        let encoded: Tensor<B, 3> = self.encode(embedding, None);
        let output: Tensor<B, 3> = self.output.forward(encoded);

        let output: Tensor<B, 2> = self.transform_output_tensor(output, batch_size);

        softmax(output, 1)
    }

    fn prepare_embeddings<const D: usize>(&self, tokens: Tensor<B, 2, Int>, seq_length: usize, batch_size: usize, device: &Device<B>) -> Tensor<B, 3> {
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        (embedding_positions + embedding_tokens) / 2
    }

    // Кодирование с использованием трансформатора
    fn encode(&self, embedding: Tensor<B, 3>, _mask_pad: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let (output, _) = self.model.forward(embedding, None);
        output
    }


    fn transform_output_tensor(&self, output: Tensor<B, 3>, batch_size: usize) -> Tensor<B, 2> {
        output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes])
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








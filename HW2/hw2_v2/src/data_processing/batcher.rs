
// Модуль определяет две структуры TextClassificationTrainingBatch и TextClassificationInferenceBatch для
// обработки пакетов данных во время обучения и вывода соответственно. Структура TextClassificationBatcher
// реализована для создания этих пакетов. Он параметризован типом B: Backend для поддержки различных
// вычислительных бэкэндов (например, CPU, CUDA).

// Два варианта реализации трейта Batcher предоставляются для TextClassificationBatcher, один для создания
// обучающих пакетов и один для создания пакетов вывода. В каждой реализации функция batch определена для
// преобразования вектора элементов в пакет. Для обучения элементы являются экземплярами TextClassificationItem
// и включают как текст, так и соответствующую метку. Для вывода элементы просто являются строками без меток.
// Функция токенизирует текст, генерирует маску заполнения и возвращает объект пакета.

use super::{dataset::TextClassificationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Data, ElementConversion, Int, Tensor},
};
use std::sync::Arc;
use derive_new::new;

/// Структура для упаковки элементов классификации текста
#[derive(new)]
pub struct TextClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>, // Токенизатор для конвертации текста в токены
    device: B::Device, // Вычислительное устройство для создания тензоров
    max_seq_length: usize, // Максимальная длина для токенизированного текста
}

/// Структура для обучающего пакета в задаче классификации текста
#[derive(Debug, Clone, new)]
pub struct TextClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Токенизированный текст
    pub labels: Tensor<B, 1, Int>,    // Метки классов
    pub mask_pad: Tensor<B, 2, Bool>, // Маска заполнения для токенизированного текста
}

/// Структура для инференса в задаче классификации текста
#[derive(Debug, Clone, new)]
pub struct TextClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

/// Реализация Batcher для TextClassificationBatcher для обучения
impl<B: Backend> Batcher<TextClassificationItem, TextClassificationTrainingBatch<B>>
for TextClassificationBatcher<B>
{
    /// Создает пакет обучения из вектора элементов
    fn batch(&self, items: Vec<TextClassificationItem>) -> TextClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        // Токенизируем каждый элемент
        items.iter().for_each(|item| {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                Data::from([(item.label as i64).elem()]),
                &self.device,
            ));
        });

        // Генерируем маску заполнения для токенизированного текста
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &self.device,
        );

        // Создаем и возвращаем пакет обучения
        TextClassificationTrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}

/// Реализация Batcher для TextClassificationBatcher для инференса
impl<B: Backend> Batcher<String, TextClassificationInferenceBatch<B>>
for TextClassificationBatcher<B>
{
    /// Создает пакет вывода из вектора элементов
    fn batch(&self, items: Vec<String>) -> TextClassificationInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Токенизируем каждый элемент
        items.iter().for_each(|item| {
            tokens_list.push(self.tokenizer.encode(&item));
        });

        // Генерируем маску заполнения для токенизированного текста
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        // Создаем и возвращаем пакет вывода
        TextClassificationInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}
// The AgNewsDataset and DbPediaDataset structs are examples of specific text
// classification datasets.  Each dataset struct has a field for the underlying
// SQLite dataset and implements methods for accessing and processing the data.
// Each dataset is also provided with specific information about its classes via
// the TextClassificationDataset trait. These implementations are designed to be used
// with a machine learning framework for tasks such as training a text classification model.

use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

// Структура для элементов классификации текста
#[derive(new, Clone, Debug)]
pub struct TextClassificationItem {
    pub text: String, // Текст для классификации
    pub label: usize, // Метка текста (категория классификации)
}

// Трейт для наборов данных классификации текста
pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn num_classes() -> usize; // Возвращает количество уникальных классов в наборе данных
    fn class_name(label: usize) -> String; // Возвращает имя класса по его метке
}

// Структура для элементов набора данных AG News
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgNewsItem {
    pub text: String, // Текст для классификации
    pub label: usize, // Метка текста (категория классификации)
}

// Структура для набора данных AG News
#[derive(Debug)]
pub struct AgNewsDataset {
    dataset: SqliteDataset<AgNewsItem>, // Набор данных SQLite
}

// Реализация методов для конструирования набора данных AG News
impl Dataset<TextClassificationItem> for AgNewsDataset {
    /// Возвращает элемент набора данных по индексу
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| TextClassificationItem::new(item.text, item.label))
    }

    /// Возвращает количество элементов в наборе данных
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub enum DatasetSplit {
    Train,
    Test,
}

// Реализация методов для конструирования набора данных AG News
impl AgNewsDataset {
    /// Возвращает обучающую часть набора данных
    pub fn train() -> Self {
        Self::new(DatasetSplit::Train)
    }

    /// Возвращает тестовую часть набора данных
    pub fn test() -> Self {
        Self::new(DatasetSplit::Test)
    }

    /// Конструирует набор данных из раздела (либо "train", либо "test")
    pub fn new(split: DatasetSplit) -> Self {
        let split_str = match split {
            DatasetSplit::Train => "train",
            DatasetSplit::Test => "test",
        };
        let dataset: SqliteDataset<AgNewsItem> = HuggingfaceDatasetLoader::new("ag_news")
            .dataset(split_str)
            .unwrap();
        Self { dataset }
    }
}

/// Перечисление для классов набора данных AG News
#[derive(EnumCount, EnumIter)]
pub enum AgNewsDatasetClasses {
    World,
    Sports,
    Business,
    Technology,
}

/// Реализация методов для классов набора данных AG News
impl TextClassificationDataset for AgNewsDataset {
    /// Возвращает количество уникальных классов в наборе данных
    fn num_classes() -> usize {
        AgNewsDatasetClasses::COUNT
    }

    /// Возвращает имя класса по его метке
    fn class_name(label: usize) -> String {
        match label {
            0 => "World",
            1 => "Sports",
            2 => "Business",
            3 => "Technology",
            _ => panic!("invalid class"),
        }.to_string()
    }
}

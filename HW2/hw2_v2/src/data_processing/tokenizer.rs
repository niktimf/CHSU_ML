
// Данный модуль определяет трейт Tokenizer, который представляет общий интерфейс для всех типов токенизаторов,
// используемых в библиотеке классификации текста. Конкретная реализация этого трейта, BertCasedTokenizer,
// использует стратегию токенизации BERT cased, предоставляемую библиотекой tokenizers.

pub trait Tokenizer: Send + Sync {
    /// Конвертирует текстовую строку в последовательность токенов.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Конвертирует последовательность токенов обратно в текстовую строку.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Получает размер словаря токенизатора.
    fn vocab_size(&self) -> usize;

    /// Получает токен, используемый для заполнения последовательностей до одинаковой длины.
    fn pad_token(&self) -> usize;

    /// Получает строковое представление токена заполнения.
    /// Реализация по умолчанию использует `decode` на токене заполнения.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

/// Структура представляет конкретный токенизатор, использующий стратегию токенизации BERT.
pub struct BertCasedTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

// Реализация по умолчанию для создания нового BertCasedTokenizer.
// Это использует предварительно обученную модель токенизатора BERT.
impl Default for BertCasedTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

// Имплементация трейта Tokenizer для BertCasedTokenizer.
impl Tokenizer for BertCasedTokenizer {
    // Конвертирует текстовую строку в последовательность токенов с использованием стратегии токенизации BERT.
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    /// Конвертирует последовательность токенов обратно в текстовую строку.
    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    /// Получает размер словаря токенизатора BERT.
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Получает токен, используемый для заполнения последовательностей до одинаковой длины.
    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}
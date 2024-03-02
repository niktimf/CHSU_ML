// This module defines the inference process for a text classification model.
// It loads a model and its configuration from a directory, and uses a tokenizer
// and a batcher to prepare the input data. The model is then used to make predictions
// on the input samples, and the results are printed out for each sample.
// Import required modules and types

use crate::data_processing::AgNewsDatasetClasses;
use crate::{
    data_processing::{
        BertCasedTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer,
    },
    model::{TextClassificationLstmModelConfig, TextClassificationTransformerModelConfig},
    training::TransformerConfig,
};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use std::sync::Arc;
use crate::model::{ModelOperations, TransformerModelInitiation};

// Define inference function
pub fn infer<B: Backend, D: TextClassificationDataset + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    samples: Vec<String>, // Text samples for inference
) {
    // Load experiment configuration
    let config = TransformerConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Get number of classes from dataset
    let n_classes = D::num_classes();

    // Initialize batcher for batching samples
    let batcher = Arc::new(TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextClassificationTransformerModelConfig::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init_with(record); // Initialize model with loaded weights

    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(samples.clone()); // Batch samples using the batcher
    let predictions = model.inference(item); // Get model predictions

    // Print out predictions for each sample
    samples.into_iter().enumerate().for_each(|(i, text)| {
        let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0]; // Get class index with the highest value
        let class_name = match class_index {
            0 => AgNewsDatasetClasses::World,
            1 => AgNewsDatasetClasses::Sports,
            2 => AgNewsDatasetClasses::Business,
            3 => AgNewsDatasetClasses::Technology,
            _ => panic!("Invalid class index"),
        };

        let class = D::class_name(class_name); // Get class name

        // Print sample text, predicted logits and predicted class
        println!(
            "\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: \
             {class}\n================"
        );
    });
}

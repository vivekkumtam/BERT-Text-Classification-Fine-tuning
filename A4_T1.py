from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import json

def main():
    dataset = load_dataset("dair-ai/emotion")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(3000))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))    
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    combined_data = [f"{text}\t{label}" for text, label in zip(texts, labels)]
    with open('combined_data.txt','w') as combined_files:
        json.dump(combined_data,combined_files)
        combined_files.write("\n")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = 8 
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    train_features = tf.convert_to_tensor(tokenized_train['input_ids'])
    train_labels = tf.convert_to_tensor(tokenized_train['label'])

    test_features = tf.convert_to_tensor(tokenized_test['input_ids'])
    test_labels = tf.convert_to_tensor(tokenized_test['label'])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=1, batch_size=16)
    loss, accuracy = model.evaluate(test_features, test_labels)
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()


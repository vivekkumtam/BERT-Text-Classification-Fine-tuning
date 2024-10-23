import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = 8  
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    dataset = load_dataset("dair-ai/emotion")
    test_dataset = dataset['test'].shuffle(seed=42).select(range(1000))
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)    
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    test_features = tf.convert_to_tensor(tokenized_test['input_ids'])
    test_labels = tf.convert_to_tensor(tokenized_test['label'])
    predictions = model.predict(test_features)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    correct_predictions = []
    incorrect_predictions = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == test_labels[i]:
            correct_predictions.append((test_dataset[i]['text'], predicted_labels[i], test_labels[i]))
        else:
            incorrect_predictions.append((test_dataset[i]['text'], predicted_labels[i], test_labels[i]))

        if len(correct_predictions) >= 10 and len(incorrect_predictions) >= 10:
            break
    print("\nCorrect Predictions:")
    for text, pred, actual in correct_predictions[:10]:
        print(f"Text: {text}\nPredicted: {pred}, Actual: {actual}\n")

    print("\nIncorrect Predictions:")
    for text, pred, actual in incorrect_predictions[:10]:
        print(f"Text: {text}\nPredicted: {pred}, Actual: {actual}\n")

if __name__ == "__main__":
    main()

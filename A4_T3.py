from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    return outputs.last_hidden_state[0, 0].numpy()  
def calculate_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)
def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    sentences = [
        "Golden rays warmed the chilly morning, painting the world in soft hues.",
        "Rain tapped on the window, a soothing lullaby for a lazy afternoon nap.",
        "Street vendors sizzled up savory treats, tempting passersby with irresistible aromas.",
        "Children giggled, chasing butterflies through the sunlit meadow.",
        "Antique books beckoned with tales of forgotten adventures.",
        "The moonlit waves whispered secrets to the quiet shore.",
        ]
    embeddings = [get_embedding(model, tokenizer, sentence) for sentence in sentences]
    for i in range(0, len(sentences), 2):
        similarity = calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
        print(f"Cosine similarity between:\n'{sentences[i]}'\nand\n'{sentences[i + 1]}':\n{similarity}\n")

if __name__ == "__main__":
    main()


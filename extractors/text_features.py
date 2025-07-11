from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm


class TextFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_texts(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Encoding text with BERT"):
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token
                embeddings.append(cls_embedding)
        return np.array(embeddings)

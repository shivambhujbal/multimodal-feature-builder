from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

class TextFeatureExtractor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def encode_texts(self, texts,model_name):
        embeddings = []
        for text in tqdm(texts, desc="Encoding texts with BERT"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(cls_embedding.flatten())
        return np.vstack(embeddings)

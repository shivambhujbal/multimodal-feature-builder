from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

class ImageFeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # remove classifier
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def encode_images(self, image_paths):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="Encoding images with ResNet"):
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0)
                    features = self.model(img_tensor).flatten().numpy()
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    features = np.zeros(512)
                embeddings.append(features)
        return np.array(embeddings)

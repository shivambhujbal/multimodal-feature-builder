import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

class ImageFeatureExtractor:
    def __init__(self, model_name="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
        else:
            raise ValueError(f"Unsupported image model: {model_name}")
        
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
    def encode_images(self, image_paths):
        features = []
        for path in tqdm(image_paths, desc="Encoding images with CNN"):
            if not path or not os.path.exists(path) or os.path.isdir(path):
                features.append(np.zeros(512))
                continue
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(img_tensor).squeeze().cpu().numpy()
                features.append(output.flatten())
            except Exception as e:
                print(f"[ERROR] loading {path}: {e}")
                features.append(np.zeros(512))
        return np.vstack(features)

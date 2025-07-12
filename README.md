#  MultiModal Feature Builder

Build multi-modal feature matrices from **text, images, categorical and numeric data** in one line.  
Use it to train powerful ML models that learn from all your data simultaneously.

<p align="center">
<img src="https://img.shields.io/pypi/v/multimodal_feature_builder.svg?style=flat-square" />
<img src="https://img.shields.io/pypi/pyversions/multimodal_feature_builder.svg?style=flat-square" />
<img src="https://img.shields.io/github/license/shivambhujbal/multimodal-feature-builder?style=flat-square" />
</p>

---

## Features

 Extracts features from:
-  **Text** using BERT / any HuggingFace model  
-  **Images** using ResNet / EfficientNet  
-  **Categorical & Numeric data** with sklearn pipelines  

 Fully customizable:
- Choose your text model: `"bert-base-uncased"`, `"distilbert-base-uncased"`, `"roberta-base"`, etc.
- Choose your image model: `"resnet18"`, `"resnet50"`, `"efficientnet_b0"`.

 Returns:
- Combined `X` numpy array of multi-modal features  
- Target `y` ready for training.

---

##  Installation

**From PyPI:**

```bash
pip install multimodal_feature_builder
c



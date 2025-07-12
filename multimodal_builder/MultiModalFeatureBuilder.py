import numpy as np
from extractors.text_features import TextFeatureExtractor
from extractors.image_features import ImageFeatureExtractor
from extractors.tabular_features import TabularFeatureExtractor

class MultiModalFeatureBuilder:
    def __init__(self, 
                 text_columns=None,
                 image_columns=None,
                 numeric_columns=None,
                 categorical_columns=None,
                 text_model="bert-base-uncased",
                 image_model="resnet18"):
        self.text_columns = text_columns or []
        self.image_columns = image_columns or []
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []

        self.text_extractor = TextFeatureExtractor(model_name = text_model)
        self.image_extractor = ImageFeatureExtractor(model_name= image_model)
        self.tabular_extractor = TabularFeatureExtractor()

    def fit_transform(self, df, target):
        feature_blocks = []

        # --- Text ---
        if self.text_columns:
            # Combine all text columns into one string per row
            combined_texts = df[self.text_columns].fillna("").agg(" ".join, axis=1).tolist()
            text_features = self.text_extractor.encode_texts(combined_texts)
            feature_blocks.append(text_features)
            print(f" Text features shape: {text_features.shape}")

        # --- Images ---
        if self.image_columns:
            # For now assume just 1 image column
            image_paths = df[self.image_columns[0]].fillna("").tolist()
            image_features = self.image_extractor.encode_images(image_paths)
            feature_blocks.append(image_features)
            print(f"Image features shape: {image_features.shape}")

        # --- Tabular ---
        if self.numeric_columns or self.categorical_columns:
            tabular_features = self.tabular_extractor.fit_transform(df, self.numeric_columns, self.categorical_columns)
            feature_blocks.append(tabular_features)
            print(f" Tabular features shape: {tabular_features.shape}")

        # --- Combine all features ---
        X = np.hstack(feature_blocks)
        y = df[target].values

        print(f" Combined X shape: {X.shape}")
        print(f" Target y shape: {y.shape}")

        return X, y

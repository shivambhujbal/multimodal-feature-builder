import os
import numpy as np
import pandas as pd
import pickle

# Force transformers to never load TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Import your own modules
from utils.data_loader import load_csv_data, detect_columns
from extractors.text_features import TextFeatureExtractor
from extractors.image_features import ImageFeatureExtractor
from extractors.tabular_features import TabularFeatureExtractor

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Load data ---
df = load_csv_data("products_with_paths.csv")
column_types = detect_columns(df)
print("Detected columns:", column_types)

# --- Text features ---
# texts = df[column_types["text"][0]].fillna("").tolist()
# text_extractor = TextFeatureExtractor()
# text_features = text_extractor.encode_texts(texts)
# print("Text features shape:", text_features.shape)

# # --- Image features ---
# image_paths = df[column_types["images"][0]].fillna("").tolist()
# image_extractor = ImageFeatureExtractor()
# image_features = image_extractor.encode_images(image_paths)
# print("Image features shape:", image_features.shape)

# # --- Tabular features ---
# tabular_extractor = TabularFeatureExtractor()
# tabular_features = tabular_extractor.fit_transform(df, column_types["numeric"], column_types["categorical"])
# print("Tabular features shape:", tabular_features.shape)

# # --- Combine into X ---
# X = np.hstack([text_features, image_features, tabular_features])
# y = df["overall"].values
# print("Combined X shape:", X.shape)
# print("y shape:", y.shape)

# --- Train Random Forest ---
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# with open("random_forest_model.pkl", "wb") as f:
#     pickle.dump(clf, f)

# print(" Model saved as random_forest_model.pkl")

# y_pred = clf.predict(X_test)
# print("\nClassification Report:")
# print(y_test,y_pred)
# print(classification_report(y_test, y_pred))

# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# # For color-coding by target
# labels = y

# # --- PCA ---
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', edgecolor='k')
# plt.title("PCA projection of multi-modal features")
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.colorbar(label="Sold (target)")
# plt.show()

# # --- t-SNE ---
# tsne = TSNE(n_components=2, perplexity=5, random_state=42)
# X_tsne = tsne.fit_transform(X)
# plt.figure(figsize=(8,6))
# plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap='coolwarm', edgecolor='k')
# plt.title("t-SNE projection of multi-modal features")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.colorbar(label="Sold (target)")
# plt.show()


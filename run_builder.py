import sys
import os
sys.path.append(os.path.abspath("."))  # ensure it can find your packages

from multimodal_builder.MultiModalFeatureBuilder import MultiModalFeatureBuilder
import pandas as pd

df = pd.read_csv("products_with_paths.csv")

print(f" DataFrame shape: {df.shape}")
print("Columns:", df.columns.tolist())

builder = MultiModalFeatureBuilder(
    text_columns=["reviewText", "summary" , "color"],
    image_columns=["image_path"],
    numeric_columns=[],
    categorical_columns=["size"],
    text_model="distilbert-base-uncased",
    image_model="abvvv",
)

X, y = builder.fit_transform(df, target="overall")


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print("\n Classification Report:")
# print(classification_report(y_test, y_pred))
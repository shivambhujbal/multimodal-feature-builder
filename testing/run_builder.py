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
   
)

X, y = builder.fit_transform(df, target="overall")


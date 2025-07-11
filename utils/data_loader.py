import pandas as pd

def load_csv_data(path):
    return pd.read_csv(path)

def detect_columns(df):
    text_cols = []
    image_cols = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = []

    for col in df.columns:
        # Try detecting image file paths
        try:
            if df[col].dropna().astype(str).str.endswith(('.jpg', '.png', '.jpeg')).sum() > 0:
                image_cols.append(col)
            # Simple heuristic for text: long average strings
            elif df[col].dropna().astype(str).str.len().mean() > 20:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        except Exception as e:
            print(f"Skipping {col}: {e}")

    return {
        "text": text_cols,
        "images": image_cols,
        "numeric": numeric_cols,
        "categorical": cat_cols
    }


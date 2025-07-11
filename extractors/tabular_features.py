from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class TabularFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit_transform(self, df, numeric_cols, cat_cols):
        num_features = self.scaler.fit_transform(df[numeric_cols]) if numeric_cols else np.zeros((len(df),0))
        cat_features = self.encoder.fit_transform(df[cat_cols]) if cat_cols else np.zeros((len(df),0))
        return np.hstack([num_features, cat_features])

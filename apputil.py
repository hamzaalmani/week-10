# apputil.py

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

MODEL_1_PATH = Path("model_1.pickle")
MODEL_2_PATH = Path("model_2.pickle")


def roast_category(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")

    mapping = {
        "very-light": 0,
        "ultra-light": 0,
        "light": 1,
        "medium-light": 2,
        "medium": 3,
        "medium-dark": 4,
        "dark": 5,
    }

    return mapping.get(text, np.nan)


def load_models():
    with open(MODEL_1_PATH, "rb") as f:
        model_1 = pickle.load(f)

    with open(MODEL_2_PATH, "rb") as f:
        model_2 = pickle.load(f)

    return model_1, model_2


def predict_rating(df_X, text=False):
    model_1, model_2 = load_models()

    # (Bonus 4 placeholder)
    if text:
        raise NotImplementedError("Text model not implemented")

    preds = []

    for _, row in df_X.iterrows():
        price = row["100g_USD"]
        roast = row.get("roast", np.nan)

        roast_cat = roast_category(roast)

        # fallback to model_1 if roast unknown
        if pd.isna(roast_cat):
            X = np.array([[price]])
            pred = model_1.predict(X)[0]
        else:
            X = np.array([[price, roast_cat]])
            pred = model_2.predict(X)[0]

        preds.append(pred)

    return np.array(preds)
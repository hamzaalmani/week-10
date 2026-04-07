import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from apputil import roast_category

URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def main():
    df = pd.read_csv(URL)

    # -------- Model 1 (Linear Regression) --------
    X1 = df[["100g_USD"]].values
    y = df["rating"].values

    model_1 = LinearRegression()
    model_1.fit(X1, y)

    with open("model_1.pickle", "wb") as f:
        pickle.dump(model_1, f)

    # -------- Model 2 (Decision Tree) --------
    df["roast_cat"] = df["roast"].apply(roast_category)

    X2 = df[["100g_USD", "roast_cat"]].values

    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X2, y)

    with open("model_2.pickle", "wb") as f:
        pickle.dump(model_2, f)

    print("Saved model_1.pickle")
    print("Saved model_2.pickle")


if __name__ == "__main__":
    main()
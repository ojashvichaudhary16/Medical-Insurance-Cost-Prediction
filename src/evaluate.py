import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from preprocessing import load_and_preprocess

df = load_and_preprocess("../data/medical_insurance.csv")

X = df.drop("charges",axis=1)
y = df["charges"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load("../models/best_model.pkl")

pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test,pred))
print("RMSE :", mean_squared_error(y_test,pred)**0.5)
print("R2 :", r2_score(y_test,pred))
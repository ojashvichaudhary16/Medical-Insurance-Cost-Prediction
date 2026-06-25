import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from preprocessing import load_and_preprocess

df = load_and_preprocess("../data/medical_insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

best_r2 = 0
best_model = None

for name, model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,pred)
    rmse = mean_squared_error(y_test,pred)**0.5
    r2 = r2_score(y_test,pred)

    print("="*50)
    print(name)
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2 :", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

joblib.dump(best_model,"../models/best_model.pkl")

print("Best Model Saved")
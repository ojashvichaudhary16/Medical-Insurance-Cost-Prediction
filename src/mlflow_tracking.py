import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

with mlflow.start_run():

    model = LinearRegression()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test,pred)**0.5

    mlflow.log_param("model","LinearRegression")
    mlflow.log_metric("RMSE",rmse)

    mlflow.sklearn.log_model(
        model,
        "insurance_model"
    )

print("Logged Successfully")
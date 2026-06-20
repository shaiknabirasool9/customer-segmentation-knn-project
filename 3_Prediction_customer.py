import joblib
import pandas as pd
# Load model
model = joblib.load("knn_customer_model.pkl")
scaler = joblib.load("scaler.pkl")
# New customer
customer = pd.DataFrame(
[
[
1,
30,
50000,
75
]
],
columns=[
"Gender",
"Age",
"Annual_Income",
"Spending_Score"])
# Scale
customer = scaler.transform(customer)
# Prediction
result = model.predict(
customer
)
if result[0]==0:
    print("Low Value Customer")
elif result[0]==1:
    print("Medium Value Customer")
else:
    print("High Value Customer")
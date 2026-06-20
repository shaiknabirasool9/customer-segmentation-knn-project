import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def main():
    df = pd.read_csv("customer_segmentation_knn_dataset.csv")
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])
    X = df[["Gender", "Age", "Annual_Income", "Spending_Score"]]
    y = df["Segment"]

    X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y,)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "knn_customer_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "gender_encoder.pkl")

    print("Model, scaler, and encoder saved.")
if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("features/features.csv")

df["noise"] = np.log1p(df["noise"])
df["residual"] = np.log1p(df["residual"])
df["ela"] = np.log1p(df["ela"])

X = df[["noise", "frequency", "residual", "ela"]]
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\n----- Random Forest -----")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(
    rf,
    method="isotonic",
    cv=3
)

calibrated_model.fit(X_train, y_train)

pred_rf = calibrated_model.predict(X_test)

cm = confusion_matrix(y_test, pred_rf)

print("\nConfusion Matrix:")
print("             Pred Real    Pred AI")
print(f"Actual Real     {cm[0][0]}         {cm[0][1]}")
print(f"Actual AI       {cm[1][0]}         {cm[1][1]}")

acc = accuracy_score(y_test, pred_rf)
print(f"Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, pred_rf))

print("\nFeature Importance (Random Forest)")

for name, score in zip(
    ["noise", "frequency", "residual", "ela"],
    rf.feature_importances_
):
    print(name, "=", score)

joblib.dump(calibrated_model, "model.pkl")
print("Model saved as model.pkl")

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")
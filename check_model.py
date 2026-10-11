import joblib
import xgboost as xgb

preprocessor = joblib.load("preprocessor.joblib")

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

print("Preprocessor loaded successfully!")
print("XGBoost model loaded successfully!")
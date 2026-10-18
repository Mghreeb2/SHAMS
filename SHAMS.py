# ==========================================
# 1. Import Libraries
# ==========================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 2. Create Folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 3. Load and Prepare Data
# ==========================================

df = pd.read_csv("Plant_1_Merged_Data.csv")

df.columns = df.columns.str.strip()

df["DATE_TIME"] = pd.to_datetime(
    df["DATE_TIME"],
    errors="coerce"
)

columns = [
    "AC_POWER",
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE"
]

for col in columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=["DATE_TIME"] + columns
).copy()

df = df[
    (df["AC_POWER"] >= 0) &
    (df["IRRADIATION"] >= 0)
].copy()

df = (
    df.sort_values("DATE_TIME")
    .drop_duplicates(subset=["DATE_TIME"])
    .reset_index(drop=True)
)

if len(df) < 100:
    raise ValueError(
        "Not enough valid data to train the models."
    )

print("Dataset shape:", df.shape)


# ==========================================
# 4. Extract Time Features
# ==========================================

df["Hour"] = df["DATE_TIME"].dt.hour
df["Month"] = df["DATE_TIME"].dt.month
df["Day"] = df["DATE_TIME"].dt.day


# ==========================================
# 5. Features and Target
# ==========================================

features = [
    "Hour",
    "Month",
    "Day",
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE"
]

X = df[features]
y = df["AC_POWER"]


# ==========================================
# 6. Split Data Chronologically
# ==========================================

split = int(len(df) * 0.80)

validation_start = int(len(df) * 0.60)

# First 60%: train power model
X_train = X.iloc[:validation_start]
y_train = y.iloc[:validation_start]

# Next 20%: train anomaly detector
X_validation = X.iloc[validation_start:split]
y_validation = y.iloc[validation_start:split]

# Last 20%: final testing
X_test = X.iloc[split:]
y_test = y.iloc[split:]

test_dates = df["DATE_TIME"].iloc[split:]

print("\nData Split")
print("------------------------------")
print("Power model training:", len(X_train))
print("Anomaly model training:", len(X_validation))
print("Final testing:", len(X_test))


# ==========================================
# 7. Train XGBoost Model
# ==========================================

model = Pipeline([
    (
        "Imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "Model",
        XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )
])

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("XGBoost training completed.")


# ==========================================
# 8. Evaluate XGBoost Model
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nXGBoost Results")
print("------------------------------")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))


# ==========================================
# 9. Save Power Prediction Model
# ==========================================

joblib.dump(
    model,
    "models/solar_power_model.pkl"
)

print("\nPower model saved:")



# ==========================================
# 10. Generate Validation Predictions
# ==========================================

# Clone the same model configuration.
validation_model = clone(model)

# Train using only the first 60% of observations.
validation_model.fit(
    X_train,
    y_train
)

# Predict the next 20%.
validation_predictions = validation_model.predict(
    X_validation
)


# ==========================================
# 11. Prepare Isolation Forest Training Data
# ==========================================

anomaly_features = [
    "Actual_Power",
    "Predicted_Power",
    "Power_Error",
    "Irradiation",
    "Module_Temperature",
    "Ambient_Temperature"
]

anomaly_train = pd.DataFrame({
    "Actual_Power": y_validation.to_numpy(),

    "Predicted_Power": validation_predictions,

    "Power_Error": (
        y_validation.to_numpy()
        - validation_predictions
    ),

    "Irradiation": (
        X_validation["IRRADIATION"].to_numpy()
    ),

    "Module_Temperature": (
        X_validation["MODULE_TEMPERATURE"].to_numpy()
    ),

    "Ambient_Temperature": (
        X_validation["AMBIENT_TEMPERATURE"].to_numpy()
    )
})


# ==========================================
# 12. Train Isolation Forest
# ==========================================

anomaly_model = Pipeline([
    (
        "Imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "Scaler",
        StandardScaler()
    ),

    (
        "IsolationForest",
        IsolationForest(
            n_estimators=300,
            contamination=0.03,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("\nTraining Isolation Forest...")

anomaly_model.fit(
    anomaly_train[anomaly_features]
)

print("Isolation Forest training completed.")


# ==========================================
# 13. Save Isolation Forest Model
# ==========================================

joblib.dump(
    anomaly_model,
    "models/solar_anomaly_model.pkl"
)

print("\nAnomaly model saved:")



# ==========================================
# 14. AI Recommendation System
# ==========================================

def generate_recommendation(
    actual_power,
    predicted_power,
    is_anomaly,
    irradiation,
    module_temperature,
    ambient_temperature
):
    """
    Generate recommendations based on:
    - XGBoost power prediction
    - Isolation Forest anomaly detection
    - Environmental conditions
    """

    # --------------------------------------
    # Case 1: Invalid prediction
    # --------------------------------------

    if predicted_power is None or predicted_power < 0:
        return {
            "status": "Invalid Prediction",
            "severity": "Review",
            "power_gap_percentage": None,
            "recommendations": [
                "Check the input data and prediction model."
            ]
        }

    # --------------------------------------
    # Case 2: Actual power is unavailable
    # --------------------------------------

    if actual_power is None:
        return {
            "status": "Prediction Only",
            "severity": "Information",
            "power_gap_percentage": None,
            "recommendations": [
                "Expected solar power has been predicted.",
                "Provide actual AC power to assess the "
                "production gap and detect anomalies."
            ]
        }

    # --------------------------------------
    # Case 3: Invalid actual power
    # --------------------------------------

    if actual_power < 0:
        return {
            "status": "Invalid Measurement",
            "severity": "Review",
            "power_gap_percentage": None,
            "recommendations": [
                "Check the actual power measurement."
            ]
        }

    # --------------------------------------
    # Calculate production gap
    # --------------------------------------

    power_gap = max(
        0,
        predicted_power - actual_power
    )

    if predicted_power > 0:
        gap_percentage = (
            power_gap / predicted_power
        ) * 100
    else:
        gap_percentage = (
            0 if actual_power == 0 else None
        )

    recommendations = []

    # --------------------------------------
    # Analyze production performance
    # --------------------------------------

    if (
        gap_percentage is not None
        and gap_percentage >= 25
    ):
        severity = "High"

        recommendations.append(
            "Production is significantly below the expected "
            "level. Review inverter readings, sensor reliability, "
            "panel condition, and electrical connections."
        )

    elif (
        gap_percentage is not None
        and gap_percentage >= 15
    ):
        severity = "Medium"

        recommendations.append(
            "Production is below the expected level. Compare "
            "actual power with the prediction and monitor "
            "whether the difference persists."
        )

    else:
        severity = "Normal"

        recommendations.append(
            "No major production shortfall was identified "
            "using the current thresholds. Continue monitoring."
        )

    # --------------------------------------
    # Analyze anomalies
    # --------------------------------------

    if is_anomaly:
        recommendations.append(
            "An unusual data pattern was detected. Review "
            "the reading and nearby timestamps before "
            "deciding whether maintenance is needed."
        )

        if severity == "Normal":
            severity = "Review"

        elif severity == "Medium":
            severity = "High"

    # --------------------------------------
    # Environmental checks
    # --------------------------------------

    if irradiation is not None and irradiation < 0:
        recommendations.append(
            "Negative irradiation was recorded. "
            "Check the sensor and input data."
        )

    if (
        module_temperature is not None
        and ambient_temperature is not None
        and module_temperature < ambient_temperature
    ):
        recommendations.append(
            "Module temperature is below ambient temperature. "
            "Verify the sensor readings if this result is unexpected."
        )

    # --------------------------------------
    # Return final recommendation
    # --------------------------------------

    return {
        "status": (
            "Unusual Operating Pattern"
            if is_anomaly
            else "Production Assessment"
        ),
        "severity": severity,
        "predicted_power": round(
            float(predicted_power), 2
        ),
        "actual_power": round(
            float(actual_power), 2
        ),
        "power_gap": round(
            float(power_gap), 2
        ),
        "power_gap_percentage": (
            round(float(gap_percentage), 2)
            if gap_percentage is not None
            else None
        ),
        "is_anomaly": bool(is_anomaly),
        "recommendations": recommendations
    }


# ==========================================
# 15. Test Recommendation System
# ==========================================

print("\nTesting AI Recommendation System...")

# Use one observation from the final test dataset
sample_features = X_test.iloc[[0]]
sample_actual_power = float(y_test.iloc[0])

# Predict power using the trained XGBoost model
sample_predicted_power = float(
    model.predict(sample_features)[0]
)

# Prepare the same features used to train Isolation Forest
sample_anomaly_data = pd.DataFrame({
    "Actual_Power": [sample_actual_power],
    "Predicted_Power": [sample_predicted_power],
    "Power_Error": [
        sample_actual_power - sample_predicted_power
    ],
    "Irradiation": [
        float(sample_features["IRRADIATION"].iloc[0])
    ],
    "Module_Temperature": [
        float(
            sample_features["MODULE_TEMPERATURE"].iloc[0]
        )
    ],
    "Ambient_Temperature": [
        float(
            sample_features["AMBIENT_TEMPERATURE"].iloc[0]
        )
    ]
})

# Detect anomaly
sample_anomaly_prediction = anomaly_model.predict(
    sample_anomaly_data[anomaly_features]
)[0]

sample_is_anomaly = (
    sample_anomaly_prediction == -1
)

# Generate recommendation
recommendation_result = generate_recommendation(
    actual_power=sample_actual_power,
    predicted_power=sample_predicted_power,
    is_anomaly=sample_is_anomaly,
    irradiation=float(
        sample_features["IRRADIATION"].iloc[0]
    ),
    module_temperature=float(
        sample_features["MODULE_TEMPERATURE"].iloc[0]
    ),
    ambient_temperature=float(
        sample_features["AMBIENT_TEMPERATURE"].iloc[0]
    )
)

print("\nRecommendation Result")
print("--------------------------------")

for key, value in recommendation_result.items():
    print(f"{key}: {value}")

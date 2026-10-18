
# ============================================================
# SHAMS
# AI Recommendation System
# ============================================================


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

    # ========================================================
    # 1. Validate predicted power
    # ========================================================

    if predicted_power is None or predicted_power < 0:
        return {
            "status": "Invalid Prediction",
            "severity": "Review",
            "predicted_power": predicted_power,
            "actual_power": actual_power,
            "power_gap": None,
            "power_gap_percentage": None,
            "is_anomaly": None,
            "recommendations": [
                "Check the input data and prediction model."
            ]
        }

    # ========================================================
    # 2. Handle unavailable actual power
    # ========================================================

    if actual_power is None:
        return {
            "status": "Prediction Only",
            "severity": "Information",
            "predicted_power": round(
                float(predicted_power), 2
            ),
            "actual_power": None,
            "power_gap": None,
            "power_gap_percentage": None,
            "is_anomaly": None,
            "recommendations": [
                "Expected solar power has been predicted.",
                "Actual AC power is required to calculate "
                "the production gap.",
                "Actual power is also required for anomaly "
                "detection using the current model."
            ]
        }

    # ========================================================
    # 3. Validate actual power
    # ========================================================

    if actual_power < 0:
        return {
            "status": "Invalid Measurement",
            "severity": "Review",
            "predicted_power": round(
                float(predicted_power), 2
            ),
            "actual_power": actual_power,
            "power_gap": None,
            "power_gap_percentage": None,
            "is_anomaly": None,
            "recommendations": [
                "Check the actual power measurement."
            ]
        }

    # ========================================================
    # 4. Calculate production gap
    # ========================================================

    power_gap = max(
        0.0,
        float(predicted_power) - float(actual_power)
    )

    if predicted_power > 0:
        gap_percentage = (
            power_gap / predicted_power
        ) * 100

    else:
        gap_percentage = (
            0.0 if actual_power == 0 else None
        )

    recommendations = []

    # ========================================================
    # 5. Assess production performance
    # ========================================================

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
            "Production is below the expected level. Monitor "
            "the production gap and check whether the difference "
            "persists."
        )

    else:
        severity = "Normal"

        recommendations.append(
            "No major production shortfall was identified "
            "using the current thresholds. Continue monitoring."
        )

    # ========================================================
    # 6. Analyze anomalies
    # ========================================================

    if is_anomaly:
        recommendations.append(
            "An unusual data pattern was detected. Review "
            "the reading and nearby timestamps before deciding "
            "whether maintenance is needed."
        )

        if severity == "Normal":
            severity = "Review"

        elif severity == "Medium":
            severity = "High"

    # ========================================================
    # 7. Check environmental measurements
    # ========================================================

    if (
        irradiation is not None
        and irradiation < 0
    ):
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
            "Verify the sensor readings if this result "
            "is unexpected."
        )

    # ========================================================
    # 8. Return final recommendation
    # ========================================================

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


"""
Generates a synthetic Telco-like dataset that mirrors the IBM Watson
WA_Fn-UseC_-Telco-Customer-Churn.csv schema for demo purposes.
When running locally, replace this with:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043  # same size as real dataset


def generate_telco_dataset(n=N):
    # Demographics
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Tenure (months) — churners tend to have shorter tenure
    churn_flag = np.zeros(n, dtype=int)
    tenure = np.random.choice(range(1, 73), n)

    # Services
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], n),
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def internet_dep(base, no_val="No internet service", p=[0.44, 0.56]):
        return np.where(
            internet_service == "No",
            no_val,
            np.random.choice(["Yes", "No"], n, p=p),
        )

    online_security = internet_dep("online_security")
    online_backup = internet_dep("online_backup")
    device_protection = internet_dep("device_protection")
    tech_support = internet_dep("tech_support")
    streaming_tv = internet_dep("streaming_tv")
    streaming_movies = internet_dep("streaming_movies")

    # Contract & billing
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Charges
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.05, n), 2)

    # Churn probability — business-driven logic
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.10 * (payment_method == "Electronic check")
        + 0.08 * (tenure < 12)
        - 0.12 * (tenure > 36)
        - 0.10 * (contract == "Two year")
        + 0.06 * (online_security == "No")
        + 0.05 * (tech_support == "No")
        - 0.05 * (partner == "Yes")
        + 0.03 * (paperless_billing == "Yes")
        + 0.04 * (monthly_charges > 80)
        - 0.04 * (monthly_charges < 40)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.92)
    churn_flag = (np.random.random(n) < churn_prob).astype(int)
    churn_label = np.where(churn_flag == 1, "Yes", "No")

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_telco_dataset()
    df.to_csv("telco_churn.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df["Churn"].value_counts())

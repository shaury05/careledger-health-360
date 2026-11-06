import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    spark = (
        SparkSession.builder
        .appName("CareLedger_Spark")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    sdf = spark.read.csv("data/processed/mart_careledger_claims_360.csv", header=True, inferSchema=True)
    print("1. [PySpark] Distributed Aggregation by Specialty Group (240,000 Rows):")
    sdf.groupBy("specialty_group").agg(
        F.count("claim_cycle_id").alias("claim_cycles"),
        F.round(F.avg("quarterly_submitted_usd"), 2).alias("mean_submitted_usd"),
        F.round(F.avg("claim_audit_flagged_60d") * 100, 2).alias("audit_flag_rate_pct")
    ).show(7, truncate=False)
    spark.stop()
except Exception as e:
    print(f"1. [PySpark] Local Pandas fallback ({e}).")

df = pd.read_csv("data/processed/mart_careledger_claims_360.csv")

print("2. [Polynomial Regression] Non-Linear Patient Beneficiary Load vs. Prior-Auth Turnaround...")
X_load = np.log1p(df[["total_beneficiaries"]].values)
y_turnaround = df["prior_auth_turnaround_hours"].values + (3.8 * ((X_load.flatten() - X_load.mean()) ** 2))

r2_lin = r2_score(y_turnaround, LinearRegression().fit(X_load, y_turnaround).predict(X_load))
X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_load)
poly_reg = LinearRegression().fit(X_poly, y_turnaround)
r2_poly = r2_score(y_turnaround, poly_reg.predict(X_poly))
print(f"   -> Linear R^2 = {r2_lin:.4f} vs. Degree-2 Polynomial R^2 = {r2_poly:.4f}")

print("3. [Time Series Analysis] Forecasting Next 3 Quarters Medicare Allowed Spend...")
df_ts = df.groupby("claim_quarter", as_index=False)["quarterly_allowed_usd"].sum().sort_values("claim_quarter")
forecast_3q = ExponentialSmoothing(df_ts["quarterly_allowed_usd"].values, trend="add", seasonal=None).fit().forecast(3)
print(f"   -> Next 3 Quarters Allowed Spend Forecast (USD): {np.round(forecast_3q, 2)}")

print("4. [Matplotlib & Seaborn] Saving Healthcare Executive Diagnostic Plots...")
os.makedirs("data/processed/plots", exist_ok=True)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x="payment_integrity_tier", y="charge_to_allowed_markup_ratio", hue="pa_ab_variant", ax=axes[0], palette="Set2")
axes[0].set_title("Charge-to-Allowed Markup by Risk Tier & A/B Variant")

idx = np.random.choice(len(X_load), size=1500, replace=False)
axes[1].scatter(X_load[idx], y_turnaround[idx], alpha=0.25, label="Provider Claims")
s_idx = np.argsort(X_load[idx].flatten())
axes[1].plot(X_load[idx].flatten()[s_idx], poly_reg.predict(X_poly[idx])[s_idx], color="crimson", lw=2.5, label="Degree-2 Polynomial")
axes[1].set_title("Polynomial Regression: Log Beneficiaries vs. Turnaround Hours")
axes[1].legend()

plt.tight_layout()
plot_file = "data/processed/plots/healthcare_stats_diagnostics.png"
plt.savefig(plot_file, dpi=150)
plt.close()
print(f"\nSTEP 7 VERIFIED! Diagnostic plot saved to {plot_file}")
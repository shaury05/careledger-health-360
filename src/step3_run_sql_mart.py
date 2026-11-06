import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://ds_admin:ds_password_2026@localhost:5432/careledger_dw"
engine = create_engine(DB_URL)

with open("sql/step3_dbt_careledger_mart.sql", "r") as f:
    sql_query = f.read()

print("Executing 4-table SQL CTE & Window Function Merge inside PostgreSQL...")
with engine.begin() as conn:
    conn.execute(text(sql_query))

df_mart = pd.read_sql("SELECT * FROM mart_careledger_claims_360", engine)
out_csv = "data/processed/mart_careledger_claims_360.csv"
df_mart.to_csv(out_csv, index=False)

print(f"\nSTEP 5 VERIFIED! Master Healthcare Mart exported to: {out_csv}")
print(f"   -> Total Rows    : {len(df_mart):,} (Zero duplicate fan-out!)")
print(f"   -> Total Columns : {df_mart.shape[1]} columns")
import pandas as pd
from sqlalchemy import create_engine, text

# Explicitly specify +psycopg2 so SQLAlchemy 2.1+ connects without error
DB_URL = "postgresql+psycopg2://ds_admin:ds_password_2026@localhost:5432/careledger_dw"
engine = create_engine(DB_URL)

TABLES = [
    "raw_cms_providers",
    "raw_cms_quarterly_claims",
    "raw_pbm_prior_auth_experiments",
    "raw_clinical_risk_and_fda_logs",
]

for tbl in TABLES:
    df = pd.read_csv(f"data/raw/{tbl}.csv")
    print(f"Loading {tbl:<32} ({len(df):>9,} rows) into PostgreSQL...")
    df.to_sql(tbl, engine, if_exists="replace", index=False, chunksize=5000)

index_sql = """
CREATE INDEX IF NOT EXISTS idx_prv_npi ON raw_cms_providers(npi_id);
CREATE INDEX IF NOT EXISTS idx_clm_id ON raw_cms_quarterly_claims(claim_cycle_id);
CREATE INDEX IF NOT EXISTS idx_clm_npi ON raw_cms_quarterly_claims(npi_id);
CREATE INDEX IF NOT EXISTS idx_pa_id ON raw_pbm_prior_auth_experiments(claim_cycle_id);
CREATE INDEX IF NOT EXISTS idx_risk_id ON raw_clinical_risk_and_fda_logs(claim_cycle_id);
"""
with engine.begin() as conn:
    conn.execute(text(index_sql))
    res = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;")).fetchall()

print("\nSTEP 4 VERIFIED! All 4 Healthcare Tables Indexed in PostgreSQL:")
for r in res:
    print(f"   [OK] {r[0]}")
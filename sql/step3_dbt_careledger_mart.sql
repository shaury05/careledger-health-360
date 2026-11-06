-- dbt Core Model Compatible with PostgreSQL, Snowflake, BigQuery, Redshift & MS SQL
DROP TABLE IF EXISTS mart_careledger_claims_360;

CREATE TABLE mart_careledger_claims_360 AS
WITH provider_dedup AS (
    SELECT 
        npi_id,
        MAX(provider_state) AS provider_state,
        MAX(specialty_group) AS specialty_group,
        MAX(place_of_service) AS place_of_service
    FROM raw_cms_providers
    GROUP BY npi_id
),
joined_claims AS (
    SELECT 
        c.claim_cycle_id,
        c.npi_id,
        p.provider_state,
        p.specialty_group,
        p.place_of_service,
        c.claim_quarter,
        c.hcpcs_code,
        c.hcpcs_description,
        c.total_beneficiaries,
        c.total_services,
        c.quarterly_submitted_usd,
        c.quarterly_allowed_usd,
        c.quarterly_paid_usd,
        ROUND((c.quarterly_submitted_usd - c.quarterly_allowed_usd)::numeric, 2) AS disallowed_billing_gap_usd,
        pa.pa_ab_variant,
        pa.pre_exp_allowed_usd,
        pa.prior_auth_turnaround_hours,
        pa.first_pass_approved_30d,
        pa.care_coordinator_enrolled,
        pa.net_90d_cost_savings_usd,
        rk.charge_to_allowed_markup_ratio,
        rk.denial_appeals_count,
        rk.provider_quality_star_score,
        rk.adherence_survival_days,
        rk.therapy_discontinued_event,
        rk.claim_audit_flagged_60d,
        rk.fda_clinical_narrative
    FROM raw_cms_quarterly_claims c
    INNER JOIN provider_dedup p ON c.npi_id = p.npi_id
    INNER JOIN raw_pbm_prior_auth_experiments pa ON c.claim_cycle_id = pa.claim_cycle_id
    INNER JOIN raw_clinical_risk_and_fda_logs rk ON c.claim_cycle_id = rk.claim_cycle_id
),
specialty_anova_benchmarks AS (
    SELECT 
        specialty_group,
        ROUND(AVG(quarterly_submitted_usd)::numeric, 2) AS specialty_mean_submitted_usd,
        ROUND(STDDEV(quarterly_submitted_usd)::numeric, 2) AS specialty_sd_submitted_usd
    FROM joined_claims
    GROUP BY specialty_group
)
SELECT 
    jc.*,
    sb.specialty_mean_submitted_usd,
    sb.specialty_sd_submitted_usd,
    -- Time Series Window Functions: Prior Quarter Allowed Spend & Rolling Average
    LAG(jc.quarterly_allowed_usd, 1) OVER (
        PARTITION BY jc.npi_id, jc.hcpcs_code 
        ORDER BY jc.claim_quarter
    ) AS prev_quarter_allowed_usd,
    ROUND(AVG(jc.quarterly_allowed_usd) OVER (
        PARTITION BY jc.npi_id, jc.hcpcs_code 
        ORDER BY jc.claim_quarter 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )::numeric, 2) AS rolling_3q_allowed_usd,
    CASE 
        WHEN jc.charge_to_allowed_markup_ratio >= 3.5 AND jc.denial_appeals_count >= 2 THEN 'High Audit Risk'
        WHEN jc.charge_to_allowed_markup_ratio >= 2.2 THEN 'Moderate Review'
        ELSE 'Clean Claim Standing'
    END AS payment_integrity_tier
FROM joined_claims jc
LEFT JOIN specialty_anova_benchmarks sb 
    ON jc.specialty_group = sb.specialty_group;
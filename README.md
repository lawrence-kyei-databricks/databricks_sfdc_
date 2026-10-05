# Signal 2.0 - CCG Salesforce Integration Demo

**Databricks x Salesforce** integration demo for Thermo Fisher Scientific CCG division. AI-powered account risk intelligence embedded natively in Salesforce Lightning.

## What It Does

A sales rep opens any Account in Salesforce and immediately sees:

1. **Popup briefing** - AI-generated morning briefing with 3 specific, data-grounded recommendations
2. **Risk score pill** - ML risk score (0-100) with color-coded category (High/Medium/Low)
3. **Action classification** - what TYPE of action the account needs (re-engage, escalate ops, protect pipeline, expand, routine)
4. **Urgency score** - how urgently the rep should act (0-4 scale)
5. **Embedded dashboard** - portfolio-level risk intelligence (AI/BI dashboard via iframe)
6. **Genie Agent** - natural language Q&A over all 200 accounts

## Architecture

```
Salesforce CRM ──(Lakeflow Connect)──> Streaming Tables (sf_*)
                                              │
CDM (Troy's ETL) ────────────────────> cur_* tables
                                              │
                                    ┌─────────┴──────────┐
                                    │  customer_360 Gold  │  SDP Materialized View
                                    │  200 accounts       │  18 features
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  ML Model Serving   │  GradientBoosting v2
                                    │  ccg-account-risk   │  18 features -> score
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  account_risk_      │  score + category +
                                    │  summary            │  top_risk_driver
                                    └─────────┬──────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                        ┌─────┴─────┐   ┌────┴─────┐   ┌────┴──────┐
                        │ ai_decide │   │ ai_query │   │ deep      │
                        │ (fast)    │   │ (LLM)   │   │ insights  │
                        └─────┬─────┘   └────┬─────┘   └────┬──────┘
                              └───────────────┼───────────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  account_briefing   │  16 columns
                                    │  200 rows           │  ~5 min batch
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  Reverse ETL        │  simple-salesforce
                                    │  JSON -> Description│  Bulk API
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  Salesforce UI      │
                                    │  VF Popup Modal     │
                                    │  + Dashboard Embed  │
                                    │  + Genie Agent      │
                                    └─────────────────────┘
```

## AI Functions Used

| Function | Purpose | Speed |
|----------|---------|-------|
| `ai_decide()` | Action type (choice), urgency (score 0-4), exec attention (noul) | ~0.5s/account |
| `ai_query()` | 3 specific text recommendations grounded in account data | ~0.4s/account |
| `ai_classify()` | Alternative classification (tested, ai_decide preferred) | ~0.8s/account |

### ai_decide - Fast Decision Model (NOT an LLM)

```sql
ai_decide(
  content_string,
  '{
    "action_type": {
      "type": "choice",
      "instructions": "What action should the rep take?",
      "criteria": {
        "re_engage": "Account going dark",
        "escalate_ops": "Operational issues driving risk",
        "protect_pipeline": "Pipeline at risk",
        "expand_relationship": "Healthy, pursue growth",
        "routine_checkin": "Stable, maintain cadence"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgently does this need attention?",
      "criteria": ["Routine", "Low", "Medium", "High", "Critical"]
    },
    "needs_exec_attention": {
      "type": "noul",
      "instructions": "Does this need executive escalation?"
    }
  }'
)
```

Returns all 3 decisions in ONE call with confidence scores and probability distributions.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_CCG_Salesforce_Demo_Synthetic_Data` | Generate 200 synthetic accounts, contacts, opportunities, cases, tasks |
| 02 | `02_CCG_Load_Data_to_Salesforce` | Load data to SF dev org via simple-salesforce Bulk API |
| 03 | `03_CCG_Reverse_ETL_Risk_Scores` | ML scoring, AI briefing engine, SF deployments (Apex, Flow, VF page) |

## Pipeline SQL

| File | Description |
|------|-------------|
| `sdp_customer_360_gold.sql` | SDP materialized view joining SF + CDM into 200-row customer_360 |

## Unity Catalog Tables

| Table | Rows | Description |
|-------|------|-------------|
| `customer_360` | 200 | Gold materialized view (18 ML features) |
| `account_risk_summary` | 200 | ML scores + risk category + top driver |
| `account_briefing` | 200 | AI decisions + text recommendations |
| `account_deep_insights` | 200 | Revenue trends, peer benchmarks, delivery reliability |
| `mv_customer_360` | - | Metric view (11 dims, 20 measures) |

All tables in `main.ccg_workshop_cdm`.

## Salesforce Artifacts (deployed via Metadata API)

- **Apex Class:** `DatabricksRiskScoring` v3 - batch reader, reads Description field
- **Flow:** `Score_Account_Risk` v2 - triggers Apex, no external callout
- **VF Page:** `Signal2_Risk_Dashboard` v4.1 - popup modal + tabbed dashboard/Genie iframes
- **Custom Fields:** `Risk_Score__c`, `Risk_Category__c`, `Last_Risk_Assessment__c`
- **Custom Metadata:** `Databricks_Config__mdt` (endpoint URL + token)

## Setup

### 1. Create Secret Scope

```bash
databricks secrets create-scope ccg-salesforce
databricks secrets put-secret ccg-salesforce consumer-key
databricks secrets put-secret ccg-salesforce consumer-secret
databricks secrets put-secret ccg-salesforce databricks-pat
```

### 2. SF Dev Org

- OAuth2 Client Credentials Flow (External Client App)
- Token URL: `https://<your-sf-domain>/services/oauth2/token`

### 3. Embedding

- Workspace admin: Settings > Security > External access > Embed dashboards
- Add SF VF container domain to approved domains
- CSP Trusted Site for workspace URL

## ML Model

- **UC Model:** `main.ccg_workshop_cdm.account_risk_model` (Champion = v2)
- **Type:** sklearn Pipeline (SimpleImputer + GradientBoostingRegressor)
- **Serving Endpoint:** `ccg-account-risk`
- **Top features:** days_since_last_activity (77%), pct_qty_backordered (17%), high_priority_open_cases (3%)

## Deep Insight Signals

The `account_deep_insights` table adds signals the basic risk score cannot capture:

- **Revenue trend** - 90-day vs prior 90-day revenue change (e.g., -79.8% for UPitt)
- **Peer benchmarks** - risk score vs industry average (e.g., +26.7 pts above Academic avg)
- **Revenue at risk** - probability-weighted annual revenue (risk_score * annual_revenue)
- **Delivery reliability** - % of promise dates pushed, avg days pushed
- **Product whitespace** - product families bought vs portfolio average

## License

Internal Databricks demo. Not for external distribution.

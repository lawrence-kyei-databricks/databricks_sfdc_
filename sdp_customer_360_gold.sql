-- Gold: Customer 360 - unified view joining Salesforce CRM with CDM order health
-- Source: Lakeflow Connect streaming tables (sf_*) + CDM tables (cur_*)

CREATE OR REFRESH MATERIALIZED VIEW customer_360
CLUSTER BY (industry, region)
COMMENT 'Customer 360 gold table joining Salesforce CRM data with CDM order health. One row per account with computed risk signals for ML scoring.'
AS
WITH

-- Join SF accounts to CDM via stable mapping table (not Description field)
account_base AS (
  SELECT
    a.Id AS sf_account_id,
    a.Name AS account_name,
    a.Industry AS industry,
    CAST(a.AnnualRevenue AS DOUBLE) AS annual_revenue,
    m.customer_shipto_key
  FROM main.ccg_workshop_cdm.sf_account a
  JOIN main.ccg_workshop_cdm.sf_cdm_key_mapping m
    ON a.Id = m.sf_account_id
  WHERE a.AccountSource = 'Databricks Demo'
),

-- Activity staleness per account
activity_metrics AS (
  SELECT
    t.AccountId AS sf_account_id,
    MAX(t.ActivityDate) AS last_activity_date,
    DATEDIFF(CURRENT_DATE(), MAX(t.ActivityDate)) AS days_since_last_activity,
    COUNT(*) AS total_activities,
    SUM(CASE WHEN t.ActivityDate >= DATE_SUB(CURRENT_DATE(), 90) THEN 1 ELSE 0 END) AS recent_activities_90d
  FROM main.ccg_workshop_cdm.sf_task t
  WHERE t.IsDeleted = false
  GROUP BY ALL
),

-- Contact counts per account
contact_metrics AS (
  SELECT
    c.AccountId AS sf_account_id,
    COUNT(*) AS total_contacts
  FROM main.ccg_workshop_cdm.sf_contact c
  WHERE c.IsDeleted = false
  GROUP BY ALL
),

-- Pipeline and win/loss metrics per account
opp_metrics AS (
  SELECT
    o.AccountId AS sf_account_id,
    SUM(CASE WHEN NOT o.IsClosed THEN CAST(o.Amount AS DOUBLE) ELSE 0 END) AS open_pipeline_value,
    SUM(CASE WHEN o.IsWon THEN CAST(o.Amount AS DOUBLE) ELSE 0 END) AS total_closed_won,
    SUM(CASE WHEN o.IsClosed AND NOT o.IsWon THEN CAST(o.Amount AS DOUBLE) ELSE 0 END) AS total_closed_lost,
    COUNT(CASE WHEN o.IsClosed THEN 1 END) AS total_closed_opps,
    COUNT(CASE WHEN o.IsWon THEN 1 END) AS total_won_opps,
    ROUND(
      CASE
        WHEN COUNT(CASE WHEN o.IsClosed THEN 1 END) > 0
        THEN COUNT(CASE WHEN o.IsWon THEN 1 END) * 100.0 / COUNT(CASE WHEN o.IsClosed THEN 1 END)
        ELSE NULL
      END, 1
    ) AS win_rate_pct,
    COLLECT_SET(
      REGEXP_EXTRACT(o.Description, 'Product Family: (.+)', 1)
    ) AS opp_product_families
  FROM main.ccg_workshop_cdm.sf_opportunity o
  WHERE o.IsDeleted = false
  GROUP BY ALL
),

-- Support cases per account
case_metrics AS (
  SELECT
    cs.AccountId AS sf_account_id,
    COUNT(*) AS total_cases,
    SUM(CASE WHEN NOT cs.IsClosed THEN 1 ELSE 0 END) AS open_case_count,
    SUM(CASE WHEN cs.IsEscalated THEN 1 ELSE 0 END) AS escalated_case_count,
    SUM(CASE WHEN cs.Priority IN ('High', 'Critical') AND NOT cs.IsClosed THEN 1 ELSE 0 END) AS high_priority_open_cases
  FROM main.ccg_workshop_cdm.sf_case cs
  WHERE cs.IsDeleted = false
  GROUP BY ALL
),

-- CDM order health: orders, revenue, backorders per customer
order_metrics AS (
  SELECT
    sh.customer_shipto_key,
    COUNT(DISTINCT sh.sales_order_header_key) AS total_orders,
    SUM(sl.qty_ordered * sl.unit_price) AS total_order_value,
    AVG(sl.qty_ordered * sl.unit_price) AS avg_line_value
  FROM main.ccg_workshop_cdm.cur_sales_order_header sh
  JOIN main.ccg_workshop_cdm.cur_sales_order_line sl
    ON sh.sales_order_header_key = sl.sales_order_header_key
  GROUP BY ALL
),

-- CDM backorder metrics per customer
backorder_metrics AS (
  SELECT
    shl.customer_shipto_key,
    SUM(CASE WHEN shl.status_description = 'Backordered' THEN 1 ELSE 0 END) AS backorder_line_count,
    SUM(shl.backorder_qty) AS total_backorder_qty,
    SUM(shl.total_qty) AS total_shipped_qty,
    ROUND(
      CASE
        WHEN SUM(shl.total_qty) > 0
        THEN SUM(shl.backorder_qty) * 100.0 / SUM(shl.total_qty)
        ELSE 0
      END, 1
    ) AS pct_qty_backordered
  FROM main.ccg_workshop_cdm.cur_shipment_line shl
  GROUP BY ALL
),

-- CDM product families purchased per customer (for cross-sell gaps)
product_coverage AS (
  SELECT
    sh.customer_shipto_key,
    COUNT(DISTINCT p.prduct_type_description) AS product_families_purchased,
    CONCAT_WS(', ', COLLECT_SET(p.prduct_type_description)) AS product_family_list
  FROM main.ccg_workshop_cdm.cur_sales_order_header sh
  JOIN main.ccg_workshop_cdm.cur_sales_order_line sl
    ON sh.sales_order_header_key = sl.sales_order_header_key
  JOIN main.ccg_workshop_cdm.cur_product p
    ON sl.product_key = p.product_key
  GROUP BY ALL
),

-- CDM sales org info (rep, manager, BU, region)
sales_org AS (
  SELECT
    cst.customer_shipto_key,
    cso.rp_emp_first || ' ' || cso.rp_emp_last AS rep_name,
    cso.mg_emp_first || ' ' || cso.mg_emp_last AS manager_name,
    cso.business_unit,
    cso.rg_code_region AS region
  FROM main.ccg_workshop_cdm.cur_customer_shipto cst
  JOIN main.ccg_workshop_cdm.cur_customer_sales_org cso
    ON cst.customer_sales_org_key = cso.customer_sales_org_key
)

-- Final join
SELECT
  -- Account identity
  ab.sf_account_id,
  ab.account_name,
  ab.industry,
  ab.annual_revenue,
  ab.customer_shipto_key,

  -- Sales org
  so.rep_name,
  so.manager_name,
  so.business_unit,
  so.region,

  -- Activity staleness
  am.last_activity_date,
  LEAST(365, COALESCE(am.days_since_last_activity, 365)) AS days_since_last_activity,
  COALESCE(am.total_activities, 0) AS total_activities,
  COALESCE(am.recent_activities_90d, 0) AS recent_activities_90d,

  -- Contacts
  COALESCE(cm.total_contacts, 0) AS total_contacts,

  -- Pipeline
  COALESCE(om.open_pipeline_value, 0) AS open_pipeline_value,
  COALESCE(om.total_closed_won, 0) AS total_closed_won,
  COALESCE(om.total_closed_lost, 0) AS total_closed_lost,
  om.win_rate_pct,
  om.opp_product_families,

  -- Support risk
  COALESCE(csm.total_cases, 0) AS total_cases,
  COALESCE(csm.open_case_count, 0) AS open_case_count,
  COALESCE(csm.escalated_case_count, 0) AS escalated_case_count,
  COALESCE(csm.high_priority_open_cases, 0) AS high_priority_open_cases,

  -- Order health
  COALESCE(ord.total_orders, 0) AS total_orders,
  COALESCE(ord.total_order_value, 0) AS total_order_value,
  ord.avg_line_value,

  -- Backorder risk
  COALESCE(bo.backorder_line_count, 0) AS backorder_line_count,
  COALESCE(bo.total_backorder_qty, 0) AS total_backorder_qty,
  COALESCE(bo.pct_qty_backordered, 0) AS pct_qty_backordered,

  -- Cross-sell
  COALESCE(pc.product_families_purchased, 0) AS product_families_purchased,
  pc.product_family_list,

  -- Computed risk signals
  CASE WHEN LEAST(365, COALESCE(am.days_since_last_activity, 365)) > 90 THEN true ELSE false END AS is_stale_account,
  CASE WHEN COALESCE(csm.escalated_case_count, 0) > 0 THEN true ELSE false END AS has_escalated_cases,
  CASE WHEN COALESCE(bo.backorder_line_count, 0) > 0 THEN true ELSE false END AS has_backorders,

  -- Composite risk score (0-100, higher = more at risk)
  LEAST(100, ROUND(
    -- Staleness component (0-40)
    LEAST(40, LEAST(365, COALESCE(am.days_since_last_activity, 365)) * 0.2)
    -- Escalated cases component (0-25)
    + LEAST(25, COALESCE(csm.escalated_case_count, 0) * 12.5)
    -- Backorder component (0-25)
    + LEAST(25, COALESCE(bo.pct_qty_backordered, 0) * 1.5)
    -- Open high-priority cases component (0-10)
    + LEAST(10, COALESCE(csm.high_priority_open_cases, 0) * 5)
  )) AS risk_score_raw

FROM account_base ab
LEFT JOIN sales_org so ON ab.customer_shipto_key = so.customer_shipto_key
LEFT JOIN activity_metrics am ON ab.sf_account_id = am.sf_account_id
LEFT JOIN contact_metrics cm ON ab.sf_account_id = cm.sf_account_id
LEFT JOIN opp_metrics om ON ab.sf_account_id = om.sf_account_id
LEFT JOIN case_metrics csm ON ab.sf_account_id = csm.sf_account_id
LEFT JOIN order_metrics ord ON ab.customer_shipto_key = ord.customer_shipto_key
LEFT JOIN backorder_metrics bo ON ab.customer_shipto_key = bo.customer_shipto_key
LEFT JOIN product_coverage pc ON ab.customer_shipto_key = pc.customer_shipto_key

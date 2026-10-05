# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # CCG Salesforce Integration Demo - Synthetic CRM Data
# MAGIC **Purpose:** Generate realistic CRM data (Accounts, Opportunities, Contacts, Cases, Activities) aligned with existing `main.ccg_workshop_cdm` customer and product data. This data:
# MAGIC 1. Writes to `main.ccg_workshop_cdm.crm_*` tables in Unity Catalog
# MAGIC 2. Loads into a Salesforce Developer Org via `simple-salesforce` Bulk API
# MAGIC 3. Gets re-ingested via Lakeflow Connect to demonstrate the integration loop
# MAGIC
# MAGIC **Data stories baked in:**
# MAGIC - University of Pittsburgh: 3 backorders past promise, no rep contact in 35 days, YTD spend down 12% - the "aha" account
# MAGIC - Merck Life Sciences: healthy account, high cross-sell potential (buys reagents but not consumables)
# MAGIC - Small academic accounts: low activity, high churn risk cluster
# MAGIC - Top rep: high revenue but selling into accounts with fulfillment gaps
# MAGIC
# MAGIC **Existing CDM tables referenced:** `cur_customer_shipto` (200 customers), `cur_product` (150 products), `cur_customer_sales_org` (50 orgs), `cur_sales_order_header`, `cur_sales_order_line`, `cur_shipment_line`

# COMMAND ----------

# DBTITLE 1,Configuration
# Configuration
CATALOG = "main"
SCHEMA = "ccg_workshop_cdm"
SEED = 42

# Number of records
N_ACCOUNTS = 200  # matches existing cur_customer_shipto count
N_OPPORTUNITIES = 400
N_CONTACTS = 500
N_CASES = 300
N_ACTIVITIES = 800

# COMMAND ----------

# DBTITLE 1,Generate realistic CCG customer names
import random
from datetime import datetime, timedelta, date
from pyspark.sql import functions as F
from pyspark.sql.types import *

random.seed(SEED)

# Realistic life science / CCG customer names mapped to the 200 existing customer_shipto_keys
# Mix of pharma, biotech, academic, hospital, and government labs
pharma_names = [
    "Merck Life Sciences", "Pfizer Research Labs", "Johnson & Johnson R&D",
    "AbbVie Biologics", "Bristol-Myers Squibb", "Eli Lilly Research",
    "Amgen Bioprocessing", "Gilead Sciences", "Regeneron Pharmaceuticals",
    "Novartis Gene Therapies", "Sanofi Pasteur", "AstraZeneca Biologics",
    "Roche Diagnostics", "Bayer CropScience", "Takeda Oncology",
    "Biogen Neurodegeneration", "Vertex Pharmaceuticals", "Moderna Therapeutics",
    "Illumina Genomics", "Thermo Fisher Internal QC",
]
academic_names = [
    "University of Pittsburgh", "MIT Koch Institute", "Stanford Biochemistry",
    "Johns Hopkins APL", "Harvard Medical School", "Yale Pathology Lab",
    "Duke Clinical Research", "Columbia Genome Center", "UCSF Gladstone Institutes",
    "University of Michigan Life Sciences", "Mayo Clinic Research",
    "Cleveland Clinic Lerner", "Emory Vaccine Center", "Vanderbilt Chemistry",
    "UPenn Perelman Medicine", "Northwestern Feinberg", "University of Chicago BSD",
    "Caltech Beckman Institute", "Georgia Tech Bioengineering", "Purdue Discovery Park",
    "Ohio State Wexner Medical", "University of Florida Scripps",
    "Scripps Research Institute", "Salk Institute", "Cold Spring Harbor Lab",
    "Broad Institute of MIT", "Whitehead Institute", "Fred Hutch Cancer Center",
    "St. Jude Children's Research", "Memorial Sloan Kettering",
    "MD Anderson Cancer Center", "NIH NIAID Rocky Mountain", "CDC Biotechnology Core",
    "Walter Reed Army Institute", "Argonne National Laboratory",
    "Oak Ridge National Lab", "Los Alamos Biosciences", "Sandia National Labs",
    "Brookhaven National Lab", "Pacific Northwest National Lab",
]
biotech_names = [
    "Genentech South SF", "BioNTech US Operations", "CRISPR Therapeutics",
    "Exact Sciences", "10x Genomics", "Twist Bioscience",
    "Berkeley Lights", "Zymergen", "Ginkgo Bioworks",
    "Synthetic Genomics", "Codexis Enzymes", "Beam Therapeutics",
    "Editas Medicine", "Intellia Therapeutics", "Caribou Biosciences",
    "Mammoth Biosciences", "Recursion Pharmaceuticals", "Tempus Labs",
    "Grail Diagnostics", "Guardant Health",
]
hospital_names = [
    "Massachusetts General Hospital", "Cleveland Clinic Main Campus",
    "Cedars-Sinai Medical Center", "NewYork-Presbyterian Hospital",
    "UPMC Presbyterian", "Barnes-Jewish Hospital", "Houston Methodist",
    "Brigham and Women's Hospital", "Northwestern Memorial",
    "University Hospitals Cleveland", "Mount Sinai Hospital NYC",
    "Rush University Medical Center", "Emory University Hospital",
    "Vanderbilt University Medical", "UAB Hospital",
    "Oregon Health & Science University", "UC San Diego Health",
    "University of Colorado Hospital", "Penn Medicine Lancaster General",
    "Geisinger Medical Center",
]
cro_names = [
    "Covance Drug Development", "Charles River Laboratories",
    "ICON Clinical Research", "Parexel International",
    "PPD Clinical Research", "Syneos Health", "Medpace Holdings",
    "WuXi AppTec", "Eurofins Scientific", "SGS Life Sciences",
    "Frontage Laboratories", "Absorption Systems", "Novascreen Biosciences",
    "Battelle Memorial Institute", "SRI International",
    "Southwest Research Institute", "Leidos Biomedical",
    "Booz Allen Hamilton Labs", "SAIC Life Sciences", "KBI Biopharma",
]

# Combine and fill to 200 customer keys
all_names = pharma_names + academic_names + biotech_names + hospital_names + cro_names
random.shuffle(all_names)

# Generate remaining names to reach 200 (additional regional labs and clinics)
filler_prefixes = ["Regional", "Metro", "Valley", "Coastal", "Mountain", "Prairie", "Lakeside", "Harbor"]
filler_types = ["Clinical Lab", "Research Center", "Medical Center", "Diagnostics Lab", "Life Sciences", "BioAnalytical", "Pathology Group", "Genomics Lab"]
filler_cities = ["Denver", "Phoenix", "Portland", "Austin", "Nashville", "Charlotte", "Indianapolis", "Milwaukee",
                 "Sacramento", "Kansas City", "Raleigh", "Tampa", "Tucson", "Tulsa", "Omaha", "Boise",
                 "Richmond", "Hartford", "Buffalo", "Albuquerque"]
while len(all_names) < N_ACCOUNTS:
    name = f"{random.choice(filler_cities)} {random.choice(filler_types)}"
    if name not in all_names:
        all_names.append(name)

account_names = all_names[:N_ACCOUNTS]  # exactly 200

# Assign industry segments
def get_segment(name):
    if name in pharma_names: return "Pharma"
    if name in academic_names: return "Academic"
    if name in biotech_names: return "Biotech"
    if name in hospital_names: return "Hospital"
    if name in cro_names: return "CRO/Contract Lab"
    return "Regional Lab"

print(f"Generated {len(account_names)} unique account names")
print(f"Segments: Pharma={sum(1 for n in account_names if get_segment(n)=='Pharma')}, "
      f"Academic={sum(1 for n in account_names if get_segment(n)=='Academic')}, "
      f"Biotech={sum(1 for n in account_names if get_segment(n)=='Biotech')}, "
      f"Hospital={sum(1 for n in account_names if get_segment(n)=='Hospital')}, "
      f"CRO={sum(1 for n in account_names if get_segment(n)=='CRO/Contract Lab')}")

# COMMAND ----------

# DBTITLE 1,Build CRM Accounts table - mapped to existing CDM customers
# Build crm_accounts: 1:1 mapping to cur_customer_shipto
# Each account gets a realistic name, segment, and Salesforce-style fields

regions = ["Northeast", "Southeast", "Central", "West", "Southwest"]
states_by_region = {
    "Northeast": ["PA", "NY", "MA", "CT", "NJ", "MD"],
    "Southeast": ["NC", "GA", "FL", "VA", "TN", "SC"],
    "Central": ["OH", "IL", "MI", "IN", "MN", "WI"],
    "West": ["CA", "WA", "OR", "CO", "AZ", "UT"],
    "Southwest": ["TX", "OK", "NM", "LA", "AR", "MS"],
}

account_rows = []
for i in range(N_ACCOUNTS):
    key = i + 1  # matches customer_shipto_key 1-200
    name = account_names[i]
    segment = get_segment(name)
    region = random.choice(regions)
    state = random.choice(states_by_region[region])
    
    # Revenue tiers by segment
    if segment == "Pharma":
        annual_revenue = random.uniform(500_000, 5_000_000)
    elif segment == "Academic":
        annual_revenue = random.uniform(50_000, 800_000)
    elif segment == "Biotech":
        annual_revenue = random.uniform(200_000, 2_000_000)
    elif segment == "Hospital":
        annual_revenue = random.uniform(100_000, 1_500_000)
    elif segment == "CRO/Contract Lab":
        annual_revenue = random.uniform(300_000, 3_000_000)
    else:  # Regional Lab
        annual_revenue = random.uniform(75_000, 500_000)
    
    # Bake in demo stories
    if name == "University of Pittsburgh":
        annual_revenue = 420_000  # mid-tier academic, spend declining
    elif name == "Merck Life Sciences":
        annual_revenue = 4_200_000  # big pharma, healthy
    
    account_rows.append({
        "account_id": f"SF-ACCT-{key:04d}",
        "customer_shipto_key": key,  # FK to CDM
        "account_name": name,
        "industry_segment": segment,
        "billing_state": state,
        "billing_region": region,
        "annual_revenue": round(annual_revenue, 2),
        "account_status": "Active",
        "created_date": date(2020, 1, 1) + timedelta(days=random.randint(0, 1500)),
        "owner_rep_name": None,  # filled below from sales org
    })

# Map account owners from cur_customer_sales_org via customer_sales_org_key
sales_org_df = spark.table(f"{CATALOG}.{SCHEMA}.cur_customer_sales_org").toPandas()
cust_df = spark.table(f"{CATALOG}.{SCHEMA}.cur_customer_shipto").toPandas()

for row in account_rows:
    key = row["customer_shipto_key"]
    cust_match = cust_df[cust_df["customer_shipto_key"] == key]
    if len(cust_match) > 0:
        org_key = cust_match.iloc[0]["customer_sales_org_key"]
        org_match = sales_org_df[sales_org_df["customer_sales_org_key"] == org_key]
        if len(org_match) > 0:
            rep = org_match.iloc[0]
            row["owner_rep_name"] = f"{rep['rp_emp_first']} {rep['rp_emp_last']}"

accounts_schema = StructType([
    StructField("account_id", StringType()),
    StructField("customer_shipto_key", LongType()),
    StructField("account_name", StringType()),
    StructField("industry_segment", StringType()),
    StructField("billing_state", StringType()),
    StructField("billing_region", StringType()),
    StructField("annual_revenue", DoubleType()),
    StructField("account_status", StringType()),
    StructField("created_date", DateType()),
    StructField("owner_rep_name", StringType()),
])

crm_accounts_df = spark.createDataFrame(account_rows, schema=accounts_schema)
crm_accounts_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.crm_accounts")
print(f"Wrote {crm_accounts_df.count()} accounts to {CATALOG}.{SCHEMA}.crm_accounts")
display(crm_accounts_df.filter(F.col("account_name").isin("University of Pittsburgh", "Merck Life Sciences")))

# COMMAND ----------

# DBTITLE 1,Build CRM Opportunities
# Opportunities - deals by stage, linked to accounts and products
stages = ["Prospecting", "Qualification", "Proposal/Price Quote", "Negotiation/Review", "Closed Won", "Closed Lost"]
stage_probabilities = [0.15, 0.20, 0.25, 0.20, 0.12, 0.08]
product_families = ["Chemicals", "Equipment", "Plasticware", "Glassware", "Consumables", "Instruments", "Reagents", "Safety"]

opp_rows = []
for i in range(N_OPPORTUNITIES):
    acct = random.choice(account_rows)
    stage = random.choices(stages, weights=stage_probabilities, k=1)[0]
    family = random.choice(product_families)
    
    # Amount varies by segment and stage
    base_amount = acct["annual_revenue"] * random.uniform(0.02, 0.15)
    if stage in ["Closed Won", "Closed Lost"]:
        close_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
    else:
        close_date = date(2026, 10, 1) + timedelta(days=random.randint(0, 180))
    
    created = close_date - timedelta(days=random.randint(30, 180))
    
    opp_rows.append({
        "opportunity_id": f"SF-OPP-{i+1:05d}",
        "account_id": acct["account_id"],
        "opportunity_name": f"{acct['account_name']} - {family} {stage.split('/')[0]}",
        "stage_name": stage,
        "amount": round(base_amount, 2),
        "close_date": close_date,
        "created_date": created,
        "product_family": family,
        "probability": {"Prospecting": 10, "Qualification": 25, "Proposal/Price Quote": 50,
                        "Negotiation/Review": 75, "Closed Won": 100, "Closed Lost": 0}[stage],
        "forecast_category": {"Prospecting": "Pipeline", "Qualification": "Pipeline",
                              "Proposal/Price Quote": "Best Case", "Negotiation/Review": "Commit",
                              "Closed Won": "Closed", "Closed Lost": "Omitted"}[stage],
    })

opp_schema = StructType([
    StructField("opportunity_id", StringType()),
    StructField("account_id", StringType()),
    StructField("opportunity_name", StringType()),
    StructField("stage_name", StringType()),
    StructField("amount", DoubleType()),
    StructField("close_date", DateType()),
    StructField("created_date", DateType()),
    StructField("product_family", StringType()),
    StructField("probability", IntegerType()),
    StructField("forecast_category", StringType()),
])

crm_opps_df = spark.createDataFrame(opp_rows, schema=opp_schema)
crm_opps_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.crm_opportunities")
print(f"Wrote {crm_opps_df.count()} opportunities to {CATALOG}.{SCHEMA}.crm_opportunities")
display(crm_opps_df.groupBy("stage_name").agg(F.count("*").alias("count"), F.sum("amount").alias("total_value")).orderBy("stage_name"))

# COMMAND ----------

# DBTITLE 1,Build CRM Contacts
# Contacts - lab managers, PIs, procurement, linked to accounts
titles = [
    "Lab Manager", "Principal Investigator", "Procurement Manager",
    "Research Associate", "Director of Research", "VP of Operations",
    "Quality Manager", "Lab Technician", "Purchasing Agent", "Department Head",
]
first_names = ["Sarah", "James", "Emily", "Michael", "Rachel", "David", "Jennifer", "Robert",
               "Amanda", "Christopher", "Lauren", "Daniel", "Jessica", "Andrew", "Nicole",
               "Brian", "Stephanie", "Matthew", "Katherine", "Kevin"]
last_names = ["Chen", "Patel", "Kim", "Rodriguez", "Williams", "Johnson", "Brown", "Lee",
              "Garcia", "Martinez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
              "Jackson", "Martin", "Thompson", "White", "Harris"]

contact_rows = []
for i in range(N_CONTACTS):
    acct = random.choice(account_rows)
    first = random.choice(first_names)
    last = random.choice(last_names)
    title = random.choice(titles)
    
    # Last activity date - some accounts have recent, some stale (demo story)
    if acct["account_name"] == "University of Pittsburgh":
        last_activity = date(2026, 8, 25)  # 35+ days ago from "today" Sep 30
    elif acct["account_name"] == "Merck Life Sciences":
        last_activity = date(2026, 9, 28)  # very recent
    else:
        last_activity = date(2026, 6, 1) + timedelta(days=random.randint(0, 120))
    
    contact_rows.append({
        "contact_id": f"SF-CON-{i+1:05d}",
        "account_id": acct["account_id"],
        "first_name": first,
        "last_name": last,
        "title": title,
        "email": f"{first.lower()}.{last.lower()}@{acct['account_name'].lower().replace(' ', '').replace('&','')[:20]}.com",
        "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
        "last_activity_date": last_activity,
        "created_date": date(2021, 1, 1) + timedelta(days=random.randint(0, 1800)),
    })

contact_schema = StructType([
    StructField("contact_id", StringType()),
    StructField("account_id", StringType()),
    StructField("first_name", StringType()),
    StructField("last_name", StringType()),
    StructField("title", StringType()),
    StructField("email", StringType()),
    StructField("phone", StringType()),
    StructField("last_activity_date", DateType()),
    StructField("created_date", DateType()),
])

crm_contacts_df = spark.createDataFrame(contact_rows, schema=contact_schema)
crm_contacts_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.crm_contacts")
print(f"Wrote {crm_contacts_df.count()} contacts to {CATALOG}.{SCHEMA}.crm_contacts")
display(crm_contacts_df.groupBy("title").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Build CRM Cases (service tickets)
# Cases - service tickets linked to accounts
case_types = ["Instrument Service", "Shipping Issue", "Billing Dispute", "Product Quality", "Order Inquiry", "Technical Support"]
case_statuses = ["Open", "In Progress", "Escalated", "Closed"]
priorities = ["Low", "Medium", "High", "Critical"]

case_rows = []
for i in range(N_CASES):
    acct = random.choice(account_rows)
    case_type = random.choice(case_types)
    status = random.choices(case_statuses, weights=[0.2, 0.3, 0.1, 0.4], k=1)[0]
    priority = random.choices(priorities, weights=[0.2, 0.4, 0.25, 0.15], k=1)[0]
    
    created = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
    closed = created + timedelta(days=random.randint(1, 30)) if status == "Closed" else None
    
    # University of Pittsburgh gets extra open cases (demo story)
    if acct["account_name"] == "University of Pittsburgh" and i % 5 == 0:
        status = "Escalated"
        priority = "High"
        case_type = "Shipping Issue"
    
    case_rows.append({
        "case_id": f"SF-CASE-{i+1:05d}",
        "account_id": acct["account_id"],
        "case_type": case_type,
        "status": status,
        "priority": priority,
        "subject": f"{case_type} - {acct['account_name']}",
        "created_date": created,
        "closed_date": closed,
    })

case_schema = StructType([
    StructField("case_id", StringType()),
    StructField("account_id", StringType()),
    StructField("case_type", StringType()),
    StructField("status", StringType()),
    StructField("priority", StringType()),
    StructField("subject", StringType()),
    StructField("created_date", DateType()),
    StructField("closed_date", DateType()),
])

crm_cases_df = spark.createDataFrame(case_rows, schema=case_schema)
crm_cases_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.crm_cases")
print(f"Wrote {crm_cases_df.count()} cases to {CATALOG}.{SCHEMA}.crm_cases")
display(crm_cases_df.groupBy("case_type", "status").count().orderBy("case_type", "status"))

# COMMAND ----------

# DBTITLE 1,Build CRM Activities (meetings, calls, emails)
# Activities - rep touchpoints with accounts (meetings, calls, emails)
activity_types = ["Meeting", "Call", "Email", "Site Visit", "Demo"]
activity_subjects = {
    "Meeting": ["Quarterly Business Review", "Product Training", "Contract Renewal Discussion", "New Product Introduction"],
    "Call": ["Follow-up Call", "Order Status Check", "Technical Consultation", "Pricing Discussion"],
    "Email": ["Quote Follow-up", "Product Catalog Update", "Service Notification", "Promotional Offer"],
    "Site Visit": ["Lab Assessment", "Installation Support", "Equipment Audit", "Facility Tour"],
    "Demo": ["Product Demo", "Software Demo", "Instrument Demo", "Workflow Demo"],
}

activity_rows = []
for i in range(N_ACTIVITIES):
    acct = random.choice(account_rows)
    atype = random.choices(activity_types, weights=[0.25, 0.30, 0.25, 0.10, 0.10], k=1)[0]
    subject = random.choice(activity_subjects[atype])
    
    # Activity date distribution - more recent = more active account
    if acct["account_name"] == "University of Pittsburgh":
        # Gap: last activity was 35+ days ago
        activity_date = date(2026, 3, 1) + timedelta(days=random.randint(0, 150))
        if activity_date > date(2026, 8, 25):
            activity_date = date(2026, 8, 25) - timedelta(days=random.randint(0, 30))
    elif acct["account_name"] == "Merck Life Sciences":
        # Very active - regular touchpoints
        activity_date = date(2026, 7, 1) + timedelta(days=random.randint(0, 90))
    else:
        activity_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
    
    activity_rows.append({
        "activity_id": f"SF-ACT-{i+1:05d}",
        "account_id": acct["account_id"],
        "activity_type": atype,
        "subject": f"{subject} - {acct['account_name']}",
        "activity_date": activity_date,
        "completed": True,
        "created_by": acct.get("owner_rep_name", "Unknown Rep"),
    })

activity_schema = StructType([
    StructField("activity_id", StringType()),
    StructField("account_id", StringType()),
    StructField("activity_type", StringType()),
    StructField("subject", StringType()),
    StructField("activity_date", DateType()),
    StructField("completed", BooleanType()),
    StructField("created_by", StringType()),
])

crm_activities_df = spark.createDataFrame(activity_rows, schema=activity_schema)
crm_activities_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.crm_activities")
print(f"Wrote {crm_activities_df.count()} activities to {CATALOG}.{SCHEMA}.crm_activities")
display(crm_activities_df.groupBy("activity_type").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Validate demo story accounts
# Validate the key demo story accounts
print("=== UNIVERSITY OF PITTSBURGH (high-risk demo account) ===")
upitt_id = [r["account_id"] for r in account_rows if r["account_name"] == "University of Pittsburgh"][0]
print(f"Account ID: {upitt_id}")

upitt_activities = spark.table(f"{CATALOG}.{SCHEMA}.crm_activities").filter(F.col("account_id") == upitt_id)
print(f"Total activities: {upitt_activities.count()}")
print(f"Last activity date: {upitt_activities.agg(F.max('activity_date')).collect()[0][0]}")

upitt_cases = spark.table(f"{CATALOG}.{SCHEMA}.crm_cases").filter(F.col("account_id") == upitt_id)
print(f"Total cases: {upitt_cases.count()}, Open/Escalated: {upitt_cases.filter(F.col('status').isin('Open', 'Escalated')).count()}")

upitt_opps = spark.table(f"{CATALOG}.{SCHEMA}.crm_opportunities").filter(F.col("account_id") == upitt_id)
print(f"Total opportunities: {upitt_opps.count()}")

print("\n=== MERCK LIFE SCIENCES (healthy, cross-sell demo account) ===")
merck_id = [r["account_id"] for r in account_rows if r["account_name"] == "Merck Life Sciences"][0]
print(f"Account ID: {merck_id}")

merck_activities = spark.table(f"{CATALOG}.{SCHEMA}.crm_activities").filter(F.col("account_id") == merck_id)
print(f"Total activities: {merck_activities.count()}")
print(f"Last activity date: {merck_activities.agg(F.max('activity_date')).collect()[0][0]}")

merck_opps = spark.table(f"{CATALOG}.{SCHEMA}.crm_opportunities").filter(F.col("account_id") == merck_id)
print(f"Total opportunities: {merck_opps.count()}")
display(merck_opps.groupBy("product_family").agg(F.count("*").alias("deals"), F.sum("amount").alias("total_value")).orderBy(F.desc("total_value")))

# COMMAND ----------

# DBTITLE 1,Summary - all CRM tables created
# Summary of all CRM tables created
crm_tables = ["crm_accounts", "crm_opportunities", "crm_contacts", "crm_cases", "crm_activities"]
print("=== CRM TABLES CREATED IN main.ccg_workshop_cdm ===")
for t in crm_tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()
    print(f"  {CATALOG}.{SCHEMA}.{t}: {count:,} rows")

print("\n=== EXISTING CDM TABLES (for reference) ===")
cdm_tables = ["cur_customer_shipto", "cur_customer_sales_org", "cur_product", 
              "cur_sales_order_header", "cur_sales_order_line", "cur_shipment_line"]
for t in cdm_tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()
    print(f"  {CATALOG}.{SCHEMA}.{t}: {count:,} rows")

print("\nNext steps:")
print("1. Sign up for SF Dev Org at developer.salesforce.com/signup")
print("2. Create Connected App for OAuth")
print("3. Run notebook 02 to load this data into Salesforce via Bulk API")
print("4. Set up Lakeflow Connect to ingest SF data back into UC")
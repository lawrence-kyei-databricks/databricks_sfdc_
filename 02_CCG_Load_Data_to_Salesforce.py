# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Load CCG CRM Data into Salesforce Developer Org
# MAGIC **Purpose:** Push synthetic CRM data from `main.ccg_workshop_cdm.crm_*` tables into Salesforce standard objects (Account, Opportunity, Contact, Case, Task). After loading, we will:
# MAGIC 1. Drop the `crm_*` tables from UC
# MAGIC 2. Set up Lakeflow Connect to ingest the SF data back - the real demo flow
# MAGIC
# MAGIC **Auth:** Username + Password + Security Token (no Connected App needed)
# MAGIC
# MAGIC **SF Standard Objects used:** Account, Opportunity, Contact, Case, Task

# COMMAND ----------

# DBTITLE 1,Install simple-salesforce
# MAGIC %pip install simple-salesforce
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configure credentials - add Connected App consumer key/secret
# Paste your Salesforce Developer Org credentials below
# Replace the empty strings with your actual values

SF_USERNAME = "lawrence.kyei.2e2f60a0b23e@agentforce.com"       # e.g. "you@example.com"
SF_PASSWORD = "Hustl995!"       # your SF password
SF_TOKEN = "BfD66HTR9qsl0kOXwebyUpjYm"          # security token from the email SF sent you

assert SF_USERNAME, "Paste your SF username above"
assert SF_PASSWORD, "Paste your SF password above"
SF_CONSUMER_KEY = dbutils.secrets.get(scope="ccg-salesforce", key="consumer-key")        # from Connected App > Manage Consumer Details
SF_CONSUMER_SECRET = dbutils.secrets.get(scope="ccg-salesforce", key="consumer-secret")     # from Connected App > Manage Consumer Details

assert SF_CONSUMER_KEY, "Paste your Connected App consumer key above"

print(f"Credentials set for: {SF_USERNAME}")

# COMMAND ----------

# DBTITLE 1,Connect to Salesforce via Client Credentials Flow
import requests
from simple_salesforce import Salesforce

# Client Credentials Flow - uses Consumer Key/Secret from External Client App
auth_response = requests.post(
    "https://orgfarm-3843d545d3-dev-ed.develop.my.salesforce.com/services/oauth2/token",
    data={
        "grant_type": "client_credentials",
        "client_id": SF_CONSUMER_KEY,
        "client_secret": SF_CONSUMER_SECRET,

    }
)
print(f"Status: {auth_response.status_code}")
print(f"Response: {auth_response.text[:300]}")
auth_response.raise_for_status()
auth = auth_response.json()

sf = Salesforce(instance_url=auth["instance_url"], session_id=auth["access_token"])
print(f"\nConnected to: {sf.sf_instance}")
print(f"API version: {sf.sf_version}")

# COMMAND ----------

# DBTITLE 1,Clean up any existing demo data in SF (run if needed)
# Delete any existing demo data from previous imports
# Run this cell ONLY if you need to clean up

objects_to_clean = ["Task", "Case", "Opportunity", "Contact", "Account"]
# Order matters: delete children before parents

for obj in objects_to_clean:
    try:
        result = sf.query_all(f"SELECT Id FROM {obj}")
        ids = [r["Id"] for r in result["records"]]
        if ids:
            print(f"Deleting {len(ids)} {obj} records...")
            delete_results = getattr(sf.bulk, obj).delete([{"Id": i} for i in ids], batch_size=200)
            success = sum(1 for r in delete_results if r["success"])
            print(f"  Deleted {success}/{len(ids)}")
        else:
            print(f"No {obj} records to delete")
    except Exception as e:
        print(f"  {obj}: {e}")

print("\nCleanup complete.")

# COMMAND ----------

# DBTITLE 1,Load CRM data from UC
CATALOG = "main"
SCHEMA = "ccg_workshop_cdm"

# Load all CRM tables
accounts_pdf = spark.table(f"{CATALOG}.{SCHEMA}.crm_accounts").toPandas()
opps_pdf = spark.table(f"{CATALOG}.{SCHEMA}.crm_opportunities").toPandas()
contacts_pdf = spark.table(f"{CATALOG}.{SCHEMA}.crm_contacts").toPandas()
cases_pdf = spark.table(f"{CATALOG}.{SCHEMA}.crm_cases").toPandas()
activities_pdf = spark.table(f"{CATALOG}.{SCHEMA}.crm_activities").toPandas()

print(f"Accounts: {len(accounts_pdf)}")
print(f"Opportunities: {len(opps_pdf)}")
print(f"Contacts: {len(contacts_pdf)}")
print(f"Cases: {len(cases_pdf)}")
print(f"Activities: {len(activities_pdf)}")

# COMMAND ----------

# DBTITLE 1,Step 1 - Create Accounts in Salesforce
# Push Accounts to SF - map our fields to SF standard fields
# We use External_Id__c pattern: store our account_id so we can link back later

# First, check if we need to create the custom External ID field
# For standard Account, we'll use the Name + Description to store our key

account_records = []
for _, row in accounts_pdf.iterrows():
    rec = {
        "Name": row["account_name"],
        "Industry": row["industry_segment"],
        "AnnualRevenue": row["annual_revenue"],
        "Description": f"CDM Key: {row['customer_shipto_key']} | Demo Account ID: {row['account_id']}",
        "Type": "Customer",
        "AccountSource": "Databricks Demo",
    }
    account_records.append(rec)

print(f"Pushing {len(account_records)} accounts to Salesforce...")
results = sf.bulk.Account.insert(account_records, batch_size=200)
success = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])
print(f"Done - {success} success, {failed} failed")

if failed:
    for r in results:
        if not r["success"]:
            print(f"  Error: {r['errors']}")

# COMMAND ----------

# DBTITLE 1,Build Account ID mapping (SF ID <-> our account_id)
# Build mapping: our account_id -> Salesforce record Id
# Query SF for all accounts we just created
import time
time.sleep(5)  # give SF a moment to index

result = sf.query_all("SELECT Id, Name, Description FROM Account WHERE AccountSource = 'Databricks Demo'")
sf_accounts = result["records"]
print(f"Found {len(sf_accounts)} accounts in Salesforce")

# Parse our account_id from Description field
account_id_map = {}  # our account_id -> SF Id
for rec in sf_accounts:
    desc = rec.get("Description", "") or ""
    if "Demo Account ID:" in desc:
        our_id = desc.split("Demo Account ID:")[1].strip()
        account_id_map[our_id] = rec["Id"]

print(f"Mapped {len(account_id_map)} accounts")

# Verify demo story accounts
for name in ["University of Pittsburgh", "Merck Life Sciences"]:
    match = [r for r in sf_accounts if r["Name"] == name]
    if match:
        print(f"  {name}: SF ID = {match[0]['Id']}")

# COMMAND ----------

# DBTITLE 1,Step 2 - Create Contacts in Salesforce
# Push Contacts - requires AccountId from our mapping
contact_records = []
skipped = 0
for _, row in contacts_pdf.iterrows():
    sf_acct_id = account_id_map.get(row["account_id"])
    if not sf_acct_id:
        skipped += 1
        continue
    rec = {
        "AccountId": sf_acct_id,
        "FirstName": row["first_name"],
        "LastName": row["last_name"],
        "Title": row["title"],
        "Email": row["email"],
        "Phone": row["phone"],
    }
    contact_records.append(rec)

print(f"Pushing {len(contact_records)} contacts (skipped {skipped} - no account match)...")
results = sf.bulk.Contact.insert(contact_records, batch_size=200)
success = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])
print(f"Done - {success} success, {failed} failed")
if failed > 0:
    errors = [r for r in results if not r["success"]][:3]
    for e in errors:
        print(f"  Error: {e['errors']}")

# COMMAND ----------

# DBTITLE 1,Step 3 - Create Opportunities in Salesforce
# Push Opportunities - requires AccountId + CloseDate + StageName + Name
opp_records = []
skipped = 0
for _, row in opps_pdf.iterrows():
    sf_acct_id = account_id_map.get(row["account_id"])
    if not sf_acct_id:
        skipped += 1
        continue
    rec = {
        "AccountId": sf_acct_id,
        "Name": row["opportunity_name"][:120],  # SF max 120 chars
        "StageName": row["stage_name"],
        "Amount": row["amount"],
        "CloseDate": str(row["close_date"]),
        "Probability": row["probability"],
        "ForecastCategoryName": row["forecast_category"],
        "Description": f"Product Family: {row['product_family']}",
    }
    opp_records.append(rec)

print(f"Pushing {len(opp_records)} opportunities (skipped {skipped})...")
results = sf.bulk.Opportunity.insert(opp_records, batch_size=200)
success = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])
print(f"Done - {success} success, {failed} failed")
if failed > 0:
    errors = [r for r in results if not r["success"]][:3]
    for e in errors:
        print(f"  Error: {e['errors']}")

# COMMAND ----------

# DBTITLE 1,Step 4 - Create Cases in Salesforce
# Push Cases - requires AccountId
case_records = []
skipped = 0
for _, row in cases_pdf.iterrows():
    sf_acct_id = account_id_map.get(row["account_id"])
    if not sf_acct_id:
        skipped += 1
        continue
    rec = {
        "AccountId": sf_acct_id,
        "Subject": row["subject"][:255],
        "Status": "New" if row["status"] == "Open" else row["status"] if row["status"] in ["Escalated", "Closed"] else "Working",
        "Priority": row["priority"],
        "Type": row["case_type"],
        "Origin": "Databricks Demo",
    }
    case_records.append(rec)

print(f"Pushing {len(case_records)} cases (skipped {skipped})...")
results = sf.bulk.Case.insert(case_records, batch_size=200)
success = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])
print(f"Done - {success} success, {failed} failed")
if failed > 0:
    errors = [r for r in results if not r["success"]][:3]
    for e in errors:
        print(f"  Error: {e['errors']}")

# COMMAND ----------

# DBTITLE 1,Step 5 - Create Activities (Tasks) in Salesforce
# Push Activities as Tasks in Salesforce
# SF Task.WhatId = Account Id, Task.Subject, Task.ActivityDate, Task.Status
task_records = []
skipped = 0
for _, row in activities_pdf.iterrows():
    sf_acct_id = account_id_map.get(row["account_id"])
    if not sf_acct_id:
        skipped += 1
        continue
    rec = {
        "WhatId": sf_acct_id,
        "Subject": row["subject"][:255],
        "ActivityDate": str(row["activity_date"]),
        "Status": "Completed" if row["completed"] else "Not Started",
        "Priority": "Normal",
        "Description": f"Type: {row['activity_type']} | Created by: {row['created_by']}",
    }
    task_records.append(rec)

print(f"Pushing {len(task_records)} tasks/activities (skipped {skipped})...")
results = sf.bulk.Task.insert(task_records, batch_size=200)
success = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])
print(f"Done - {success} success, {failed} failed")
if failed > 0:
    errors = [r for r in results if not r["success"]][:3]
    for e in errors:
        print(f"  Error: {e['errors']}")

# COMMAND ----------

# DBTITLE 1,Verify data in Salesforce
# Verify counts in Salesforce
print("=== SALESFORCE DATA VERIFICATION ===")
for obj in ["Account", "Contact", "Opportunity", "Case", "Task"]:
    result = sf.query(f"SELECT COUNT() FROM {obj}")
    print(f"  {obj}: {result['totalSize']} records")

# Check demo story accounts
print("\n=== DEMO STORY ACCOUNTS ===")
for name in ["University of Pittsburgh", "Merck Life Sciences"]:
    result = sf.query(f"SELECT Id, Name, AnnualRevenue, Industry FROM Account WHERE Name = '{name}'")
    if result["records"]:
        r = result["records"][0]
        print(f"  {r['Name']}: Revenue=${r['AnnualRevenue']:,.0f}, Industry={r['Industry']}")

# COMMAND ----------

# DBTITLE 1,Drop CRM tables from UC (data now lives in Salesforce)
# Now that data is in Salesforce, drop the crm_* tables from UC
# Lakeflow Connect will re-ingest from SF as streaming tables
CATALOG = "main"
SCHEMA = "ccg_workshop_cdm"

crm_tables = ["crm_accounts", "crm_opportunities", "crm_contacts", "crm_cases", "crm_activities"]

print("Dropping CRM tables from UC (data is now in Salesforce)...")
for t in crm_tables:
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{t}")
    print(f"  Dropped {CATALOG}.{SCHEMA}.{t}")

print("\nDone. CRM data now lives ONLY in Salesforce.")
print("Next step: Set up Lakeflow Connect to ingest SF data back into UC.")
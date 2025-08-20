from streamlit_app.dal import Customer, dal_from_env
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timezone
import pandas as pd


def main():
    """
    Exporte d'abord tes variables :
      export DATABRICKS_HOST="adb-xxxx.azuredatabricks.net"
      export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxx"
      export DATABRICKS_TOKEN="dapiXXXXXXXX"
      export DATABRICKS_CATALOG="main"
      export DATABRICKS_SCHEMA="analytics"

    Puis exécute: python dal_orm_databricks.py
    """

    load_dotenv(find_dotenv(), override=False)

    dal = dal_from_env(init_models=True)

    # --- CREATE (ORM)
    alice = Customer(
        customer_id="C001",
        name="Alice",
        country="FR",
        updated_at=datetime.now(timezone.utc),
    )
    bob = Customer(
        customer_id="C002",
        name="Bob",
        country="ES",
        updated_at=datetime.now(timezone.utc),
    )
    dal.add_all([alice, bob])
    print("Inserted 2 customers via ORM")

    # --- READ (ORM)
    fr_customers = dal.select_where(Customer, Customer.country == "FR", limit=10)
    print("FR customers:", [c.customer_id for c in fr_customers])

    # --- UPDATE (ORM/Core)
    n_upd = dal.update_fields(
        Customer, Customer.customer_id == "C002", {"country": "DE"}
    )
    print("Updated rows:", n_upd)

    # --- UPSERT (MERGE) depuis pandas
    df_upsert = pd.DataFrame(
        [
            {
                "customer_id": "C002",
                "name": "Bob Jr",
                "country": "DE",
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "customer_id": "C003",
                "name": "Cara",
                "country": "FR",
                "updated_at": datetime.now(timezone.utc),
            },
        ]
    )
    n_merged = dal.upsert_df(
        Customer,
        df_upsert,
        match_keys=["customer_id"],
        update_when_newer_col="updated_at",
    )
    print("Merge affected rows (approx):", n_merged)

    # --- DELETE (ORM/Core)
    n_del = dal.delete_where(Customer, Customer.customer_id == "C001")
    print("Deleted rows:", n_del)

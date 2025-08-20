from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Type, TypeVar

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker
from sqlalchemy import select, func, update as sa_update, delete as sa_delete, text
from datetime import datetime

# =========================
# Base ORM
# =========================


class Base(DeclarativeBase):
    """Base ORM. Tes modèles doivent hériter de Base."""

    pass


# =========================
# Config & Engine
# =========================


@dataclass(frozen=True)
class DatabricksConfig:
    server_hostname: str  # ex: adb-xxxx.xx.azuredatabricks.net
    http_path: str  # ex: /sql/1.0/warehouses/abcd1234...
    access_token: str  # dapi********
    catalog: str  # ex: main, prod, ...
    schema: str  # ex: analytics, sales, ...


def create_databricks_engine(cfg: DatabricksConfig) -> Engine:
    """
    Engine SQLAlchemy pour Databricks SQL (dialecte officiel).
    On fixe catalog & schema dans l'URL — les tables ORM seront donc créées/visées là-bas par défaut.
    """
    # Tu peux ajouter des connect_args (user-agent, TLS, etc.) si besoin
    url = (
        f"databricks://token:{cfg.access_token}@{cfg.server_hostname}"
        f"?http_path={cfg.http_path}&catalog={cfg.catalog}&schema={cfg.schema}"
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
        future=True,
        connect_args={"user_agent_entry": "streamlit-app"},  # 👈 corrige le warning
    )
    return engine


# =========================
# Session factory & context
# =========================


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(SessionFactory: sessionmaker[Session]) -> Iterable[Session]:
    """Unit of Work: ouvre une session, commit si OK, rollback sinon."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# =========================
# Helpers table name (FQN affiché)
# =========================


def _quoted_fqn_for_model(engine: Engine, model: Type[Base]) -> str:
    """
    Construit un FQN `` `catalog`.`schema`.`table` `` pour logs/SQL bruts.
    Le catalog/schema proviennent de l'URL de l'engine. Le nom de table vient du mapper ORM.
    """
    table = model.__table__
    # Databricks SQLAlchemy prend catalog & schema depuis l'URL.
    # table.schema peut être None ici; on s'appuie donc sur l'URL.
    # On lit les paramètres de l'URL pour catalog/schema:
    url = engine.url
    catalog = url.query.get("catalog") if hasattr(url, "query") else None
    schema = url.query.get("schema") if hasattr(url, "query") else None
    # fallback si besoin
    if not catalog or not schema:
        # Essaye de découper table.schema si renseigné, sinon laisse simple
        if table.schema and "." in table.schema:
            catalog, schema = table.schema.split(".", 1)
        else:
            catalog = catalog or "main"
            schema = schema or "default"
    return f"`{catalog}`.`{schema}`.`{table.name}`"


# =========================
# DAL ORM-first
# =========================

T = TypeVar("T", bound=Base)


class OrmDAL:
    """
    DAL ORM-first pour Databricks SQL:
      - CRUD via SQLAlchemy ORM (Session)
      - UPSERT (MERGE) Delta via SQL brut encapsulé (staging + MERGE)
      - Insert/Upsert depuis pandas.DataFrame
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionFactory = session_factory(engine)

    # ---------- CREATE (ORM)

    def add(self, obj: T) -> T:
        with session_scope(self.SessionFactory) as s:
            s.add(obj)
            # flush pour obtenir d'éventuelles valeurs server-side (IDENTITY, etc.)
            s.flush()
            return obj

    def add_all(self, objs: Sequence[T]) -> None:
        if not objs:
            return
        with session_scope(self.SessionFactory) as s:
            s.add_all(objs)

    # ---------- READ (ORM)

    def get_by_pk(self, model: Type[T], pk: Any) -> Optional[T]:
        with session_scope(self.SessionFactory) as s:
            return s.get(model, pk)

    def select_where(
        self,
        model: Type[T],
        where_clause,  # SQLAlchemy expression (ex: Model.country == "FR")
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> list[T]:
        from sqlalchemy import select

        stmt = select(model).where(where_clause)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        if limit is not None:
            stmt = stmt.limit(limit)
        with session_scope(self.SessionFactory) as s:
            return list(s.scalars(stmt))

    # ---------- UPDATE (ORM)

    def _count_where(self, model: Type[T], where_clause) -> int:
        with session_scope(self.SessionFactory) as s:
            n = s.scalar(select(func.count()).select_from(model).where(where_clause))
            return int(n or 0)

    # ---------- UPDATE (ORM) avec estimation fiable sur Databricks
    def update_fields(
        self,
        model: Type[T],
        where_clause,
        set_values: Mapping[str, Any],
    ) -> int:
        """
        Met à jour via ORM/Core: UPDATE ... WHERE ...
        Retourne un décompte *estimé* (count avant) car rowcount n'est pas disponible sur Databricks.
        """
        pre = self._count_where(model, where_clause)
        if pre == 0:
            return 0
        with session_scope(self.SessionFactory) as s:
            s.execute(sa_update(model).where(where_clause).values(**set_values))
        return pre

    # ---------- DELETE (ORM) avec estimation fiable sur Databricks
    def delete_where(self, model: Type[T], where_clause) -> int:
        """
        Supprime via ORM/Core: DELETE ... WHERE ...
        Retourne un décompte *estimé* (count avant) car rowcount n'est pas disponible sur Databricks.
        """
        pre = self._count_where(model, where_clause)
        if pre == 0:
            return 0
        with session_scope(self.SessionFactory) as s:
            s.execute(sa_delete(model).where(where_clause))
        return pre

    # ---------- INSERT depuis pandas (ORM-style)
    def insert_df(self, model: Type[T], df: pd.DataFrame) -> int:
        """
        Convertit chaque ligne du DataFrame en instance ORM et les INSERT.
        Pour des runs idempotents, préfère upsert_df(...).
        """
        if df.empty:
            return 0
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        objs = [model(**rec) for rec in records]  # type: ignore[arg-type]
        self.add_all(objs)
        return len(objs)

    # ---------- UPSERT (MERGE) Delta via staging + métriques Delta
    def upsert_df(
        self,
        model: Type[T],
        df: pd.DataFrame,
        *,
        match_keys: Sequence[str],
        update_when_newer_col: Optional[str] = None,
        columns_to_update: Optional[Sequence[str]] = None,
        create_missing: bool = True,
        staging_prefix: str = "_stg_",
    ) -> int:
        """
        UPSERT robuste (MERGE Delta) depuis pandas avec staging.
        Retourne un décompte basé sur DESCRIBE HISTORY (insert+update+delete).
        """
        import json  # local pour éviter d'ajouter un import global

        if df.empty:
            return 0
        for k in match_keys:
            if k not in df.columns:
                raise ValueError(f"match key '{k}' not present in DataFrame")

        if columns_to_update is None:
            columns_to_update = [c for c in df.columns if c not in set(match_keys)]

        # Noms de table fully qualified (catalog.schema.table)
        target_fqn = _quoted_fqn_for_model(self.engine, model)

        # staging dans le même catalog/schema
        staging_name = f"{staging_prefix}{uuid.uuid4().hex[:8]}"
        # Compose FQN depuis engine URL
        url = self.engine.url
        catalog = url.query.get("catalog") if hasattr(url, "query") else None
        schema = url.query.get("schema") if hasattr(url, "query") else None
        if not (catalog and schema):
            table_schema = model.__table__.schema or "main.default"
            if "." in table_schema:
                catalog, schema = table_schema.split(".", 1)
            else:
                catalog = catalog or "main"
                schema = schema or "default"
        staging_fqn = f"`{catalog}`.`{schema}`.`{staging_name}`"

        # Clauses MERGE
        on_pred = " AND ".join([f"t.`{k}` = s.`{k}`" for k in match_keys])
        when_matched_pred = ""
        if update_when_newer_col:
            when_matched_pred = (
                f" AND s.`{update_when_newer_col}` > t.`{update_when_newer_col}`"
            )
        set_clause = ", ".join([f"`{c}` = s.`{c}`" for c in columns_to_update])
        insert_cols_sql = ", ".join([f"`{c}`" for c in df.columns])
        insert_vals_sql = ", ".join([f"s.`{c}`" for c in df.columns])

        merge_sql = f"""
        MERGE INTO {target_fqn} AS t
        USING {staging_fqn} AS s
        ON {on_pred}
        WHEN MATCHED{when_matched_pred} THEN
          UPDATE SET {set_clause}
        {"WHEN NOT MATCHED THEN INSERT (" + insert_cols_sql + ") VALUES (" + insert_vals_sql + ")" if create_missing else ""}
        """

        # Exécution (CREATE LIKE -> INSERT -> MERGE -> DROP)
        with self.engine.begin() as conn:
            # 1) staging LIKE target
            conn.exec_driver_sql(f"CREATE TABLE {staging_fqn} LIKE {target_fqn}")

            # 2) insert DataFrame dans staging (executemany paramétré)
            cols = list(df.columns)
            records = df.where(pd.notnull(df), None).to_dict(orient="records")
            if records:
                placeholders = ", ".join([f":{c}" for c in cols])
                col_sql = ", ".join([f"`{c}`" for c in cols])
                insert_sql = text(
                    f"INSERT INTO {staging_fqn} ({col_sql}) VALUES ({placeholders})"
                )
                conn.execute(insert_sql, records)

            # 3) MERGE
            conn.exec_driver_sql(merge_sql)

            # 4) cleanup staging
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {staging_fqn}")

        # Récupère les métriques de la dernière opération MERGE
        # DESCRIBE HISTORY renvoie operationMetrics (JSON) avec numTargetRowsInserted/Updated/Deleted
        try:
            with self.engine.begin() as conn:
                row = (
                    conn.exec_driver_sql(f"DESCRIBE HISTORY {target_fqn} LIMIT 1")
                    .mappings()
                    .first()
                )
            if row:
                opm = row.get("operationMetrics")
                if isinstance(opm, str):
                    metrics = json.loads(opm)
                elif isinstance(opm, dict):
                    metrics = opm
                else:
                    metrics = {}

                # Clés courantes sur Delta
                inserted = int(metrics.get("numTargetRowsInserted", 0))
                updated = int(metrics.get("numTargetRowsUpdated", 0))
                deleted = int(metrics.get("numTargetRowsDeleted", 0))

                total = inserted + updated + deleted
                if total == 0:
                    # fallback sur d'autres clés parfois présentes
                    total = int(metrics.get("numOutputRows", 0))

                return total
        except Exception:
            # si l'historique n'est pas dispo, on garde la compat par défaut
            pass

        # Fallback si métriques non disponibles
        return -1


# =========================
# Modèle d'exemple (tu peux le supprimer/adapter)
# =========================


class Customer(Base):
    __tablename__ = "customers_demo"
    customer_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]]
    country: Mapped[Optional[str]]
    updated_at: Mapped[Optional[datetime]]


# =========================
# Bootstrap pratique
# =========================


def dal_from_env(init_models: bool = True) -> OrmDAL:
    """
    Construit un DAL à partir des variables d'env :
      - DATABRICKS_HOST
      - DATABRICKS_HTTP_PATH
      - DATABRICKS_TOKEN
      - DATABRICKS_CATALOG
      - DATABRICKS_SCHEMA
    """
    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")
    catalog = os.getenv("DATABRICKS_CATALOG")
    schema = os.getenv("DATABRICKS_SCHEMA")
    missing = [
        k
        for k, v in {
            "DATABRICKS_HOST": host,
            "DATABRICKS_HTTP_PATH": http_path,
            "DATABRICKS_TOKEN": token,
            "DATABRICKS_CATALOG": catalog,
            "DATABRICKS_SCHEMA": schema,
        }.items()
        if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    cfg = DatabricksConfig(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
        catalog=catalog,
        schema=schema,
    )
    engine = create_databricks_engine(cfg)

    # Crée les tables mappées si demandé (ex: Customer)
    if init_models:
        Base.metadata.create_all(engine)

    return OrmDAL(engine)

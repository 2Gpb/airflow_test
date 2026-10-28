import pendulum
from airflow.sdk import dag, task, Param, BaseHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from typing import Any
import requests

from batch_common import *

@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "business_date": Param("2026-09-01", type="string", format="date"), 
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000)
    },
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(seconds=5),
        "execution_timeout": pendulum.duration(minutes=2),
    },
    tags=["homework", "basic"],
)
def hw_basic_batch_inference():
    @task
    def extract(**context: Any):
        day = business_date(
            context["params"]["business_date"], 
            context.get("data_interval_start")
        )

        host = BaseHook.get_connection("source_api").host
        data_response = requests.get(f"{host.rstrip('/')}/basic/{day}.json", timeout=10)
        data_response.raise_for_status()
        
        data = data_response.json()
        if data["business_date"] != day:
            raise ValueError

        model_response = requests.get(f"{host.rstrip('/')}/models/v1.json", timeout=10)
        model_response.raise_for_status()
        
        model_sha256 = digest(model_response.content)
        model = model_response.json()

        artifacts= {
            "business_date": day,
            "rows": data["rows"],
            "model": model,
            "model_sha256": model_sha256,
        }
        
        return save(artifacts)

    @task(retries=0)
    def infer(ref: dict, **context: Any):
        batch_size = context["params"]["batch_size"]
        artifacts = read(ref)

        rows = artifacts["rows"]
        day = artifacts["business_date"]
        model = artifacts["model"]
        model_sha256 = artifacts["model_sha256"]

        rows = validate_rows(rows, day)
        expected_ids = [row["customer_id"] for row in rows]
        predictions = infer_batches(rows, model, batch_size)
        
        validate_predictions(predictions, expected_ids)

        artifacts= {
            "business_date": day,
            "model_version": "v1",
            "predictions": predictions, 
            "model_sha256": model_sha256,
        }
        
        return save(artifacts)

    @task
    def publish(ref: dict):
        artifacts = read(ref)
        model_version = artifacts["model_version"]
        predictions = artifacts["predictions"]
        day = artifacts["business_date"]
        
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try: 
            replace_day(conn, day, model_version, predictions)
        finally:
            conn.close()

        return {
            "business_date": day,
            "model_version": model_version,
            "rows": len(predictions)
        }

    publish(infer(extract()))

hw_basic_batch_inference() 
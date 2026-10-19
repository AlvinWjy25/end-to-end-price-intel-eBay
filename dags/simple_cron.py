import sys
from pathlib import Path
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.ingest import main


def check_status():
    print(f"ROOT_DIR: {ROOT_DIR}")
    print("DAG Airflow successfully executed.")


with DAG(
    dag_id="simple_cron",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Jakarta"),
    schedule="1 8 * * *",
    catchup=False,
    tags=["belajar"],
) as dag:

    check_health = PythonOperator(
        task_id="check_health",
        python_callable=check_status,
    )

    ingestion = PythonOperator(
        task_id="ingestion",
        python_callable=main,
    )

    check_health >> ingestion
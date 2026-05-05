from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="moex_candles_loader",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    run_script = BashOperator(
        task_id="run_candles_loader",
        bash_command="python /opt/airflow/dags/scripts/candles_loader.py",
    )
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="moex_candles_loader",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    run_candles_loader = BashOperator(
        task_id="run_candles_loader",
        bash_command="python /opt/airflow/dags/scripts/candles_loader.py",
    )

    run_transfer = BashOperator(
        task_id="run_transfer",
        bash_command="python /opt/airflow/dags/scripts/transfer.py",
    )

    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command="python /opt/airflow/dags/scripts/feature_engineering_loader.py",
    )

    run_feature_ML = BashOperator(
        task_id="run_feature_ML",
        bash_command="python /opt/airflow/dags/scripts/gazp_ml_v2.py",
    )

    run_dashboard_create = BashOperator(
        task_id="run_dashboard_create",
        bash_command="python /opt/airflow/dags/scripts/create_superset_view.py",
    )

    run_candles_loader >> run_transfer >> run_feature_engineering >> run_feature_ML >> run_dashboard_create
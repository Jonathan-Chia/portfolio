import os

import iris_model_pipeline
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

default_dag_args = {"owner": "airflow", "retries": 2, "start_date": days_ago(1)}

dag = DAG(
    dag_id="iris_model_pipeline_dag",
    schedule_interval="*/15 * * * *",
    max_active_runs=1,
    catchup=False,
    default_args=default_dag_args,
)


iris_elasticnet_model = PythonOperator(
    dag=dag,
    task_id="iris_elasticnet_model_task",
    python_callable=iris_model_pipeline.iris_elasticnet_model,
)

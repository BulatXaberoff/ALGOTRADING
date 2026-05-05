"""
DAG для автоматического запуска Python-скрипта по расписанию.

📁 Расположение: <ваш_проект>/dags/schedule_dag.py
🔄 Airflow автоматически подхватит файл в течение 30-60 секунд.
"""

# =============================================================================
# 🔧 ИМПОРТЫ (обязательные для Airflow)
# =============================================================================
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException

# =============================================================================
# ⚙️ КОНФИГУРАЦИЯ DAG
# =============================================================================
@dag(
    # 🕒 Расписание: @daily, @hourly, cron-строка или timedelta
    schedule="@daily",  # Каждый день в 00:00 UTC
    
    # 📅 Точка старта: должна быть в прошлом, фиксированная (не datetime.now()!)
    start_date=datetime(2024, 1, 1),
    
    # 🚫 Не запускать задачи за пропущенные интервалы при первом запуске
    catchup=False,
    
    # 🏷 Теги для фильтрации в UI
    tags=["automation", "python", "diploma"],
    
    # 📝 Описание в интерфейсе
    doc_md="""
    ### Автоматизация Python-скрипта
    * Запускает my_script.py по расписанию
    * Логи доступны в UI Airflow
    * Повтор при ошибке: 2 раза с интервалом 1 мин
    """,
    
    # ⏱ Таймаут выполнения всего DAG
    dagrun_timeout=timedelta(hours=2),
    
    # 🔄 Политики повторных попыток по умолчанию
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "email_on_failure": False,  # Настройте, если нужна почта
        "email_on_retry": False,
    },
)
def python_scheduler_dag():
    """
    Основная функция DAG.
    Все задачи определяются внутри через @task или через Operators.
    """
    
    # ==========================================================================
    # 🎯 ВАРИАНТ 1: Логика прямо внутри @task (рекомендуется для простоты)
    # ==========================================================================
    @task(
        task_id="run_python_logic",
        # Можно переопределить retries для конкретной задачи
        retries=3,
        retry_delay=timedelta(seconds=30),
    )
    def run_my_logic():
        """
        Здесь размещается ваш код или вызов функций из my_script.py.
        Все импорты сторонних библиотек — СТРОГО внутри функции!
        """
        # ── ИМПОРТЫ ВНУТРИ ФУНКЦИИ (критично для Airflow!) ──
        import sys
        import os
        
        # Пример: если используете pandas, requests и т.д.
        # import pandas as pd
        # import requests
        # from bs4 import BeautifulSoup
        
        # ── ВАШ КОД ──
        print("🚀 Запуск задачи...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        # Пример: импорт и запуск вашей функции из my_script.py
        # from my_script import main
        # result = main()
        
        # Пример: простая логика
        data = {"status": "success", "timestamp": datetime.now().isoformat()}
        print(f"✅ Результат: {data}")
        
        # Возврат значения (можно передать в следующую задачу через XCom)
        return data
    
    # ==========================================================================
    # 🎯 ВАРИАНТ 2: Запуск внешнего файла через subprocess
    # (если my_script.py — автономный скрипт с аргументами)
    # ==========================================================================
    @task(task_id="run_external_script")
    def run_external_script():
        """
        Запускает отдельный файл my_script.py как subprocess.
        Удобно, если скрипт уже готов и не хотите менять его структуру.
        """
        import subprocess
        import sys
        import os
        
        # Путь к скрипту ВНУТРИ контейнера Airflow
        script_path = "/opt/airflow/dags/my_script.py"
        
        # Проверка существования файла
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Скрипт не найден: {script_path}")
        
        try:
            # Запуск скрипта
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True,  # Выбросит CalledProcessError при ошибке
                timeout=3600,  # Таймаут 1 час
                cwd="/opt/airflow/dags",  # Рабочая директория
            )
            
            # Логирование вывода
            print("📤 STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("⚠️ STDERR:")
                print(result.stderr)
                
            return {"stdout": result.stdout, "returncode": result.returncode}
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка выполнения скрипта (код {e.returncode}):")
            print(e.stderr)
            raise AirflowException(f"Скрипт завершился с ошибкой: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise AirflowException("Скрипт не уложился в таймаут (1 час)")
    
    # ==========================================================================
    # 🔗 ЦЕПОЧКА ЗАДАЧ (если задач несколько)
    # ==========================================================================
    # Запустите одну из задач, закомментировав лишнюю:
    
    run_my_logic()  # ← Вариант 1: логика внутри DAG
    # run_external_script()  # ← Вариант 2: запуск внешнего файла
    
    # Пример цепочки: задача_1 >> задача_2 >> задача_3
    # Если задач несколько, используйте оператор >> для задания порядка:
    # task_a = task_one()
    # task_b = task_two()
    # task_a >> task_b  # task_b выполнится после task_a


# =============================================================================
# 🚀 ИНИЦИАЛИЗАЦИЯ (обязательно для декораторного стиля!)
# =============================================================================
# Создаём экземпляр DAG для регистрации в Airflow
dag = python_scheduler_dag()
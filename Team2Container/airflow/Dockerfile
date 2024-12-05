# base.Dockerfile
FROM my-base-image  

ENV AIRFLOW_HOME=/usr/local/airflow

# Install Python dependencies
RUN pip install apache-airflow apache-airflow-providers-slack
RUN pip install JPype1 konlpy
RUN pip install mlflow

# Create Airflow home directory and set working directory
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME

# Airflow 데이터베이스 초기화
RUN airflow db init

# Airflow 관리자 계정 생성
RUN airflow users create \  
    --username team2 \
    --firstname team2 \
    --lastname team2 \
    --role Admin \
    --email team2@example.com \
    --password team2

# dags 폴더 복사
COPY dags/ $AIRFLOW_HOME/dags/    

CMD airflow webserver -p 8080 & airflow scheduler


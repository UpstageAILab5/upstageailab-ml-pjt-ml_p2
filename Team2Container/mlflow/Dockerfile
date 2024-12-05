# base.Dockerfile
FROM my-base-image

# MLflow 및 필요한 패키지 설치
RUN pip install --no-cache-dir mlflow==2.16.2

# 작업 디렉토리 설정
WORKDIR /mlflow

# 아티팩트 저장을 위한 디렉토리 생성
RUN mkdir -p /mlflow/artifacts

# SQLite 데이터베이스 파일 생성
RUN touch /mlflow/mlflow.db

# MLflow 서버 실행
CMD mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
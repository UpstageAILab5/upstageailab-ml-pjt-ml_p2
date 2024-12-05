0. D:\Team2Container 처럼 원하는 곳에 폴더 생성
   docker desktop 실행

1.  Team2Container 폴더로 이동

2. 설치 컨테이너라 오래 걸림 
   docker build -t my-base-image -f base.Dockerfile .

3. 모든 서비스 빌드하고 실행해서
   docker compose up --build
   docker compose down

4. 각각의 서비스에 접속 가능
   127.0.0.1       streamlit
   120.0.0.1:5001  mlflow
   120.0.0.1:8081  airflow
   120.0.0.1:8051  streamlit
   120.0.0.1:8000  fastapi
   120.0.0.1:8000/docs  fastapi

5. docker desktop에서 컨테이너 확인
                   

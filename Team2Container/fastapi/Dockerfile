# 기본 설치는 base.Dockerfile 에서 공통으로 작성
FROM my-base-image

# 작업 디렉토리 설정
WORKDIR /app

COPY . /app

# uvicorn 및 fastapi 설치
RUN pip install --no-cache-dir fastapi uvicorn

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]


######################################################################
# Terminal에서 CLI 명령어 docker 하나만 실행하는 경우 예시
# 일반적으로는 docker-compose.yml을 만들고 실행한다.
# 1. 이미지 빌드 방법
#    >docker build -t hello-docker
# 2. 이미지 빌드 확인
#     > docker images
# 3. 빌드 이미지를 실행 (컨테이너로 실행)
#     > docker run hello-docker
# 4. 파이썬 코드 업데이트 후 도커 이미지를 실행하면 이전과 동일
# 5. 소스 변경 되었다면 다시 빌드 필요
######################################################################
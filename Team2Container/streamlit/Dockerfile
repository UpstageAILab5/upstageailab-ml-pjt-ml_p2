# base.Dockerfile
FROM my-base-image

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y fonts-nanum

RUN pip install --no-cache-dir streamlit streamlit-aggrid 

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
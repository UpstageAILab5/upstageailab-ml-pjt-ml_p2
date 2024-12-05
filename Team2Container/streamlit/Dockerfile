# base.Dockerfile
FROM my-base-image

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir streamlit streamlit-aggrid

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
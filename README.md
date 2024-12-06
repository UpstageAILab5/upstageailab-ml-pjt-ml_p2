# 영화 리뷰 감성 분석 모델 & MLops 구축
## 2조 오뚝이

| ![최진호](https://avatars.githubusercontent.com/u/40931237?s=88&v=4) | ![송주은](https://avatars.githubusercontent.com/u/182833254?s=88&v=4) | ![김남섭](https://avatars.githubusercontent.com/u/178737930?s=88&v=4) | ![김현진](https://avatars.githubusercontent.com/u/180828922?s=88&v=4) | ![박지은](https://avatars.githubusercontent.com/u/182731776?s=88&v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최진호](https://github.com/lojino)             |            [송주은](https://github.com/jsonghcbiz)             |            [김남섭](https://github.com/PotatoKim1)             |            [김현진](https://github.com/jinibizsite)             |            [박지은](https://github.com/FULLMOOONBY)                |
|                            팀장, Airflow, EC2, S3, 도커 및 총괄 담당                             |                            모델 실험, Streamlit, Docker                             |                            Docker, Container                             |                            Docker, Container, 모델실험                             |                            가상환경, Streamlit                             |

## 0. Overview
### Environment
- [환경 설정 페이지](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p2/blob/main/Team2Container/README.txt)

### Requirements
- [요구사항](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p2/blob/main/Team2Container/requirements.txt)

## 1. Competiton Info

### Overview

- 지금까지 배운 내용을 토대로, MLflow, Airflow를 활용한 모델 개발 및 모니터링 가능한 환경을 구축

### Timeline
- 2024-11-25 : Start Date
- ~2024. 11. 27 : 모델 개발 및 전체적인 MLops 아키텍처flow 도면 작성
- ~2024. 11. 29 : 1차 Streamlit을 활용하여 리뷰의 긍/부정 여부를 확인할 수 있다. 크롤링->mlflow->airflow 흐름 테스트. Docker Container를 활용하여 각 앱을 컨테이너 단위로 구성하고 연결. - Streamlit, nginx, mlflow, airflow
- ~2024. 12. 04 : 2차 Streamlit에서 추가 기능으로, openAI API의 영화 프롬프트를 활용하여 이미지를 만들 수 있고, 영화 내 나의 MBTI와 매칭하는 캐릭터를 찾아볼 수 있다. AWS에 도커 연결 및 각 기능의 작동여부 테스트.
- ~2024. 12. 06 : Dodcker 및 개발 환경을 AWS EC2 & S3에 업로드하여 외부 user들과 interactive한 환경 구축. Final submission deadline.

## 2. Components

### Directory

- 각 기능을 도커의 컨테이너 기능을 활용해 Build-up.
- Base.Dockerfile에 각 앱에 필요한 언어,라이브러리,모듈 등을 담아서 base.dodckerfile 이미지의 버전을 토대로 앱 동작.
- 그 이후 docker compose build 기능을 활용해 모든 컨테이너( airflow, fastAPI, MLflow, nginx, streamlit )들을 구동시켰습니다.
- 작동 확인 이후 모델은 aws s3에, 도커는 ec2에 업로드하였습니다.

e.g.
```
Team2Container
├── airflow
│   ├── dags
│   │   ├── data
│   │   │    └── rating.txt
│   │   ├── cgv_model_predict.py
│   │   ├── cgv_review_crawling.py
│   │   ├── crawling.py
│   │   ├── megabox_model_predict.py
│   │   ├── megabox_review_crawling.py
│   │   ├── model_train.py
│   │   └── s3.py
│   ├── Dockerfile
│   └── airflow.cfg
│
├── fastapi
│   ├── app
│   │    ├── __init__.py
│   │    ├── app.py
│   │    └── main.py
│   └── Dockerfile
│
└── mlflow
│    └── Dockerfile
│
└── nginx
│    ├── html
│    │     ├── images
│    │     │      └── 1.png
│    │     └── index.html
│    ├── Dockerfile
│    └── nginx.conf
│       
├── streamlit
│       ├── .streamlit
│       │       ├── config.toml
│       │       └── secrets.toml
│       └── app
│       │    ├── library
│       │    │     ├── cgv_crawler.py
│       │    │     ├── crawling.py
│       │    │     ├── megabox_crawler.py
│       │    │     ├── s3.py
│       │    │     └── senti_classifier_kobert.py
│       │    ├── menu
│       │    │     ├── home.py
│       │    │     ├── movie_review_analysis.py
│       │    │     └── text_analysis.py
│       │    └── app.py
│       └── Dockerfile
│
├── base.Dockerfile
│       
├── docker-compose.yml
│       
└── requirements.txt



```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_


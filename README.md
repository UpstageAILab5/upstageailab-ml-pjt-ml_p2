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

- 기존 IMDB 데이터셋 기반의 영어 감성 분석은 김인섭 강사님께서 시현을 해주셨던 관계로, 한국어 리뷰 분석 모델을 만드는 것으로 방향성 변경
- 데이터셋은 모델 학습을 위해 긍정/부정 라벨링이 되어 있는 대용량 데이터셋을 확보
- 이후 모델 파인 튜닝을 위하여 매일 cgv/megabox로부터 리뷰 데이터를 크롤링하여 지속적으로 확대함

  

### EDA

- 초기 학슴 데이터셋은 600000만개 리뷰의 긍정과 부정 라벨링이 완료되어 있었고, 이후 데이터는 cgv/megabox로부터 자연 형태의 리뷰를 그대로 크롤링하였음
- 여러 패키지와 토크나이저를 전용해보며 단어 출현 빈도, wordcloud 등을 통해 전처리 정화도 확인

### Data Processing

- 먼저 긍정과 부정의 의미가 명확하게 담긴 단어들을 집중적으로 학습시킬 수 있도록 cleaning 작업 진행 (기호 문자, html, 숫자 등)
- Konply를 활용해서 명사, 형태소 등 명확히 긍정과 부정을 구분할 수 있는 단어 단위 추출
- 이후 Kobert 모델 tokenizing 진행 

## 4. Modeling

### Model descrition

- SKT에서 배포한 Kobert 모델 최종 선택
- tinybert/albert를 사용해봤지만 영어에 최적화되어 있었고, 타 한국어 기반 모델 (WhitePeak, monologg/kobert 등)은 정확도 또는 confidence 레벨에서 차이가 있어 SKT/Kobert를 선택함

### Modeling Process
전체적인 아키텍쳐
<img width="716" alt="image" src="https://github.com/user-attachments/assets/b4b852e3-dee0-4b0c-8d53-b12b482f5896">
모델 비교 후 선정
<img width="477" alt="image" src="https://github.com/user-attachments/assets/e80db55b-2379-4769-8d1a-660e42614e9f">
Crawling Airflow
<img width="477" alt="image" src="https://github.com/user-attachments/assets/7a12730c-d524-44c6-b683-b02823284c84">
ML Airflow
<img width="477" alt="image" src="https://github.com/user-attachments/assets/3d348737-82c8-436d-9e96-0372c7d7d1be">
모델 학습 Slack Bot
<img width="432" alt="image" src="https://github.com/user-attachments/assets/e3b66f18-90a9-4480-b19b-aaf19a80bfc9">






## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation
[MLOPS.pdf](https://github.com/user-attachments/files/18034192/MLOPS.pdf)

- _Insert your presentaion file(pdf) link_


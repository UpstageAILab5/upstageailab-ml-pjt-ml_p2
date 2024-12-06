# 영화 리뷰 감성 분석 모델 & MLops 구축
## 2조 오뚝이

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최진호](https://github.com/UpstageAILab)             |            [송주은](https://github.com/UpstageAILab)             |            [김남섭](https://github.com/UpstageAILab)             |            [김현진](https://github.com/UpstageAILab)             |            [박지은](https://github.com/UpstageAILab)                |
|                            팀장, Airflow, EC2, S3, 도커 및 총괄 담당                             |                            모델 실험, Streamlit, Docker                             |                            Docker, Container                             |                            Docker, Container, 모델실험                             |                            가상환경, Streamlit                             |

## 0. Overview
### Environment
- Windows 개발환경 : 김남섭,김현진,박지은
- Mac 개발환경 : 최진호, 송주은
- AWS 자원활용 : EC2, S3

### Requirements
- tensorflow==2.18.0
- torch==2.5.1
- numpy==2.0.2
- pandas==2.2.3
- scikit-learn
- accelerate==1.1.1
- datasets==3.1.0
- Flask==3.1.0
- huggingface-hub==0.26.2
- keras==3.6.0
- transformers==4.46.3
- evaluate==0.4.3
- boto3==1.35.69
- botocore==1.35.69
- joblib==1.4.2
- jsonschema==4.23.0
- jsonschema-specifications==2024.10.1
- matplotlib==3.9.2
- matplotlib-inline==0.1.7
- scipy==1.13.1
- seaborn==0.13.2
- tf_keras==2.18.0
- tokenizers==0.20.3
- JPype1
- konlpy
- mlflow
- git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf
- SentencePiece

## 1. Competiton Info

### Overview

- 지금까지 배운 내용을 토대로, MLops를 활용한 모델 개발 및 모니터링 가능한 환경을 구축

### Timeline

- 2024. 11. 25 - Start Date
- ~2024. 11. 27 - 모델 개발 및 전체적인 MLops 아키텍처flow 도면 작성
- ~2024. 11. 29 - 1차 Streamlit을 활용하여 리뷰의 긍/부정 여부를 확인할 수 있다. Docker Container를 활용하여 각 앱을 컨테이너 단위로 구성하고 연결. - Streamlit, nginx, mlflow, airflow
- ~2024. 12. 04 - 2차 Streamlit에서 추가 기능으로, openAI API의 영화 프롬프트를 활용하여 이미지를 만들 수 있고, 영화 내 나의 MBTI와 매칭하는 캐릭터를 찾아볼 수 있다.
- ~2024. 12. 06 - Dodcker 및 개발 환경을 AWS EC2 & S3에 업로드하여 외부 user들과 interactive한 환경 구축.
- 2024. 12. 06 - Final submission deadline

## 2. Components

### Directory

- 각 기능을 도커의 컨테이너 기능을 활용해 Build-up 했습니다.
- Base.Dockerfile을 통하여 모든 이미지에 필요한 모듈과 라이브러리르 설치하고
- 그 아래에 airflow, fastAPI, MLflow, nginx, streamlit 컨테이너가 동작할 수 있는 환경으로 구축했습습니다.

e.g.

Team2Container
├── airflow
│   └── dags
│   │   └── data
│   │        └── rating.txt
│   │   ├── cgv_model_predict.py
│   │   ├── cgv_review_crawling.py
│   │   ├── crawling.py
│   │   ├── megabox_model_predict.py
│   │   ├── megabox_review_crawling.py
│   │   ├── model_train.py
│   │   └── s3.py
│   └── Dockerfile
│   └── airflow.cfg
│
├── fastapi
│   ├── app
│   │    └── __init__.py
│   │    └── app.py
│   │    └── main.py
│   └── Dockerfile
│
└── mlflow
│    └── Dockerfile
│
└── nginx
│    └── html
│          └── images
│                 └── 1.png
│          └── index.html
│    └── Dockerfile
│    └── nginx.conf
│       
└── streamlit
│       
└── base.Dockerfile
│       
└── docker-compose.yml
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

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_

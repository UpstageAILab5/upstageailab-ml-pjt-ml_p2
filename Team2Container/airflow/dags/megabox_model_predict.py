import torch
import os
import re
import json
import pandas as pd
from datetime import datetime       # 시간 설정

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from safetensors.torch import load_file

# Airflow 라이브러리
from airflow import DAG
from airflow.operators.python import PythonOperator

CINEMA = "megabox"
S3_BUCKET_PATH = f"cinema/{CINEMA}/predict/"
OBJECT_NAME_POSTFIX = f"_sentiment.json"

megabox_movies = {"24036800":"moana2", "24010200":"wicked", "24073500":"1승", "24043900":"gladiator2"}
model_path = 'data/kobert_konply'  # 예시 경로 (실제 저장된 모델 경로로 변경 필요)
file_path = f'{model_path}/{CINEMA}/'

# Custom BERT Classifier class (needs to be identical to the training class)
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.nn.functional.softmax(logits, dim=1)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.lower()  # 소문자로 변환
    text = text.strip()  # 문자열 양쪽 공백 제거
    text = text.replace('br', '')  # 'br' 태그 제거
    return text

def predict_sentiment(text, model_path, model_name='skt/kobert-base-v1'):
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = CustomBertClassifier(model_name, num_labels=2)
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 토크나이저 로드
    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name,
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True}
    )
    
    # 텍스트 전처리
    cleaned_text = clean_text(text)
    
    # 토크나이징
    encoded = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=300,
        return_tensors='pt'
    )
    
    # 예측
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = outputs.squeeze()
        predicted_class = torch.argmax(probabilities).item()
        
    # 결과 해석
    sentiment = "긍정" if predicted_class == 1 else "부정"
    confidence = probabilities[predicted_class].item()
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': f'{confidence:.2%}',
        'probabilities': {
            '부정': f'{probabilities[0].item():.2%}',
            '긍정': f'{probabilities[1].item():.2%}'
        }
    }

def prepare_data(index:int, **context):
    from s3 import s3_client
    from airflow.hooks.base import BaseHook

    # airflow connection 데이터를 불러옵니다.
    conn = BaseHook.get_connection("my_s3")
    # conn.extra를 JSON으로 파싱
    extra = json.loads(conn.extra)

    # 필요한 값 가져오기
    access_key = extra['aws_access_key_id']
    secret_access_key = extra['aws_secret_access_key']
    print(f"access_key: {access_key} / secret_access_key: {secret_access_key}")

    code_list = []
    code_list += list(megabox_movies.keys())

    s3 = s3_client(access_key=access_key, secret_access_key=secret_access_key)
    df = s3.get_reviews_csv(which_cinema=CINEMA, movie_code=code_list[index])

    ti = context['ti']
    json_str = df.to_json()  # DataFrame을 JSON 문자열로 변환
    ti.xcom_push(key='movie_reviews', value=json_str)
    ti.xcom_push(key='movie_codes', value=code_list)

def predict_data(index:int, **context):
    ti = context['ti']
    movie_codes = ti.xcom_pull(task_ids=f'prepare_{index}_task', key='movie_codes')
    json_str = ti.xcom_pull(task_ids=f'prepare_{index}_task', key='movie_reviews')
    movie_review_df = pd.read_json(json_str)  # JSON 문자열을 DataFrame으로 복원

    print(f"model_path: {model_path}")
    result = [predict_sentiment(review, model_path) for review in movie_review_df['review']]
    
    # JSON 파일로 저장
    file_name = f'{movie_codes[index]}{OBJECT_NAME_POSTFIX}'
    full_path = file_path + file_name
    os.makedirs(file_path, exist_ok=True)  # 디렉토리 생성 (존재하지 않을 경우)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"JSON 파일이 저장되었습니다: {full_path}")

def upload_file(index:int, **context):
    # S3
    from s3 import s3_client
    from airflow.hooks.base import BaseHook

    ti = context['ti']
    movie_codes = ti.xcom_pull(task_ids=f'prepare_{index}_task', key='movie_codes')
    movie_code = movie_codes[index]

    object_name = f'{movie_code}{OBJECT_NAME_POSTFIX}'
    full_path = file_path + object_name

    with open(full_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # airflow connection 데이터를 불러옵니다.
    conn = BaseHook.get_connection("my_s3")
    # conn.extra를 JSON으로 파싱
    extra = json.loads(conn.extra)

    # 필요한 값 가져오기
    access_key = extra['aws_access_key_id']
    secret_access_key = extra['aws_secret_access_key']
    print(f"access_key: {access_key} / secret_access_key: {secret_access_key}")
    
    # 파일 업로드
    s3 = s3_client(access_key=access_key, secret_access_key=secret_access_key)
    s3.upload_file(full_path, S3_BUCKET_PATH, object_name)

    ti.xcom_push(key=f'movie_code', value=movie_code)

##### 슬랙 알림 함수 #####
def slack_notification(**context):
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

    ti = context['ti']
    movie_0_code = ti.xcom_pull(task_ids='upload_0_task', key='movie_code')
    movie_1_code = ti.xcom_pull(task_ids='upload_1_task', key='movie_code')
    movie_2_code = ti.xcom_pull(task_ids='upload_2_task', key='movie_code')
    movie_3_code = ti.xcom_pull(task_ids='upload_3_task', key='movie_code')

    movie_codes = ti.xcom_pull(task_ids=f'prepare_task', key='movie_codes')
    
    message = f"""
* Megabox 영화 데이터 업로드 정보 *
- {megabox_movies[movie_0_code]}: 업로드 성공
- {megabox_movies[movie_1_code]}: 업로드 성공
- {megabox_movies[movie_2_code]}: 업로드 성공
- {megabox_movies[movie_3_code]}: 업로드 성공
    """
    slack_notification = SlackWebhookOperator(
        task_id='send_slack_notification_task',
        slack_webhook_conn_id='slack_webhook',
        message=message,
        username='news_bot',
        dag=context['dag']
    )
    
    # Slack 메시지를 실제로 전송
    slack_notification.execute(context=context)
    print(f'슬랙 알림 완료')

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 12, 4),
    'retries': 3,
    'env': {
        'NO_PROXY': '*',   # airflow로 외부 요청할 때 
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'   # 파이썬 버전 이슈 해결
    }
}

# DAG 정의
dag = DAG(
    'megabox_review_sentiment_for_4_movies',
    default_args=default_args,
    description='Predict review sentiment analysis for 4 movies and upload them to S3.',
    schedule_interval="30 17 * * *",  # 매일 KST 새벽 2시 30분 실행
    catchup=False,
    max_active_runs=4  # 동시에 실행 가능한 DAG 인스턴스 수
)

# Task 정의
prepare_0_task = PythonOperator(
    task_id='prepare_0_task',
    python_callable=prepare_data,
    op_kwargs={'index':0},
    provide_context=True,
    dag=dag
)
prepare_1_task = PythonOperator(
    task_id='prepare_1_task',
    python_callable=prepare_data,
    op_kwargs={'index':1},
    provide_context=True,
    dag=dag
)
prepare_2_task = PythonOperator(
    task_id='prepare_2_task',
    python_callable=prepare_data,
    op_kwargs={'index':2},
    provide_context=True,
    dag=dag
)
prepare_3_task = PythonOperator(
    task_id='prepare_3_task',
    python_callable=prepare_data,
    op_kwargs={'index':3},
    provide_context=True,
    dag=dag
)

predict_0_task = PythonOperator(
    task_id='predict_0_task',
    python_callable=predict_data,
    op_kwargs={'index':0},
    provide_context=True,
    dag=dag
)

predict_1_task = PythonOperator(
    task_id='predict_1_task',
    python_callable=predict_data,
    op_kwargs={'index':1},
    provide_context=True,
    dag=dag
)

predict_2_task = PythonOperator(
    task_id='predict_2_task',
    python_callable=predict_data,
    op_kwargs={'index':2},
    provide_context=True,
    dag=dag
)

predict_3_task = PythonOperator(
    task_id='predict_3_task',
    python_callable=predict_data,
    op_kwargs={'index':3},
    provide_context=True,
    dag=dag
)

upload_0_task = PythonOperator(
    task_id='upload_0_task',
    python_callable=upload_file,
    op_kwargs={'index':0},
    provide_context=True,
    dag=dag
)

upload_1_task = PythonOperator(
    task_id='upload_1_task',
    python_callable=upload_file,
    op_kwargs={'index':1},
    provide_context=True,
    dag=dag
)

upload_2_task = PythonOperator(
    task_id='upload_2_task',
    python_callable=upload_file,
    op_kwargs={'index':2},
    provide_context=True,
    dag=dag
)

upload_3_task = PythonOperator(
    task_id='upload_3_task',
    python_callable=upload_file,
    op_kwargs={'index':3},
    provide_context=True,
    dag=dag
)

# Slack 메시지 전송 Task
notify_task = PythonOperator(
    task_id='notify_task',
    python_callable=slack_notification,
    provide_context=True,
    dag=dag
)

# Task 의존성 설정
prepare_0_task >> predict_0_task
prepare_1_task >> predict_1_task
prepare_2_task >> predict_2_task
prepare_3_task >> predict_3_task
predict_0_task >> upload_0_task
predict_1_task >> upload_1_task
predict_2_task >> upload_2_task
predict_3_task >> upload_3_task
[upload_0_task, upload_1_task, upload_2_task, upload_3_task] >> notify_task
# 기본 라이브러리
import time
import pandas as pd
import os                           # 시스템 경로 설정
import json
import pytz                         # 시간대 설정
from datetime import datetime       # 시간 설정

# Airflow 라이브러리
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

# 변수
MEGABOX_URL = "https://www.megabox.co.kr/movie-detail/comment?rpstMovieNo="
OBJECT_NAME_POSTFIX = "_reviews.csv"
S3_BUCKET_PATH = "cinema/megabox/review/"

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'                                    # 환경 변수 설정 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

# 한국 시간 설정
def get_kst_time():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst).strftime('%y%m%d')

def prepare_data(**context):
    megabox_movies = {"24036800":"moana2", "24010200":"wicked", "24073500":"1승", "24043900":"gladiator2"}
    megabox_movie_codes = list(megabox_movies.keys())

    data = context['ti']
    data.xcom_push(key='movies', value=megabox_movies)
    data.xcom_push(key='movie_codes', value=megabox_movie_codes)

def get_movie_reviews(movie_index, review_count, **context):
    from crawling import crawling

    data = context['ti']
    movie_codes = data.xcom_pull(task_ids='prepare_task', key='movie_codes')
    movie_code = movie_codes[movie_index]
    full_url = MEGABOX_URL + movie_code

    crawl = crawling()
    movie_review_df = crawl.get_movie_reviews_on_cgv(url=full_url, review_count=10)

    json_str = movie_review_df.to_json()  # DataFrame을 JSON 문자열로 변환
    data.xcom_push(key='movie_code', value=movie_code)
    data.xcom_push(key='movie_review_df', value=json_str)

def upload_file(movie_index, **context):
    # S3
    from s3 import s3_client
    from airflow.hooks.base import BaseHook

    data = context['ti']
    movie_code = data.xcom_pull(task_ids=f'crawling_{movie_index}_task', key='movie_code')
    json_str = data.xcom_pull(task_ids=f'crawling_{movie_index}_task', key='movie_review_df')
    movie_review_df = pd.read_json(json_str)  # JSON 문자열을 DataFrame으로 복원

    # airflow connection 데이터를 불러옵니다.
    conn = BaseHook.get_connection("my_s3")
    # conn.extra를 JSON으로 파싱
    extra = json.loads(conn.extra)

    # 필요한 값 가져오기
    access_key = extra['aws_access_key_id']
    secret_access_key = extra['aws_secret_access_key']
    print(f"access_key: {access_key} / secret_access_key: {secret_access_key}")
    
    # S3 객체이름 / CSV 파일 경로와 이름
    object_name = f"{movie_code}{OBJECT_NAME_POSTFIX}"
    file_path = f"data/{object_name}"
    print(f"S3 Object name : {object_name}")
    
    # CSV 파일 생성
    movie_review_df.to_csv(file_path, index=False, encoding="utf-8")
    
    # 파일 업로드
    s3 = s3_client(access_key=access_key, secret_access_key=secret_access_key)
    is_successful = s3.upload_file(file_path, S3_BUCKET_PATH, object_name)
    result = "성공" if is_successful else "실패"
    
    data.xcom_push(key=f'movie_code', value=movie_code)
    data.xcom_push(key=f'movie_result', value=result)

##### 슬랙 알림 함수 #####
def slack_notification(**context):
    data = context['ti']
    movie_0_code = data.xcom_pull(task_ids='upload_0_task', key='movie_code')
    movie_1_code = data.xcom_pull(task_ids='upload_1_task', key='movie_code')
    movie_2_code = data.xcom_pull(task_ids='upload_2_task', key='movie_code')
    movie_3_code = data.xcom_pull(task_ids='upload_3_task', key='movie_code')

    movie_0_result = data.xcom_pull(task_ids='upload_0_task', key='movie_result')
    movie_1_result = data.xcom_pull(task_ids='upload_1_task', key='movie_result')
    movie_2_result = data.xcom_pull(task_ids='upload_2_task', key='movie_result')
    movie_3_result = data.xcom_pull(task_ids='upload_3_task', key='movie_result')

    megabox_movies = data.xcom_pull(task_ids='prepare_task', key='movies')
    
    message = f"""
* Megabox 영화 리뷰 크롤링 업로드 정보 *
- {megabox_movies[movie_0_code]}: 업로드 {movie_0_result}
- {megabox_movies[movie_1_code]}: 업로드 {movie_1_result}
- {megabox_movies[movie_2_code]}: 업로드 {movie_2_result}
- {megabox_movies[movie_3_code]}: 업로드 {movie_3_result}
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
    'megabox_review_crawling_for_4_movies',
    default_args=default_args,
    description='We crawl reviews for each of the four movies and upload them to S3.',
    schedule_interval="30 16 * * *",  # 매일 KST 새벽 1시 실행
    catchup=False
)

# Task 정의
prepare_task = PythonOperator(
    task_id='prepare_task',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag
)

crawling_0_task = PythonOperator(
    task_id='crawling_0_task',
    python_callable=get_movie_reviews,
    op_kwargs={'movie_index':0, 'review_count':100},
    provide_context=True,
    dag=dag
)

crawling_1_task = PythonOperator(
    task_id='crawling_1_task',
    python_callable=get_movie_reviews,
    op_kwargs={'movie_index':1, 'review_count':100},
    provide_context=True,
    dag=dag
)

crawling_2_task = PythonOperator(
    task_id='crawling_2_task',
    python_callable=get_movie_reviews,
    op_kwargs={'movie_index':2, 'review_count':100},
    provide_context=True,
    dag=dag
)

crawling_3_task = PythonOperator(
    task_id='crawling_3_task',
    python_callable=get_movie_reviews,
    op_kwargs={'movie_index':3, 'review_count':100},
    provide_context=True,
    dag=dag
)

upload_0_task = PythonOperator(
    task_id='upload_0_task',
    python_callable=upload_file,
    op_kwargs={'movie_index':0},
    provide_context=True,
    dag=dag
)

upload_1_task = PythonOperator(
    task_id='upload_1_task',
    python_callable=upload_file,
    op_kwargs={'movie_index':1},
    provide_context=True,
    dag=dag
)

upload_2_task = PythonOperator(
    task_id='upload_2_task',
    python_callable=upload_file,
    op_kwargs={'movie_index':2},
    provide_context=True,
    dag=dag
)

upload_3_task = PythonOperator(
    task_id='upload_3_task',
    python_callable=upload_file,
    op_kwargs={'movie_index':3},
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
prepare_task >> [crawling_0_task, crawling_1_task, crawling_2_task, crawling_3_task]
crawling_0_task >> upload_0_task
crawling_1_task >> upload_1_task
crawling_2_task >> upload_2_task
crawling_3_task >> upload_3_task
[upload_0_task, upload_1_task, upload_2_task, upload_3_task] >> notify_task
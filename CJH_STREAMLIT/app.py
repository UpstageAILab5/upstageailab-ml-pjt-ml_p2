import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from s3 import s3_client

# 페이지 설정과 스타일링
st.set_page_config(page_title="영화 리뷰 감성 분석", layout="wide")
st.markdown(
    """
    <style>
        #fce8dc1b {
            text-align: center;
        }
        [data-testid="stMainBlockContainer"] {
            width: 80%;
            margin: auto;
        }
        .stHorizontalBlock st-emotion-cache-ocqkz7 e1f1d6gn5 {
            display: flex;
            width: 100%;
        }
        .stColumn st-emotion-cache-1r6slb0 e1f1d6gn3 {
            flex: 1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 데이터 준비
images = s3_client(st.secrets['access_key'], st.secrets['secret_access_key']).get_images()
detail_images = [
    "https://fastcampus-ml-p2-bucket.s3.ap-northeast-2.amazonaws.com/image/detail_image1.avif"
]
titles = [
    "Moana2",
    "wicked",
    "Gladiator2"
]
buttons = []

# 메인 페이지 시작
st.title("영화 리뷰 감성 분석")

cols = st.columns(3)
for idx, col in enumerate(cols):
    with col.container(border=True, key=f"{titles[idx]}_container"):
        st.image(image=images[idx], use_container_width=True)
        buttons.append(st.button(label=titles[idx], key=f"{titles[idx]}_btn", use_container_width=True))

def view(idx):
    
    with st.container(border=True):
        st.progress(80, text="⭐️⭐️ 별점 ⭐️⭐️")

    exp_cols = st.columns(2)
    with exp_cols[0].container(border=True):
        st.image(image=detail_images[0], use_container_width=True)
    with exp_cols[1].container(border=True):
        # 데이터 설정
        labels = ['유쾌', '상쾌', '통쾌']
        sizes = [50, 30, 20]
        colors = ['lightskyblue', 'lightcoral', 'lightgreen']

        # 파이 차트 그리기
        plt.figure(figsize=(7, 7))
        plt.axis('equal')  # 원형으로 그리기 위해 비율 설정
        plt.rcParams['font.family'] = 'AppleGothic'  # 맥의 경우
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

        st.pyplot(fig)

with st.expander("영화 리뷰 감성 보기",expanded=True):
    if buttons[0]: 
        st.title(titles[0])
        view(0)
    elif buttons[1]:
        st.title(titles[1])
    elif buttons[2]:
        st.title(titles[2])
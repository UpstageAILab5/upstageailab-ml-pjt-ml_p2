import pandas as pd
import streamlit as st
import requests
import plotly.express as px
from datetime import datetime


# 스트림릿 페이지 설정
st.title('IMDB Movie Info') 
movie_title = st.text_input('영화 제목 입력', '')  # 영화 제목 입력
current_year = datetime.now().year  # 현재 연도 설정
movies_df = pd.DataFrame()  # 영화 검색 결과 저장할 데이터프레임 초기화

# 영화 검색 조건 설정
type_filter = st.selectbox('콘텐츠 유형', ['movie', 'series', 'episode'])  # 영화 유형 선택
year_filter = st.slider('출시년도', 2000, current_year, (2000, current_year))  # 영화 연도 선택
rating_filter = st.slider('IMDB평점', 0.0, 10.0, (0.0, 10.0))  # 평점 선택

api_key = st.secrets['api_key']

# 영화 검색 요청
if movie_title:
    # OMDb API 요청 URL 설정
    omdb_api_url = f'http://www.omdbapi.com/'
    # api_key = st.secrets['api_key']

    movie_df = pd.DataFrame()

    # OMDb API 요청 파라미터 설정

    for year in range(year_filter[0], year_filter[1] + 1):
        search_params = {
            'apikey': api_key,                          # API 키 설정
            's': movie_title,                           # 영화 제목 설정
            'type': type_filter,                        # 콘텐츠 유형 설정
            'y': f'{year_filter[0]}-{year_filter[1]}',  # 출시년도 설정
            'r': 'json',                                # 응답 형식 설정
        }
        with st.spinner('영화 검색 중...'):
        # OMDb API 요청 실행
            response = requests.get(omdb_api_url, params=search_params)
            
            if response.status_code == 200:
                data = response.json()
                # 영화 검색 결과 처리
                if "Search" in data:
                    for movie in data['Search']:
                        # 영화 상세 정보 요청
                        detail_params = {
                        'apikey': api_key,          # API 키 설정
                        'i': movie['imdbID'],       # 영화 ID 설정
                        'plot': 'full',             # 영화 상세 정보 설정
                            'r': 'json',                # 응답 형식 설정
                        }
                        detail_response = requests.get(omdb_api_url, params=detail_params)
                        if detail_response.status_code == 200:
                            detail_data = detail_response.json()
                            detail_data['Year'] = detail_data['Year'].rstrip('-')  # 연도 데이터 형식 변환 rstrip: 문자열 오른쪽 공백 제거
                            
                            # 추가 정보를 위한 영화 검색 조건 충족 여부 확인
                            try: 
                                imdb_rating = detail_data.get('imdbRating', 'N/A')
                                if imdb_rating != 'N/A' and float(imdb_rating) >= rating_filter[0] and float(imdb_rating) <= rating_filter[1]:
                                    year_data = detail_data.get('Year', '').split('-')[0].strip()
                            
                    # 영화 상세 정보 데이터프레임 생성
                                    new_row_df = pd.DataFrame({
                                        'Poster': detail_data['Poster'],
                                        'Title': [f'{detail_data["Title"]} ({detail_data["Year"]})'],
                                        'Year': detail_data['Year'],
                                        'Rated': detail_data['Rated'],
                                        'Runtime': detail_data['Runtime'],
                                        'Released': detail_data['Released'],
                                        'Genre': detail_data['Genre'],
                                        'Director': detail_data['Director'],
                                        'Writer': detail_data['Writer'],
                                        'Actors': detail_data['Actors'],
                                        'Plot': detail_data['Plot'],
                                        'Language': detail_data['Language'],
                                        'Country': detail_data['Country'],
                                        'Awards': detail_data['Awards'],
                                        'IMDB Rating': detail_data['imdbRating'],
                                        'IMDB Votes': detail_data['imdbVotes']})

                                    # 영화 검색 결과 데이터프레임에 추가
                                    movies_df = pd.concat([movies_df, new_row_df], ignore_index=True)
                            except Exception as e:
                                st.error(f'영화 평점 정보를 불러올 수 없습니다. {str(e)}')
                else:
                    st.error('영화를 찾을 수 없습니다.')
            else:
                st.error('영화 제목을 확인해주세요.')

# 영화 검색 결과 탭 생성
tab1, tab2 = st.tabs(['영화 검색 결과', '영화 평점 정보'])
# 영화 검색 결과 탭
with tab1:
    if (len(movies_df) > 0):
        st.header('영화 검색 결과')
        for i in range(len(movies_df)):
            col1, col2 = st.columns([1, 2])  # 컬럼 생성 1,2 = 이미지, 제목
            # 이미지 출력
            with col1:
                if movies_df.iloc[i]['Poster'] != 'N/A':
                    st.image(movies_df.iloc[i]['Poster'], caption=movies_df.iloc[i]['Title'], use_column_width=True)
                else:
                    st.image('포스터 사진 없음')
            # 영화 정보 출력
            with col2:
                st.subheader(movies_df.iloc[i]['Title'])

                col1, col2, col3 = st.columns(3)
                col1.write(f'IMDB Rating: {movies_df.iloc[i]["IMDB Rating"]}')
                col2.write(f'Rated: {movies_df.iloc[i]["Rated"]}')
                col3.write(f'Runtime: {movies_df.iloc[i]["Runtime"]}')

                st.write(f'Released: {movies_df.iloc[i]["Released"]}')
                st.write(f'Genre: {movies_df.iloc[i]["Genre"]}')
                st.write(f'Director: {movies_df.iloc[i]["Director"]}')
                st.write(f'Writer: {movies_df.iloc[i]["Writer"]}')
                st.write(f'Actors: {movies_df.iloc[i]["Actors"]}')
                st.write(f'Plot: {movies_df.iloc[i]["Plot"]}')
                st.write(f'Language: {movies_df.iloc[i]["Language"]}')
                st.write(f'Country: {movies_df.iloc[i]["Country"]}')
                st.write(f'Awards: {movies_df.iloc[i]["Awards"]}')
            st.divider()
# 영화 평점 정보 탭
with tab2:
    if (len(movies_df) > 0):
        fig = px.bar(movies_df, x='Title', y='IMDB Rating')
        st.header('IMDB Rating')
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
        fig = px.bar(movies_df, x='Title', y='IMDB Votes')
        st.header('IMDB Votes')
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)


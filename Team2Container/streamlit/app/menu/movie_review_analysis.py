import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
import os

from library.senti_classifier_kobert import predict_sentiment
from library.cgv_crawler import CGVTitleCrawler, CGVDetailCrawler
from library.megabox_crawler import MegaboxTitleCrawler, MegaboxDetailCrawler



def analyze_sentiment(reviews_df, model_path="kobert_konply"):
    try: 
        reviews_df["sentiment_analysis"] = reviews_df["review"].apply(
            lambda x: predict_sentiment(x, model_path)
        )
        reviews_df["sentiment"] = reviews_df["sentiment_analysis"].apply(lambda x: x["sentiment"])
        reviews_df["confidence"] = reviews_df["sentiment_analysis"].apply(lambda x: x["confidence"])
    except Exception as e:
        st.error(f"감성 분석 중 오류 발생: {e}")
    return reviews_df

def review_analysis():
    # Custom CSS for styling
    st.markdown(
        """
        <style>
            /* General page styling */
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #1C1C1E;
                color: #F0F0F0;
                font-family: 'Arial', sans-serif;
            }

            h1 {
                color: #FF416C;
                text-align: center;
                font-size: 36px;
                margin-bottom: 10px;
            }

            h2 {
                color: #FFFFFF;
                font-size: 28px;
                margin-top: 20px;
            }

            .stButton > button {
                background: linear-gradient(90deg, #FF416C, #FF4B2B);
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease-in-out;
                box-shadow: 0 4px 10px rgba(255, 65, 108, 0.3);
            }

            .stButton > button:hover {
                background: linear-gradient(90deg, #FF4B2B, #FF416C);
                transform: scale(1.05);
            }

            .metric-box {
                background-color: #2C2C2E;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }

            .section-header {
                color: #28A745;
                font-size: 24px;
                margin-top: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page Title
    st.title("🎥 영화 리뷰 크롤링 및 감성 분석")
    st.write("이 페이지는 영화 리뷰를 크롤링하고 감성 분석을 수행하는 페이지입니다!")

    def generate_prompt_from_review(review_text, sentiment):
        """Generate an appropriate prompt based on review content and sentiment"""
        base_prompt = f"Create a cinematic movie poster based on the following review: {review_text}"
        
        if sentiment == "긍정":
            style_prompt = "Use bright, vibrant colors and uplifting imagery with dramatic lighting"
        else:
            style_prompt = "Use dark, moody colors and dramatic shadows with tense atmosphere"
        
        return f"{base_prompt}. {style_prompt}. Make it in professional movie poster style with high quality rendering."

    if 'cgv_reviews' not in st.session_state:
        st.session_state['cgv_reviews'] = None
    if 'megabox_reviews' not in st.session_state:
        st.session_state['megabox_reviews'] = None

    # 영화 제목 선택 세션 상태 변수
    if 'movie_titles' not in st.session_state:
        st.session_state.movie_titles = []

    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    if "show_megabox_sentiment" not in st.session_state:
        st.session_state.show_megabox_sentiment = False
    if "show_cgv_sentiment" not in st.session_state:
        st.session_state.show_cgv_sentiment = False


    st.markdown("---")
    st.header("🔍 CGV 리뷰 크롤링")

    with st.expander("CGV 리뷰 크롤링", expanded=False):
        cgv_search_query = st.text_input("검색할 영화 제목을 입력하세요:", key="cgv_search_query")
        if st.button("CGV 검색", key="movie_search"):
            if cgv_search_query:
                with st.spinner('CGV 검색중...'):
                    cgv = CGVTitleCrawler()
                    movie_titles, cgv_movie_info = cgv.fetch_movie_titles(cgv_search_query)
                    if movie_titles:
                        st.session_state.movie_titles = movie_titles
                        st.session_state.cgv_movie_info = cgv_movie_info
                        st.session_state.selected_movie = movie_titles[0]
                    else:
                        st.warning("CGV에서 영화를 찾을 수 없습니다.")
            else:
                st.warning("영화 제목을 입력해주세요.")
    
        if st.session_state.movie_titles:
            st.session_state.cgv_selected_movie = st.selectbox(
                "CGV 검색 결과:", 
                st.session_state.movie_titles,
                key="cgv_movie_selectbox",
                format_func=lambda x: f"{x} ({[info['rating'] for info in st.session_state.cgv_movie_info if info['title'] == x][0]})"
            )
        if st.session_state.get("cgv_selected_movie"):
            cgv_review_limit = st.number_input("크롤링할 CGV 리뷰 수", min_value=1, value=5, key="cgv_review_limit")
            if st.button("영화 정보 가져오기", key="cgv_info_button"):
                with st.spinner("영화 정보 가져오는 중... 잠시만 기다려주세요."):
                    cgv_detail = CGVDetailCrawler()
                    cgv_movie_data = cgv_detail.crawl_movie_details(st.session_state.selected_movie, review_limit=cgv_review_limit)
                    if cgv_movie_data:
                        st.write('### 영화 정보')
                        st.write(f'**제목:** {cgv_movie_data["Title"]}')
                        st.write(f'**감독:** {cgv_movie_data["Directors"]}')
                        st.write(f'**출연:** {cgv_movie_data["Cast"]}')
                        st.write(f'**줄거리:** {cgv_movie_data["Plot"]}')

                        st.dataframe(cgv_movie_data["Reviews"])
                        st.session_state.cgv_reviews = cgv_movie_data["Reviews"]
                        st.session_state.show_cgv_sentiment = True
                    else:
                        st.warning("영화 정보를 찾을 수 없습니다.")

                    #     if st.button("리뷰 저장", key="cgv_save"):
                    #         output_path = f'{movie_data["Title"]}_reviews.csv'
                    #         movie_data["Reviews"].to_csv(output_path, index=False)
                    #         st.success(f"리뷰를 {output_path}에 저장했습니다.")
                    #         # Store reviews in session state
                    #         st.session_state.cgv_reviews = movie_data["Reviews"]
                    # else:
                    #     st.warning("영화 정보를 찾을 수 없습니다.")
                    #     return movie_data

        # Display sentiment analysis section if reviews exist or show_sentiment_analysis is True
        if st.session_state.cgv_reviews is not None and st.session_state.show_cgv_sentiment:
            st.markdown("### CGV 감성 분석")
            if st.button("CGV 리뷰 감성 분석", key="cgv_sentiment"):
                with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                    # 파일 경로 설정 (절대 경로)
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(BASE_DIR, "../kobert_konply")
                    st.session_state.cgv_reviews = analyze_sentiment(st.session_state.cgv_reviews, model_path)
                    st.success("감성 분석 완료!")
                    st.subheader("감성 분석 결과")
                    st.dataframe(st.session_state.cgv_reviews[["review", "sentiment", "confidence"]], use_container_width=True)


                    st.download_button(
                        "CGV 감성 분석 결과 다운로드",
                        st.session_state.cgv_reviews.to_csv(index=False).encode("utf-8"),
                        f"cgv_sentiment_results.csv",
                        "text/csv",
                        key="cgv_sentiment_download_button"
                    )

                    st.subheader("📊 시각화 결과")

                    fig = px.histogram(
                        st.session_state.cgv_reviews,
                        x="confidence",
                        color="sentiment",
                        title="감성 신뢰도 분포",
                        labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                        color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Add pie chart for sentiment distribution
                    sentiment_counts = st.session_state.cgv_reviews['sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="감성 분석 분포",
                        color=sentiment_counts.index,
                        color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.subheader("📋 분석 결과")
                    gb = GridOptionsBuilder.from_dataframe(st.session_state.cgv_reviews[["review", "sentiment", "confidence"]])
                    gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
                    gb.configure_default_column(
                        groupable=True,
                        value=True,
                        enableRowGroup=True,
                        editable=False,
                        filterable=True,
                    )
                    gb.configure_column("confidence", type=["numericColumn"], precision=2)
                    grid_options = gb.build()

                    AgGrid(
                        st.session_state.cgv_reviews[["review", "sentiment", "confidence"]],
                        gridOptions=grid_options,
                        height=300,
                        theme="balham",  # "light", "dark", "blue", "fresh", "material"
                        update_mode="MODEL_CHANGED",
                        fit_columns_on_grid_load=True,
                    )

    st.markdown("---")
    st.header("🔍 Megabox 리뷰 크롤링")
    with st.expander("Megabox 리뷰 크롤링", expanded=False):
        megabox_search_query = st.text_input("검색할 영화 제목을 입력하세요:", key="megabox_search_query")
        if st.button("Megabox 검색", key="megabox_movie_search"):
            if megabox_search_query:
                with st.spinner('Megabox 검색중...'):
                    megabox = MegaboxTitleCrawler()
                    movie_titles, megabox_movie_info = megabox.fetch_movie_titles(megabox_search_query)
                    if movie_titles:
                        st.session_state.movie_titles = movie_titles
                        st.session_state.megabox_movie_info = megabox_movie_info
                        st.session_state.selected_movie = movie_titles[0]
                    else:
                        st.warning("Megabox에서 영화를 찾을 수 없습니다.")
            else:
                st.warning("영화 제목을 입력해주세요.")
    
        if st.session_state.movie_titles:
            st.session_state.megabox_selected_movie = st.selectbox(
                "Megabox 검색 결과:", 
                st.session_state.movie_titles,  # Access the session state variable
                key="megabox_movie_selectbox",  # Unique key for Megabox
                format_func=lambda x: f"{x}".replace("()", "")  # Remove parentheses
            )
        if st.session_state.get("megabox_selected_movie"):
            megabox_review_limit = st.number_input("크롤링할 Megabox 리뷰 수", min_value=1, value=5, key="megabox_review_limit")
            if st.button("영화 정보 가져오기", key="megabox_info_button"):
                with st.spinner("영화 정보 가져오는 중... 잠시만 기다려주세요."):
                    megabox_detail = MegaboxDetailCrawler()
                    megabox_movie_data = megabox_detail.crawl_movie_details(st.session_state.selected_movie, 
                                                                    review_limit=megabox_review_limit)
                    if megabox_movie_data:
                        st.write('### 영화 정보')
                        st.write(f'**제목:** {megabox_movie_data["Title"]}')
                        st.write(f'**감독:** {megabox_movie_data["Directors"]}')
                        st.write(f'**출연:** {megabox_movie_data.get("Casts", "정보 없음")}')
                        st.write(f'**줄거리:** {megabox_movie_data["Plot"]}')

                        st.dataframe(megabox_movie_data["Reviews"])
                        st.session_state.megabox_reviews = megabox_movie_data["Reviews"]
                        st.session_state.show_megabox_sentiment = True
                    else:
                        st.warning("영화 정보를 찾을 수 없습니다.")
                    #     if st.button("리뷰 저장", key="megabox_save"):
                    #         output_path = f'{movie_data["Title"]}_reviews.csv'
                    #         movie_data["Reviews"].to_csv(output_path, index=False)
                    #         st.success(f"리뷰를 {output_path}에 저장했습니다.")
                    #         # Store reviews in session state
                    #         st.session_state.megabox_reviews = movie_data["Reviews"]
                    #         # Show the sentiment analysis button immediately after saving
                    #         st.session_state.show_megabox_sentiment = True
                    # else:
                    #     st.warning("영화 정보를 찾을 수 없습니다.")


        # Display sentiment analysis section if reviews exist or show_megabox_sentiment is True
        if st.session_state.megabox_reviews is not None and st.session_state.show_megabox_sentiment:
            st.markdown("### Megabox 감성 분석")
            if st.button("Megabox 리뷰 감성 분석", key="megabox_sentiment"):
                with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(BASE_DIR, "../kobert_konply")
                    st.session_state.megabox_reviews = analyze_sentiment(st.session_state.megabox_reviews, model_path)
                    st.success("감성 분석 완료!")
                    st.subheader("감성 분석 결과")
                    
                    # Create DataFrame for display and download
                    st.dataframe(st.session_state.megabox_reviews[["review", "sentiment", "confidence"]])


                    # Download button
                    st.download_button(
                        "Megabox 감성 분석 결과 다운로드",
                        st.session_state.megabox_reviews.to_csv(index=False).encode("utf-8"),
                        f"megabox_sentiment_results.csv",
                        "text/csv",
                        key="megabox_sentiment_download_button"
                    )

                    # Visualizations
                    st.subheader("📊 시각화 결과")
                    
                    # Histogram
                    fig_hist = px.histogram(
                        st.session_state.megabox_reviews,
                        x="confidence",
                        color="sentiment",
                        title="감성 신뢰도 분포",
                        labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                        color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Pie chart
                    sentiment_counts = st.session_state.megabox_reviews['sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="감성 분석 분포",
                        color=sentiment_counts.index,
                        color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Data grid
                    st.subheader("📋 상세 분석 결과")
                    gb = GridOptionsBuilder.from_dataframe(st.session_state.megabox_reviews)
                    gb.configure_pagination(paginationAutoPageSize=True)
                    gb.configure_default_column(
                        groupable=True,
                        value=True,
                        enableRowGroup=True,
                        editable=False,
                        filterable=True,
                    )
                    gb.configure_column("confidence", type=["numericColumn"], precision=2)
                    grid_options = gb.build()

                    AgGrid(
                        st.session_state.megabox_reviews,
                        gridOptions=grid_options,
                        height=300,
                        theme="balham",
                        update_mode="MODEL_CHANGED",
                        fit_columns_on_grid_load=True,
                    )
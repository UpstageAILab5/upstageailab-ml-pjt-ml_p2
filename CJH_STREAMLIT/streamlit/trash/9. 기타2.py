import streamlit as st
from review_crawling import (  # Import your predefined crawling functions
    get_movie_reviews_on_cgv,
    get_movie_reviews_on_megabox,
    CGV_URL,
    MEGABOX_URL,
    CGV_MOVIE_CODES,
    MEGABOX_MOVIE_CODES,
)
from senti_classifier_kobert import predict_sentiment
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.express as px
from nav import inject_custom_navbar
from image_generator import ImageGenerator



# Page configuration
st.set_page_config(
    page_title="영화 리뷰 크롤링 및 감성 분석",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_navbar()

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

def analyze_sentiment(reviews, model_path="kobert_konply"):
    """Perform sentiment analysis using the senti_classifier_kobert model."""
    reviews["sentiment_analysis"] = reviews["review"].apply(lambda x: predict_sentiment(x, model_path))
    reviews["sentiment"] = reviews["sentiment_analysis"].apply(lambda x: x["sentiment"])
    reviews["confidence"] = reviews["sentiment_analysis"].apply(lambda x: x["confidence"])
    return reviews

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

def streamlit_movie_search():
    st.markdown("---")
    st.header("🔍 CGV 리뷰 크롤링")
    with st.expander("CGV 리뷰 크롤링", expanded=False):
        movie_list_cgv = list(CGV_MOVIE_CODES.keys())
        selected_movie_cgv = st.selectbox(
            "영화 선택 (CGV)",
            movie_list_cgv,
            index=None,
            placeholder="영화를 선택하세요...",
            key="cgv_selectbox",  # Unique key for CGV selectbox
        )
        cgv_review_limit = st.number_input("크롤링할 CGV 리뷰 수", min_value=1, value=5, key="cgv_limit")

        if st.button("CGV 리뷰 크롤링 시작", key="cgv_button"):
            if selected_movie_cgv:
                with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                    cgv_url = CGV_URL + CGV_MOVIE_CODES[selected_movie_cgv]
                    st.session_state.cgv_reviews = get_movie_reviews_on_cgv(url=cgv_url, review_limit=cgv_review_limit)

                    st.success("리뷰 크롤링 완료!")
                    st.subheader("CGV 리뷰")
                    st.dataframe(st.session_state.cgv_reviews, use_container_width=True)

                    st.download_button(
                        "CGV 리뷰 다운로드",
                        st.session_state.cgv_reviews.to_csv(index=False).encode("utf-8"),
                        f"cgv_reviews_{selected_movie_cgv}.csv",
                        "text/csv",
                    )

        # Always display sentiment analysis button if reviews exist
        if st.session_state.cgv_reviews is not None:
            if st.button("CGV 리뷰 감성 분석", key="cgv_sentiment"):
                with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                    st.session_state.cgv_reviews = analyze_sentiment(st.session_state.cgv_reviews)
                    st.success("감성 분석 완료!")
                    st.subheader("감성 분석 결과")
                    st.dataframe(st.session_state.cgv_reviews[["review", "sentiment", "confidence"]], use_container_width=True)

                    st.download_button(
                        "CGV 감성 분석 결과 다운로드",
                        st.session_state.cgv_reviews.to_csv(index=False).encode("utf-8"),
                        f"cgv_sentiment_results_{selected_movie_cgv}.csv",
                        "text/csv",
                    )
                    # with st.expander("감성 분석 결과 보기", expanded=False):
                    fig = px.histogram(
                        st.session_state.cgv_reviews[["review", "sentiment", "confidence"]],
                        x="confidence",
                        color="sentiment",
                        title="감성 신뢰도 분포",
                        labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                        color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

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

                    if st.button("Generate Movie Poster from Selected Review"):
                        selected_review = st.session_state.cgv_reviews.iloc[0]  # You can modify this to let users select a review
                        prompt = generate_prompt_from_review(
                            selected_review['review'],
                            selected_review['sentiment']
                        )
                        
                        with st.spinner("Generating movie poster... This may take a few minutes."):
                            generator = ImageGenerator()
                            image_paths = generator.generate_image(prompt)
                            
                            if image_paths:
                                for path in image_paths:
                                    st.image(path, caption="Generated Movie Poster", use_column_width=True)
                            else:
                                st.error("Failed to generate image. Please try again.")

    st.markdown("---")
    st.header("🔍 메가박스 리뷰 크롤링")
    with st.expander("메가박스 리뷰 크롤링", expanded=False):
        movie_list_megabox = list(MEGABOX_MOVIE_CODES.keys())
        selected_movie_megabox = st.selectbox(
            "영화 선택 (메가박스)",
            movie_list_megabox,
            index=None,
            placeholder="영화를 선택하세요...",
            key="megabox_selectbox",  # Unique key for Megabox selectbox
        )
        megabox_review_limit = st.number_input("크롤링할 메가박스 리뷰 수", min_value=1, value=5, key="megabox_limit")

        if st.button("메가박스 리뷰 크롤링 시작", key="megabox_button"):
            if selected_movie_megabox:
                with st.spinner("리뷰 크롤링 중... 잠시만 기다려주세요."):
                    megabox_url = MEGABOX_URL + MEGABOX_MOVIE_CODES[selected_movie_megabox]
                    st.session_state.megabox_reviews = get_movie_reviews_on_megabox(
                        url=megabox_url, review_limit=megabox_review_limit
                    )

                    st.success("리뷰 크롤링 완료!")
                    st.subheader("메가박스 리뷰")
                    st.dataframe(st.session_state.megabox_reviews, use_container_width=True)

                    st.download_button(
                        "메가박스 리뷰 다운로드",
                        st.session_state.megabox_reviews.to_csv(index=False).encode("utf-8"),
                        f"megabox_reviews_{selected_movie_megabox}.csv",
                        "text/csv",
                    )

        # Always display sentiment analysis button if reviews exist
        if st.session_state.megabox_reviews is not None:
            if st.button("메가박스 리뷰 감성 분석", key="megabox_sentiment"):
                with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                    st.session_state.megabox_reviews = analyze_sentiment(st.session_state.megabox_reviews)
                    st.success("감성 분석 완료!")
                    st.subheader("감성 분석 결과")
                    st.dataframe(st.session_state.megabox_reviews[["review", "sentiment", "confidence"]], use_container_width=True)

                    st.download_button(
                        "메가박스 감성 분석 결과 다운로드",
                        st.session_state.megabox_reviews.to_csv(index=False).encode("utf-8"),
                        f"megabox_sentiment_results_{selected_movie_megabox}.csv",
                        "text/csv",
                    )
                    with st.expander("감성 분석 결과 보기", expanded=False):
                        fig = px.histogram(
                            st.session_state.megabox_reviews[["review", "sentiment", "confidence"]],
                            x="confidence",
                            color="sentiment",
                            title="감성 신뢰도 분포",
                            labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                            color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📋 분석 결과")
                    gb = GridOptionsBuilder.from_dataframe(st.session_state.megabox_reviews[["review", "sentiment", "confidence"]])
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
                        st.session_state.megabox_reviews[["review", "sentiment", "confidence"]],
                        gridOptions=grid_options,
                        height=300,
                        theme="balham",  # "light", "dark", "blue", "fresh", "material"
                        update_mode="MODEL_CHANGED",
                        fit_columns_on_grid_load=True,
                    )

                    if st.button("Generate Movie Poster from Selected Review"):
                        selected_review = st.session_state.megabox_reviews.iloc[0]  # You can modify this to let users select a review
                        prompt = generate_prompt_from_review(
                            selected_review['review'],
                            selected_review['sentiment']
                        )
                        
                        with st.spinner("Generating movie poster... This may take a few minutes."):
                            generator = ImageGenerator()
                            image_paths = generator.generate_image(prompt)
                            
                            if image_paths:
                                for path in image_paths:
                                    st.image(path, caption="Generated Movie Poster", use_column_width=True)
                            else:
                                st.error("Failed to generate image. Please try again.")

if __name__ == "__main__":
    streamlit_movie_search()
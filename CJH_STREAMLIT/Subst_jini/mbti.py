import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# **MBTI와 캐릭터 매칭 데이터**
MBTI_CHARACTERS = {
    "Wicked": {
        "INTJ": "Elphaba - Strategic and Independent Witch",
        "ENTJ": "Minister of Magic - Leader with Authority",
        "INFJ": "Glinda - Thoughtful and Devoted Witch",
        "ENFJ": "Fiyero - Charismatic Leader",
        "INTP": "Magic Book Scholar - Analytical and Creative",
        "ENTP": "The Great Wizard of Oz - Witty Innovator",
        "INFP": "Fiyero - Idealistic and Conflicted",
        "ENFP": "Glinda - Optimistic and Energetic Friend",
        "ISTJ": "Emerald City Mayor - Traditional and Reliable",
        "ESTJ": "Head of Ministry - Organized and Responsible",
        "ISFJ": "Elphaba's Mother - Protective and Caring",
        "ESFJ": "Wicked's Companion - Warm and Sociable",
        "ISTP": "Forest Wizard - Practical and Independent",
        "ESTP": "Fiyero - Adventurous and Bold",
        "ISFP": "Forest Fairy - Sensitive and Artistic Soul",
        "ESFP": "Wicked Festival Host - Energetic Performer",
    },
    "Moana": {
        "INTJ": "Moana - Visionary and Independent Explorer",
        "ENTJ": "Maui - Charismatic and Bold Demigod",
        "INFJ": "Gramma Tala - Wise and Mystical Guide",
        "ENFJ": "Moana's Father - Inspiring Leader",
        "INTP": "Hei Hei - Analytical but Distracted",
        "ENTP": "Maui - Witty and Problem-Solving Hero",
        "INFP": "Moana - Dreamy and Determined Adventurer",
        "ENFP": "Pua - Cheerful and Loyal Companion",
        "ISTJ": "Chief Tui - Dutiful and Responsible Leader",
        "ESTJ": "Village Elder - Practical and Assertive",
        "ISFJ": "Gramma Tala - Caring and Devoted Mentor",
        "ESFJ": "Moana's Mother - Supportive and Nurturing",
        "ISTP": "Maui - Practical and Skillful Hero",
        "ESTP": "Moana - Bold and Action-Oriented",
        "ISFP": "Ocean Spirit - Gentle and Free-Spirited",
        "ESFP": "Tamatoa - Showy and Entertaining Crab",
    },
}

# MBTI 네 가지 차원을 표현하는 유사도 데이터
MBTI_DIMENSIONS_SCORES = {
    "INTJ": {"I/E": 0.8, "S/N": 0.9, "T/F": 0.7, "J/P": 0.8},
    "ENTJ": {"I/E": 0.4, "S/N": 0.8, "T/F": 0.8, "J/P": 0.9},
    "INFJ": {"I/E": 0.7, "S/N": 0.9, "T/F": 0.8, "J/P": 0.7},
    "ENFJ": {"I/E": 0.3, "S/N": 0.7, "T/F": 0.9, "J/P": 0.8},
    "INTP": {"I/E": 0.8, "S/N": 0.9, "T/F": 0.6, "J/P": 0.4},
    "ENTP": {"I/E": 0.3, "S/N": 0.8, "T/F": 0.6, "J/P": 0.5},
    "INFP": {"I/E": 0.7, "S/N": 0.9, "T/F": 0.9, "J/P": 0.6},
    "ENFP": {"I/E": 0.4, "S/N": 0.7, "T/F": 0.9, "J/P": 0.4},
    "ISTJ": {"I/E": 0.9, "S/N": 0.6, "T/F": 0.8, "J/P": 0.9},
    "ESTJ": {"I/E": 0.2, "S/N": 0.6, "T/F": 0.8, "J/P": 0.9},
    "ISFJ": {"I/E": 0.9, "S/N": 0.6, "T/F": 0.8, "J/P": 0.8},
    "ESFJ": {"I/E": 0.3, "S/N": 0.5, "T/F": 0.9, "J/P": 0.8},
    "ISTP": {"I/E": 0.8, "S/N": 0.7, "T/F": 0.7, "J/P": 0.4},
    "ESTP": {"I/E": 0.4, "S/N": 0.7, "T/F": 0.7, "J/P": 0.5},
    "ISFP": {"I/E": 0.8, "S/N": 0.7, "T/F": 0.8, "J/P": 0.4},
    "ESFP": {"I/E": 0.3, "S/N": 0.6, "T/F": 0.8, "J/P": 0.3},
}

# # 고정된 감정 분포 데이터
# PREDEFINED_SENTIMENT_DISTRIBUTIONS = {
#     "Wicked": {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
#     "Moana": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
# }

# # 코사인 유사도 계산 함수
# def calculate_similarity(sentiment_distribution):
#     MBTI_SENTIMENT_PROFILES = {
#         "positive": {"positive": 0.6, "neutral": 0.2, "negative": 0.2},
#         "neutral": {"positive": 0.4, "neutral": 0.3, "negative": 0.3},
#         "negative": {"positive": 0.2, "neutral": 0.2, "negative": 0.6},
#     }
#     sentiment_vector = np.array([sentiment_distribution.get(key, 0) for key in ["positive", "neutral", "negative"]])
#     similarity_scores = {
#         category: cosine_similarity([sentiment_vector], [np.array(list(profile.values()))])[0][0]
#         for category, profile in MBTI_SENTIMENT_PROFILES.items()
#     }
#     return similarity_scores


# Streamlit 대시보드
def main():
    st.set_page_config(page_title="Movie MBTI Character Match", layout="wide", page_icon="🎬")

    # 헤더
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 48px;
            font-weight: bold;
            color: #f8f9fa;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="main-header"> Monthly Movie MBTI Character Match </h1>', unsafe_allow_html=True)

    # 영화 선택창
    movie_choice = st.selectbox("Choose a Movie:", ["Wicked", "Moana"], key="movie_choice")

    # MBTI 선택창
    user_mbti = st.selectbox(
        "Select Your MBTI Type:",
        [
            "INTJ", "ENTJ", "INFJ", "ENFJ", "INTP", "ENTP", "INFP", "ENFP",
            "ISTJ", "ESTJ", "ISFJ", "ESFJ", "ISTP", "ESTP", "ISFP", "ESFP"
        ],
        key="user_mbti"
    )

    
    
    # 캐릭터 매칭
    if st.button("Find Your Match!", key="find_match"):
        matched_character = MBTI_CHARACTERS[movie_choice].get(user_mbti, "Unknown Character")
        st.subheader("✨ Your Character Match")
        st.write(f"**Your Character:** {matched_character}")

        # MBTI 네 가지 차원 점수 그래프
        dimension_scores = MBTI_DIMENSIONS_SCORES[user_mbti]
        st.subheader(f"📊 Personality Dimensions for {user_mbti}")
        fig_dimensions = go.Figure(
            data=[
                go.Bar(
                    x=list(dimension_scores.keys()),
                    y=list(dimension_scores.values()),
                    marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
                    text=[f"{v:.2f}" for v in dimension_scores.values()],
                    textposition="outside",
                )
            ]
        )
        fig_dimensions.update_layout(
            title={"text": f"MBTI Dimensions for {user_mbti}", "x": 0.5},
            xaxis_title="Dimension",
            yaxis_title="Score",
            margin=dict(l=400, r=400, t=60, b=40),
            height=400,  # 줄어든 높이
            width=600,   # 줄어든 너비
            showlegend=False,
            font=dict(size=12),  # 폰트 크기 축소
        )
        st.plotly_chart(fig_dimensions, use_container_width=True)

if __name__ == "__main__":
    main()



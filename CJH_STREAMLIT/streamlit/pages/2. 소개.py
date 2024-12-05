import streamlit as st
from nav import inject_custom_navbar

st.set_page_config(page_title="Team Members", layout="wide", initial_sidebar_state="collapsed")
inject_custom_navbar()
# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Background and font settings */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #1C1C1E;
            color: #ff416c;
            font-family: 'Arial', sans-serif;
        }

        /* Subheader styling */
        .subheader {
            color: #FFFFFF;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        /* Role styling */
        .role {
            color: #EBEBF5;
            font-size: 18px;
            margin-top: -5px;
            font-weight: normal;
        }

        /* Bio styling */
        .bio {
            font-size: 16px;
            color: #555;
            margin-top: 10px;
        }

        /* Divider line */
        .divider {
            height: 2px;
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            margin: 15px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Team Members
st.title("🎉 2조")
st.write("어메이징한 팀원을 소개합니다!")

# Team Members Data
team_members = [
    {"name": "팀원 1", "role": "파트", "bio": "추가 정보"},
    {"name": "팀원 2", "role": "파트", "bio": "추가 정보"},
    {"name": "팀원 3", "role": "파트", "bio": "추가 정보"},
    {"name": "팀원 4", "role": "파트", "bio": "추가 정보"},
    {"name": "팀원 5", "role": "파트", "bio": "추가 정보"},
]



for member in team_members:
    with st.container():
        st.markdown(f"<h3 class='subheader'>{member['name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='role'>Role: {member['role']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='bio'>{member['bio']}</p>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# # Example team member cards
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("Team Member 1")
#     st.write("Role: Developer")
# with col2:
#     st.subheader("Team Member 2")
#     st.write("Role: Designer")

if __name__ == "__main__":
    pass
    # show()
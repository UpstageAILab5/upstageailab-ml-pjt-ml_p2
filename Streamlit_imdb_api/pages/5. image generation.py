import streamlit as st

st.title("AI 이미지 생성")
st.write("AI 이미지 생성 페이지 내용을 여기에 추가하세요.")

# Add prompt input
prompt = st.text_input("이미지 생성 프롬프트를 입력하세요:")

if st.button("생성"):
    if prompt:
        st.write("생성된 이미지가 여기에 표시됩니다.")
    else:
        st.warning("이미지 생성 프롬프트를 입력하세요.")
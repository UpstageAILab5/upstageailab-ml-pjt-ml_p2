import streamlit as st
from nav import inject_custom_navbar
from image_generator import ImageGenerator
import huggingface_hub
huggingface_hub.login(token=st.secrets["huggingface"]["token"])

st.set_page_config(
    page_title="AI 이미지 생성",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_navbar()

st.title("AI 이미지 생성")
st.write("AI 이미지 생성 페이지 내용을 여기에 추가하세요.")

# Add prompt input
prompt_input = st.text_input("이미지 생성 프롬프트를 입력하세요:")

# Movie poster generation from review
if st.button("이미지 생성"):
    if prompt_input:

        with st.spinner("이미지를 생성하는 중..."):
            generator = ImageGenerator()
            image_paths = generator.generate_image(prompt_input)
            
            if image_paths:
                for path in image_paths:
                    st.image(path, caption="생성된 이미지", use_column_width=True)
            else:
                st.error("이미지 생성에 실패했습니다. 다시 시도해주세요.")
    else:
        st.warning("이미지 생성 프롬프트를 입력하세요.")

# # General image generation
# if st.button("생성"):
#     if prompt:
#         with st.spinner("이미지를 생성하는 중..."):
#             generator = ImageGenerator()
#             image_paths = generator.generate_image(prompt)
            
#             if image_paths:
#                 for path in image_paths:
#                     st.image(path, caption="Generated Image", use_column_width=True)
#             else:
#                 st.error("이미지 생성에 실패했습니다. 다시 시도해주세요.")
#     else:
#         st.warning("이미지 생성 프롬프트를 입력하세요.")
import streamlit as st

def inject_custom_navbar(page):
    """Inject a custom navigation bar with dynamic active link highlighting."""
    # Custom CSS for styling
    st.markdown(
        f"""
        <style>
            /* Sidebar background with gradient */
            [data-testid="stSidebar"] {{
                background: linear-gradient(90deg, #1E1E2F, #3A3A4F); /* Gradient background */
                color: #FFFFFF; /* White text */
            }}

            /* Sidebar links */
            .sidebar-link {{
                text-decoration: none;
                color: #FFD700; /* Gold text */
                font-size: 18px;
                display: block;
                margin: 10px 0;
                font-weight: bold;
                padding: 5px 15px;
                transition: color 0.3s, background 0.3s;
            }}

            /* Active link styling */
            .sidebar-link-active {{
                color: #FF4B2B !important; /* Red for active links */
                background: rgba(255, 75, 43, 0.2); /* Highlight background */
                border-radius: 10px;
            }}

            /* Hover effect */
            .sidebar-link:hover {{
                color: #FF4B2B;
                background: rgba(255, 75, 43, 0.1);
                border-radius: 5px;
            }}

            /* Footer styling */
            .sidebar-footer {{
                color: #FFD700; /* Gold */
                font-size: 12px;
                text-align: center;
                margin-top: 20px;
                padding-top: 10px;
                border-top: 1px solid #FFD700;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation links with dynamic active class
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://via.placeholder.com/150" alt="Logo" style="width: 120px; border-radius: 10px;">
            <h2 style="color: #FFD700; font-family: 'Arial';">My App</h2>
        </div>

        <ul style="list-style-type: none; padding: 0;">
            <li><a href="/?page=home" class="sidebar-link {'sidebar-link-active' if page == 'home' else ''}">🏠 Home</a></li>
            <li><a href="/?page=main" class="sidebar-link {'sidebar-link-active' if page == 'main' else ''}">🏠 Home</a></li>
            <li><a href="/?page=team" class="sidebar-link {'sidebar-link-active' if page == 'team' else ''}">👨‍👩‍👦 Team</a></li>
            <li><a href="/?page=sentiment" class="sidebar-link {'sidebar-link-active' if page == 'sentiment' else ''}">📊 Sentiment Analysis</a></li>
            <li><a href="/?page=crawling" class="sidebar-link {'sidebar-link-active' if page == 'crawling' else ''}">🔍 Crawling</a></li>
            <li><a href="/?page=image" class="sidebar-link {'sidebar-link-active' if page == 'image' else ''}">🖼️ Image Generation</a></li>
            <li><a href="/?page=prediction" class="sidebar-link {'sidebar-link-active' if page == 'prediction' else ''}">🔮 Prediction</a></li>
        </ul>

        <div class="sidebar-footer">
            <p>© 2024 My App</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Query parameters to manage navigation
query_params = st.query_params
page = query_params.get("page", ["home"])[0]  # Default to 'home'

# Inject the navigation bar with the current page
inject_custom_navbar(page)

# Display content based on the selected page
if page == "home":
    st.title("🏠 Home Page")
    st.write("Welcome to the homepage!")
elif page == "main":
    st.title("🏠 Main Page")
    st.write("Welcome to the main page!")
elif page == "team":
    st.title("👨‍👩‍👦 Team Page")
    st.write("Learn about the team members.")
elif page == "sentiment":
    st.title("📊 Sentiment Analysis Page")
    st.write("Analyze sentiment from reviews.")
elif page == "crawling":
    st.title("🔍 Crawling Page")
    st.write("Extract data from external sources.")
elif page == "image":
    st.title("🖼️ Image Generation Page")
    st.write("Generate AI-powered images.")
elif page == "prediction":
    st.title("🔮 Prediction Page")
    st.write("Use predictive models to forecast outcomes.")

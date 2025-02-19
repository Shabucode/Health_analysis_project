
import streamlit as st

def custom_styling():
    # Apply custom styling
    st.markdown(
        """
        <style>
            body {
                background-color: #E6E6FA;
                color: #4B0082;
                font-family: Arial, sans-serif;
            }
            .title-container {
                background-color: #E6E6FA;
                padding: 10px;
                text-align: center;
                border-radius: 10px;
                position: relative;
            }
            .title {
                background-color: #E6E6FA; 
                font-size: 2.5em;
                font-weight: bold;
            }
            .logo {
                position: absolute;
                top: 45px;
                right: 10px;
                transform: translateY(-50%);
            }
            .logo img {
                width: 90px;
            }
            .stCheckbox label {
                background-color: #E6E6FA; 
                border-radius: 10px; 
                padding: 5px;
            }
            .stTitle {
                background-color: #E6E6FA;
                font-size: 2.5em;
                border-radius: 10px;
                padding: 10x10px;
                center: center
                text-align: center;
            }
            .stButton {
            display: flex;
            justify-content: center; /* Centers horizontally */
            }
            .stButton > button {
                background-color: #333366;  /* Dark Purple */
                border: 2px solid #D8D8D8;
                border-radius: 5px;
                padding: 10px;
                color: white;
                font-size: 16px;
                center: center
            }
            .stButton > button:hover {
                background-color: #6A006A;  /* Lighter Purple */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # padding: 10px;
def render_header():
    st.markdown(
        '''<div class="title-container">
            <div class="logo"><a href="https://img1.wsimg.com/isteam/ip/a35bc262-11ec-4025-9c43-03f883722c6a/REP%20Logo-2023.png" target="_blank">
                <img src="https://img1.wsimg.com/isteam/ip/a35bc262-11ec-4025-9c43-03f883722c6a/REP%20Logo-2023.png" alt="Logo">
            </a></div>
            <div class="title">Literary Review - REP Group Ltd</div>
        </div>''', 
        unsafe_allow_html=True
    )
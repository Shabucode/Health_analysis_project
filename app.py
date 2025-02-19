
#streamlit application
import streamlit as st
import pandas as pd
import random
import os
import tempfile
#import defined modules
from modules.new_extract_sentences import *
from modules.domain_analysis import *
from modules.disorder_analysis import *
from modules.benefit_analysis import *
from modules.book_fe import *


#import from utils 
from utils.session_state import initialize_session_state
from utils.ui import custom_styling, render_header
from utils.design import story_analysis_design
from utils.logger import logger

# # Initialize session state variables
initialize_session_state()
# Set page configuration
st.set_page_config(page_title="Literary Review", page_icon="📖", layout="wide")
custom_styling() #Applying custom styles
render_header() #Rendering the header

def main():

    st.write("") # – for a simple empty line.
    st.write("")
    st.write("")
    tab = st.selectbox("Choose a tab", ["Story Literature Analysis", "Domain_Disorder_Benefit_finding_analysis", "Story Publishing-Readiness Analysis"])

    st.write("")
    container = st.container()

    with container:
        if tab == "Story Literature Analysis":
            story_analysis_design()
            
        elif tab == "Domain_Disorder_Benefit_finding_analysis":
            domain_design_analysis()
            disorder_design()
            benefit_finding_design()
            
        elif tab == "Story Publishing-Readiness Analysis":
            # Function to append new rows to an existing Excel file
            publish_readiness_analysis()
            
        else:
            st.write("Please select any tabs")

if __name__ == "__main__":
    main()


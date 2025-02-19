
import streamlit as st
from utils.functions_decor import *
from utils.logger import logger
import pandas as pd
import matplotlib.pyplot as plt

def domain(data):
    # Load data from Excel
    # data = pd.read_excel("domain_scores.xlsx")

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define color based on the threshold of 0.5
    df['Color'] = df['Score'].apply(lambda x: '#FF7F7F' if x > 0.5 else '#FFFF99')  # Light red (#FF7F7F) and light yellow (#FFFF99)

    # Filter the domains where the score is above 0.5
    impactful_domains = df[df['Score'] > 0.5]

    # Plotting the bar chart and the text output side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))  # Create 1 row, 2 columns for subplots

    # --- Left subplot: Bar chart ---
    bars = ax1.bar(df['Domain'], df['Score'], color=df['Color'])

    # Adding threshold line (light blue)
    ax1.axhline(y=0.5, color='#ADD8E6', linestyle='--', label='Threshold (0.5)')  # Light blue (#ADD8E6)

    # Add text labels on top of the bars
    for bar in bars:
        yval = bar.get_height()  # Get the height of each bar (score)
        ax1.text(bar.get_x() + bar.get_width()/2, yval,  # Positioning the text slightly above the bar
                f'{yval:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    # Adding title and labels for the bar chart
    ax1.set_title('Impact of Story in Different Domains')
    ax1.set_xlabel('Domain')
    ax1.set_ylabel('Score')
    ax1.legend()

    # --- Right subplot: Text output ---
    if len(impactful_domains) >= 2:
        ax2.text(0.1, 1, "The story is impactful in the following domains and their scores:", fontsize=12, ha='left')
        for i, (index, row) in enumerate(impactful_domains.iterrows()):
            ax2.text(0.1, 0.9 - 0.1*i, f"Domain: {row['Domain']}, Score: {row['Score']:.2f}", fontsize=10, ha='left')
    else:
        ax2.text(0.1, 0.9, "The story is not impactful with respect to at least two domains.", fontsize=12, ha='left')

    # Removing axes for the text subplot
    ax2.axis('off')

    # Show the plot with the two subplots side by side
    plt.tight_layout()
    # plt.show()
    return fig

def domain_design_analysis():
    st.write("")
    st.write("")
    st.markdown('<h3 class="stTitle">Domain Analysis</h3>', unsafe_allow_html=True)
    # st.subheader("Domain Analysis")
    st.write("")
    st.write("")
    domain_file = st.file_uploader("Upload a domain scores excel file", accept_multiple_files = False)
    st.write("")
    domain_ana = st.button("Start Domain Analyis")
    st.write("")
    if domain_ana:
        if domain_file is not None:
            try:
                # Read the uploaded Excel file
                data = pd.read_excel(domain_file)
                # Check if required columns are present
                required_columns = {"Domain", "Score"}
                if not required_columns.issubset(data.columns):
                    st.error("Uploaded file is missing required columns: 'Domain' and 'Score'. Please upload a valid file.")
                    logger.error(f"Invalid file uploaded. Missing columns. Found columns: {list(data.columns)}")
                else:
                    # Store the data in session state
                    st.session_state.domain_data = data
                # domain analysis
                fig = domain(data)
                st.session_state.domain_fig = fig
                # st.subheader("Domain Analysis")
                st.write("")
                st.markdown('<h3 class="stTitle">Domain Analysis</h3>', unsafe_allow_html=True)
                st.write("")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while processing the domain file")
                logger.error(f"Error details: {e}", exc_info=True)
    else:
        # If a file was processed earlier, display the saved figure
        if st.session_state.domain_fig:
            # st.subheader("Domain Analysis")
            st.write("")
            st.markdown('<h3 class="stTitle">Domain Analysis</h3>', unsafe_allow_html=True)
            st.write("")
            st.pyplot(st.session_state.domain_fig)
        else:
            st.info("Upload a file to start the analysis.")
        
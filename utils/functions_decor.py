from modules.readability_check import *
from modules.text_statistics import *
from modules.grammarcheck import *
from modules.vocab import *
from modules.coher import *
from modules.emotion_rep_inc import *
from modules.coping_mech import *
from modules.educational_value import *
from modules.rhetorical_devices import *
import streamlit as st
import pandas as pd

def read_analysis(text):
    try:
        result = analyze_readability(text)

        # Ensure function returns exactly two values
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Unexpected output format from readability analysis.")

        df_readability, summary_text = result

        st.subheader("Readability Analysis")
        st.dataframe(df_readability)
        st.markdown(summary_text, unsafe_allow_html=True)

    except ValueError as ve:
        if "SMOG requires 30 sentences" in str(ve):
            st.error("The readability analysis requires at least 30 sentences. Please provide a longer text.")
        elif "too many values to unpack" in str(ve):
            st.error("Unexpected error in readability analysis. Please check if the input text is valid.")
        else:
            st.error(f"Readability Analysis Error: {str(ve)}")
    
    except Exception as e:
        st.error("An unexpected error occurred while analyzing readability. Please try again with a different text.")

def text_statstcs(text):
    text_stats_df = analyze_text(text) ## Analyze text statistics
    st.subheader("Text Statistics")
    st.dataframe(text_stats_df)
    return 

        # def read_analysis(text):
        #     try:
        #         df_readability, summary_text = analyze_readability(text)
        #         st.subheader("Readability Analysis")
        #         st.dataframe(df_readability)
        #         st.markdown(summary_text, unsafe_allow_html=True)
        #         return 
        #     except Exception as e:
        #         st.error(f"An error occurred while processing the file: {str(e)}")
        #         # logger.error(f"Error details: {e}", exc_info=True)
        #         return "Error"
       
def vocab_analysis(text):
    words = preprocess_text(text)
    # Get the vocabulary
    vocabulary = get_vocabulary(words)
    
    # Calculate TTR for the vocabulary
    ttr = type_token_ratio(words)
    
    # Print feedback based on the TTR
    vocab_result = (words, vocabulary, ttr)
    return vocab_result

def print_ttr_feedback(ttr):
    if ttr < 0.05:
        st.write("This Type-Token Ratio (TTR) is very low. The text may have an overuse of certain words.")
    elif 0.05 <= ttr < 0.15:
        st.write("This Type-Token Ratio (TTR) is typical for narrative or fiction. It suggests a moderate variety of vocabulary.")
    elif 0.15 <= ttr < 0.30:
        st.write("This Type-Token Ratio (TTR) is typical for formal or complex documents, indicating a richer vocabulary.")
    else:
        st.write("This Type-Token Ratio (TTR) is high, suggesting a very diverse vocabulary, typical for academic or sophisticated writing.")

def coher(pdf_path):
    st.subheader("Coherence Analysis")
    overall_coherence, coher_summary = coher_function_call(pdf_path)
    st.write(f"Overall Coherence Score: {overall_coherence:.2f}")
    st.write(coher_summary)
    return 

def emotion(senten_c):
    if len(senten_c)>0:
        Rep_Inc_Emo_Intensity = emotional_intensity(senten_c)
        st.subheader("Empathy and Emotional Intensity Analysis")
        st.write(Rep_Inc_Emo_Intensity)    
        st.session_state.emotion_result = Rep_Inc_Emo_Intensity
    else:
        st.write("No text provided. Please upload a PDF.")  
    return

def coping(senten_c):
    if len(senten_c)>0:
        coping_summary, coping_df = coping_classification(senten_c)
        st.subheader("Coping Mechanism Analysis")
        st.write(coping_summary)
        st.dataframe(coping_df)   
        st.session_state.coping_result = (coping_summary, coping_df)
    else:
        st.write("No text provided. Please upload a PDF.")  
    return

def edu_val(senten_c):
    if len(senten_c)>0:
        edu_val_summary, edu_val_df = educational_value_category(senten_c)
        st.subheader("Analysis of Educational Value in the Story:")
        st.write(edu_val_summary)
        st.dataframe(edu_val_df)   
        st.session_state.edu_val_result = (edu_val_summary, edu_val_df)
    else:
        st.write("No text provided. Please upload a PDF.")  
    return

def rhe_dev(senten_c):
    if len(senten_c)>0:
        df_rd, rhe_dev_summary, rhe_dev_df = rhetorical_classification(senten_c)
        st.subheader("Analysis of Rhetorical Device in the Story:")
        st.dataframe(df_rd) 
        st.write(rhe_dev_summary)
        st.dataframe(rhe_dev_df)  
        st.session_state.rhe_dev_result = (df_rd, rhe_dev_summary, rhe_dev_df) 
    else:
        st.write("No text provided. Please upload a PDF.")  
    return

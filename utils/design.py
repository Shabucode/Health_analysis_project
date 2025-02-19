

import streamlit as st
# from utils.session_state import initialize_session_state
from utils.functions_decor import *
from utils.logger import logger

# Initialize session state variables
# initialize_session_state()

def story_analysis_design():
    # st.write("Content for Tab 1")
        # st.title("Story Analysis For Publication Readiness")
        st.markdown('<h3 class="stTitle">Story Analysis For Publication Readiness</h3>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        # Allow the user to upload a new PDF
        pdf_path = st.file_uploader("Upload a PDF story", accept_multiple_files=False)
        
        # First row: "Select all" checkbox
        select_all = st.checkbox("Select all")
        # Second row: First group of checkboxes (horizontal layout)
        col1, col2, col3 = st.columns(3)
        with col1:
            text_stats = st.checkbox("Text Statistics")
        with col2:
            read = st.checkbox("Readability")
        with col3:
            gram_chk = st.checkbox("Grammar Check")

        # Third row: Second group of checkboxes (horizontal layout)
        col4, col5, col6 = st.columns(3)
        with col4:
            vocab = st.checkbox("Vocabulary Analysis")
        with col5:
            coher_ana = st.checkbox("Coherence Analysis")
        with col6:
            emp = st.checkbox("Empathy")

        # Fourth row: Remaining checkboxes (horizontal layout)
        col7, col8, col9 = st.columns(3)
        with col7:
            cope = st.checkbox("Coping Strategy Analysis")
        with col8:
            ed_val = st.checkbox("Educational Value")
        with col9:
            rh_dv = st.checkbox("Rhetorical Device Analysis")
        
        st.write("")
        st.write("")
        evaluate = st.button("Start Literary Review")  

        if evaluate:
            # check if pdf is uploaded
            if pdf_path:
                try:
                    #text statistics for the pdf story
                    print("Extracting text from pdf")
                    text = pdf_extract_text(pdf_path) ## Read and extract text from the uploaded PDF file
                    print("sentencing from text")
                    senten_c = clean_text(text)
                    st.session_state.senten_c = senten_c
                    # senten_c = clean_sentences(text)
                    # print("sentenced")
                    
                    if select_all or text_stats:
                        text_statstcs(text)
                        
                    if select_all or read:
                        read_analysis(text)
                      
                    if select_all or gram_chk:
                        grammar_score = process_pdf_for_grammar(text)
                        st.write(f"Overall Grammar Correctness Score: {round(grammar_score, 2)}%")
                        
                    if select_all or vocab:   
                        words, vocabulary, ttr = vocab_analysis(text)
                        st.session_state.words = words
                        st.session_state.vocabulary = vocabulary
                        st.session_state.ttr = ttr
                        # Display results
                        st.subheader("Vocabulary Analysis Results:")
                        st.write(f"""
                        **Total Word Count:** {len(words)}  
                        **Vocabulary Length:** {len(vocabulary)}  
                        **Type-Token Ratio (TTR):** {ttr:.2f}
                        """)
                        vocab_button = st.button("Show Vocabulary")
                        if vocab_button:
                            st.write(f"**Vocabulary:** {', '.join(vocabulary)}")
                        # st.write(f"**Vocabulary:** {', '.join(vocabulary)}")
                        
                    if select_all or coher_ana:
                        coher(pdf_path)
                        
                    if select_all or emp:
                        emotion(senten_c) 
                            
                    if select_all or cope:
                        coping(senten_c)
                        
                    if select_all or ed_val:
                        if len(senten_c)>0:
                            edu_val(senten_c)
                      
                    if select_all or rh_dv:
                        rhe_dev(senten_c) 

                    else:
                        st.write("Please select features from the list.")
                    # Store the results in session state to persist across interactions
                    # st.session_state.previous_results = results  # Save the previous results
                except Exception as e:
                    st.error(f"An error occurred while processing the file")
                    logger.error(f"Error details: {e}", exc_info=True)
            else:
                
                st.info("Please upload the story pdf file.")
                
            # logger.info("Streamlit app started.")
        else:

            if st.session_state.get("words") is not None:
                # st.session_state.vocab_result
                words = st.session_state.words
                vocabulary = st.session_state.vocabulary
                ttr = st.session_state.ttr
                st.subheader("Vocabulary Analysis Results:")
                st.write(f"""
                        **Total Word Count:** {len(words)}  
                        **Vocabulary Length:** {len(vocabulary)}  
                        **Type-Token Ratio (TTR):** {ttr:.2f}
                        """)
                vocab_button = st.button("Show Vocabulary")
                if vocab_button:
                    st.write(f"**Vocabulary:** {', '.join(st.session_state.vocabulary)}")  # optional

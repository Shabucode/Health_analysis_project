import streamlit as st

def initialize_session_state():
    default_values = {
        "domain_data": None,
        "domain_fig": None,
        "disorder_file": None,
        "anthology_file": None,
        "story_id": "",
        "selected_disorder": "Select a disorder",
        "proportion": None,
        "story_id_focus": None,
        "vocab_result": None,
        "words": None
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# if "pdf_path" not in st.session_state:
#     st.session_state.pdf_path = None
# if "text_stats_df" not in st.session_state:
#     st.session_state.text_stats_df = None
# if "st.session_state.readability_result" not in st.session_state:
#     st.session_state.readability_result = None
# if "st.session_state.overall_coherence" not in st.session_state:
#     st.session_state.overall_coherence = None
# if "st.session_state.emotion_result" not in st.session_state:
#     st.session_state.emotion_result = None
# if "st.session_state.coping_summary" not in st.session_state:
#     st.session_state.coping_summary = None
# if "st.session_state.edu_val_summary" not in st.session_state:
#     st.session_state.edu_val_summary = None
# if "st.session_state.rhe_dev_summary" not in st.session_state:
#     st.session_state.rhe_dev_summary = None
# if "st.session_state.grammar_score" not in st.session_state:
#     st.session_state.grammar_score = None


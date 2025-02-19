import pandas as pd
from transformers import pipeline
import spacy
from utils.logger import logger
import streamlit as st

def ben(ben_path):
    # file_path = "/content/benefit_data-2024-12-17 (1).csv"
    bfwf = pd.read_csv(ben_path)
    # bf = pd.read_csv(file_path)
    bf = bfwf[bfwf['probability'] > 0.25]
    proportion = round(len(bfwf)/len(bf), 2)
    # print("Benefit-Finding Analysis of the story\n")
    # print("Total number of sentences in the story are : ", len(bfwf))
    # print("Total number of sentences relevant to benefit finding are :", len(bf))
    # print("Proportion of benefit finding in the story is {:.2f}".format(proportion))
    summary = f"""Benefit-Finding Analysis of the story\n
    Total number of sentences in the story: {len(bfwf)}\n
    Total number of sentences relevant to benefit finding: {len(bf)}\n
    Proportion of benefit finding in the story: {proportion:.2f}"""
    return summary, proportion


def categorize_bf(proportion):
    if proportion < 5:
        proportion_summary = f"The story highlights very few benefits alongside challenges with the benefit-finding proportion {proportion}%"
    elif 5 <= proportion < 10:
        proportion_summary = f"The story highlights moderate benefits alongside challenges with the benefit finding proportion {proportion}%"
    elif 10 <= proportion < 20:
        proportion_summary = f"The story highlights noticeable benefits alongside challenges with the benefit finding proportion {proportion}%"
    elif 20 <= proportion < 50:
        proportion_summary = f"The story highlights significant benefits alongside challenges with the benefit finding proportion {proportion}%"
    else:
        proportion_summary =f"The story highlights high level benefits alongside challenges with the benefit finding proportion {proportion}%"
    return proportion_summary
# categorize_bf(proportion)


# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Define possible labels
labels = ["opinion-based coping mechanism", "fact-based coping mechanism"]
# Function to classify sentences
def classify_subjectivity(sentence):
    result = classifier(sentence, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]  # Returning label and confidence score
def cope_category(cope_path):
    # path = "copings_mechanisms.xlsx"
    df2=pd.read_excel(cope_path)
    dfac2 = df2[df2["Maladaptive Coping"].isna() | (df2["Maladaptive Coping"] == "")]
    dfac2 = dfac2.drop(columns=['Maladaptive Coping'])
    # Apply classification to the 'Sentence' column in the DataFrame
    dfac2[['Subjectivity_Label', 'Model_Confidence']] = dfac2['Sentence'].apply(lambda x: pd.Series(classify_subjectivity(x)))
    # Filter the DataFrame for objective sentences with Subjectivity_Confidence > 0.8
    dfac2 = dfac2[(dfac2['Subjectivity_Label'] == 'fact-based coping mechanism') & (dfac2['Model_Confidence'] > 0.6)]
    return dfac2

# Function to extract named entities from the sentence
def extract_entities(sentence):
    # Load spaCy's pre-trained English NER model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)  # Process sentence with spaCy
    entities = [ent.text for ent in doc.ents]  # Extract entities as a list of texts
    return entities

# Function to find support references
def find_support_references(sentences, support_terms):
    support_references = []
    for sentence in sentences:
        found = {}
        for category, terms in support_terms.items():
            if any(term in sentence for term in terms):
                found[category] = True
        if found:
            support_references.append({"sentence": sentence, "categories": list(found.keys())})
    return support_references

def ent_supp(dfac2):
    # Support terms dictionary
    support_terms = {
        "helplines": ["hotline", "helpline", "emergency number", "call center", "crisis hotline", "suicide prevention"],
        "organizations": ["NHS", "Red Cross", "WHO", "mental health organization", "Mental Health America", "SAMHSA"],
        "books": ["book", "manual", "memoir", "guide", "e-book", "novel", "reference"],
        "therapy": ["therapist", "counselor", "therapy", "psychologist", "psychiatrist", "psychotherapist", "life coach"],
        "tools": ["app", "website", "platform", "software", "mobile app", "online resource"],
        "community_support": ["support group", "local group", "community group", "peer support", "online community", "social group"]
    }
    # Apply the function to the 'Sentence' column
    dfac2['Entities'] = dfac2['Sentence'].apply(extract_entities)
    # Apply the function to the 'Sentence' column
    dfac2['Support References'] = dfac2['Sentence'].apply(lambda x: find_support_references([x], support_terms))
    # Display the updated DataFrame
    # print(dfac2[['Sentence', 'Entities']])
    return dfac2

# Filter the DataFrame for objective sentences with Subjectivity_Confidence > 0.8
# filtered_df = dfac2[(dfac2['Subjectivity_Label'] == 'fact-based coping mechanism') & (dfac2['Model_Confidence'] > 0.6)]
# dfac2 = cope_category(cope_path)
# dfac2= ent_supp(dfac2)

def benefit_finding_design():
    # st.subheader("Benefit-Finding Analysis")
    st.write("")
    st.write("")
    st.write("")
    st.markdown('<h3 class="stTitle">Benefit-Finding Analysis</h3>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    benefit_file = st.file_uploader("Upload a benefit-finding analysis excel file", accept_multiple_files = False)
    st.write("")
    benefit_button = st.button("Start Benefit-Finding Analysis")
    st.write("")
    if benefit_button:
        if benefit_file is not None:
            try:
                summary, proportion = ben(benefit_file)
                proportion_summary = categorize_bf(proportion)
                st.write(summary)
                st.write(proportion_summary)
            except Exception as e:
                st.error(f"An error occurred while processing the benefit-finding file")
                logger.error(f"Error details: {e}", exc_info=True)
        else:
            st.info("Upload a file to start the analysis.")
    # summary, proportion = ben(benefit_file)
    # proportion_summary = categorize_bf(proportion)
    # st.write(summary)
    # st.write(proportion_summary)
    # st.subheader("Benefit-Finding Analysis Part 2 - Fact-Based Coping Mechanism Analysis")
    st.write("")
    st.markdown('<h3 class="stTitle">Benefit-Finding Analysis Part 2 - Fact-Based Coping Mechanism Analysis</h3>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    cope_check_file = st.file_uploader("Upload the coping mechanism analysis excel file", accept_multiple_files = False)
    st.write("")
    cope_check_file_button = st.button("Start Fact-Based Coping Mechanism Analysis")
    st.write("")
    if cope_check_file_button:
        if cope_check_file is not None:
            try:
                dfac2 = cope_category(cope_check_file)
                dfac2= ent_supp(dfac2)
                st.dataframe(dfac2)
                # cope_check = cope_check(cope_check_file)
                # st.write(cope_check)
            except Exception as e:
                st.error(f"An error occurred while processing the coping mechanism analysis file")
                logger.error(f"Error details: {e}", exc_info=True)
        else:
            st.info("Upload a file to start the analysis.")


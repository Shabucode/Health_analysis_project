import os
import pandas as pd
import pdfplumber
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM
# from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from io import BytesIO
import re
import logging
from utils.logger import logger
from pydantic import ValidationError
import warnings
from modules.bookclsfr import classify_story, load_model  # Import the classifier function

# HuggingFace BGE Embeddings Initialization
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
embeddings = hf

# Initialize the LLM model
# llm = Ollama(model="llama3")
llm = OllamaLLM(model="llama3")


def process_pdf(uploaded_file):
    """Extract text from a PDF file using pdfplumber."""
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        documents = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                document = Document(page_content=text, metadata={"page": page_num + 1})
                documents.append(document)
            else:
                print(f"Warning: No text found on page {page.page_number}")
    return documents


# def create_or_load_vectorstore(documents, vectorstore_dir):
#     """Create or load a FAISS vectorstore for the given documents."""
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     if not os.path.exists(vectorstore_dir):
#         os.makedirs(vectorstore_dir)

#     vectorstore_path = os.path.join(vectorstore_dir, "index.faiss")

#     if not os.path.exists(vectorstore_path):
#         vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
#         vectorstore.save_local(vectorstore_dir)
#     else:
#         vectorstore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)

#     return vectorstore


def label_story_criteria(uploaded_file):
    
    """Generate criteria labels for a story and return as a DataFrame row."""
   
    # Process the PDF to extract full content
    documents = process_pdf(uploaded_file)
    # full_story_content = "\n".join([doc.page_content for doc in documents])
    criteria = [
    "Storytelling", "Storyline", "Writing quality", "Tone", "Pacing", "Heartwarming", "Insight", "Honesty",
    "Empathy", "Writing style", "Enthralling", "Inspirational story", "Thought provoking", "Readability", 
    "Warmth", "Sadness", "Comfort level", "Mental health issue", "Health", "Romance", "Humor", 
    "Educational value", "Value", "Boredom"
    ]

    # Catch warnings about empty pages
    warnings.filterwarnings("ignore", message="No text found on page")
    # context = "\n".join([doc.page_content for doc in retrieved_docs])
    full_story_content = "\n".join([doc.page_content for doc in documents])
    # warnings.filterwarnings("ignore", message="No text found on page")
    template = f"""
    You are a book critic. Read the following story and evaluate it on the following criteria and provide the score which should be either 0 or 1 only: 
    - "Storytelling"
    - "Storyline"
    - "Writing quality"
    - "Tone"
    - "Pacing"
    - "Heartwarming"
    - "Insight"
    - "Honesty"
    - "Empathy"
    - "Writing style"
    - "Enthralling"
    - "Inspirational story"
    - "Thought provoking"
    - "Readability"
    - "Warmth"
    - "Sadness"
    - "Comfort level"
    - "Mental health issue"
    - "Health"
    - "Romance"
    - "Humor"
    - "Educational value"
    - "Value"
    - "Boredom"

    For each of these criteria, provide a score from 0 to 1 based on how strongly the story demonstrates each of these qualities. A score of 0 means the story does not exhibit that quality, and a score of 1 means the story demonstrates the quality strongly.

    Story:
    {full_story_content}

    Please strictly output only the scores for each criterion in the following format in a dictionary:

    "Storytelling": <0 to 4>
    "Storyline": <0 to 4>
    "Writing quality": <0 to 4>
    "Tone": <0 to 4>
    "Pacing": <0 to 4>
    "Heartwarming": <0 to 4>
    "Insight": <0 to 4>
    "Honesty": <0 to 4>
    "Empathy": <0 to 4>
    "Writing style": <0 to 4>
    "Enthralling": <0 to 4>
    "Inspirational story": <0 to 4>
    "Thought provoking": <0 to 4>
    "Readability": <0 to 4> 
    "Warmth": <0 to 4>
    "Sadness": <0 to 4>
    "Comfort level": <0 to 4>
    "Mental health issue": <0 to 4>
    "Health": <0 to 4>
    "Romance": <0 to 4>
    "Humor": <0 to 4>
    "Educational value": <0 to 4>
    "Value": <0 to 4>
    "Boredom": <0 to 4>
    """

    
     # Try-except block for creating the PromptTemplate instance
    try:
        prompt = PromptTemplate(input_variables=["full_story_content"], template=template)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return {"Book_ID": uploaded_file.name, "Error": f"Validation error: {e}"}
    
    prompt = PromptTemplate(input_variables=["full_story_content"], template=template)
    chain = prompt | llm
    result = chain.invoke({"context": full_story_content})
    print(result)
    # st.write(result)
    def preprocess_result(result):
        """Replace 'N/A' with '0' and '0.5' with '1' in the result string."""
        result = re.sub(r"N/A", "0", result)  # Replace N/A with 0
        # result = re.sub(r"0\.5", "0", result)  # Replace 0.5 with 1
        # Replace values with a lambda function for conditional substitution
        # result = re.sub(
        #     r"(\d+\.\d+)",  # Regex to match decimal numbers
        #     lambda x: "1" if float(x.group(1)) >= 0.8 else "0",
        #     result)
        return result

    try:
        # row = parse_scores(uploaded_file, result, criteria)
        if not isinstance(result, str):
            raise ValueError("Expected 'result' to be a string")
        if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
            raise ValueError("Expected 'criteria' to be a list of strings")
        
        # Preprocess the result to replace N/A and 0.5
        result = preprocess_result(result)

        scores = {}
        # Search for each criterion in the result dictionary
        for criterion in criteria:
            # Use a case-insensitive search pattern to match the criterion in the dictionary
            # Also, this pattern ignores anything after the score (including explanations in parentheses)
            pattern = rf'"{re.escape(criterion)}"\s*:\s*([0-9\.]+)(?:\s*\(.*\))?'  # Matches the number or float value after the criterion
            matches = re.findall(pattern, result, re.IGNORECASE)  # Case-insensitive search
            if matches:
                value = matches[-1]  # Use the last match (in case there are duplicates)
                scores[criterion] = float(value) if '.' in value else int(value)
            else:
                # If no match or N/A is found, set it as 0
                scores[criterion] = 0
                logging.warning(f"No match found for criterion '{criterion}' or found N/A, set to 0")

        row = {"Book_ID": uploaded_file.name}  # Add the file ID as a string, not as an object
        row.update(scores)  # Add the scores for each criterion
        return row
    
    except Exception as e:
        logging.error(f"Error parsing scores: {e}")
        return {"Book_ID": uploaded_file, "Error": str(e)}
    
# # Function to append new rows to an existing Excel file
def append_to_excel(existing_excel, new_data, output_file):
            # Load the existing Excel file
            with pd.ExcelFile(existing_excel) as xl:
                # Read the existing data from the Excel sheet
                existing_df = xl.parse(xl.sheet_names[0])
                
            # Append the new rows to the existing DataFrame
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)

            # Save the updated DataFrame back to the Excel file
            updated_df.to_excel(output_file, index=False)
            return output_file
def publish_readiness_analysis():
    # Streamlit App UI
    # st.title("Book Critic - Story Features Evaluation")
    st.write("")
    st.markdown('<h3 class="stTitle">Book Critic - Story Features Evaluation</h3>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    # Upload files
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Upload an existing Excel file to update
    existing_excel = st.file_uploader("Upload existing Excel file (optional)", type=["xlsx"])
    col1, col2 = st.columns(2)
    with col1:
        st.write("")
        evaluate_button = st.button("Evaluate Story For Features")

    if evaluate_button:
        if uploaded_files:
            rows = []
            criteria = [
            "Storytelling", "Storyline", "Writing quality", "Tone", "Pacing", "Heartwarming", "Insight", "Honesty",
            "Empathy", "Writing style", "Enthralling", "Inspirational story", "Thought provoking", "Readability", 
            "Warmth", "Sadness", "Comfort level", "Mental health issue", "Health", "Romance", "Humor", 
            "Educational value", "Value", "Boredom"
            ]
            columns = ["Book_ID"] + criteria # Add Book_ID as the first column
            for uploaded_file in uploaded_files:
                st.write(f"Processing {uploaded_file.name}...")
                row = label_story_criteria(uploaded_file)
                if row:
                    rows.append(row)
            
            if rows:
                # Create DataFrame and display
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df)

            # If an existing Excel file is uploaded, append new data
            if existing_excel:
                # Create a temporary output file path
                output_file = "updated_books.xlsx"
                output_file = append_to_excel(existing_excel, df, output_file)
                st.download_button(
                    label="Download Updated Excel File",
                    data=open(output_file, "rb").read(),
                    file_name="updated_books.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # If no existing file, save the new data as a new file
                df.to_excel("new_books.xlsx", index=False)
                st.download_button(
                    label="Download New Excel File",
                    data=open("new_books.xlsx", "rb").read(),
                    file_name="new_books.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("Please upload PDF files.")
    with col2:
        st.write("")
        publish_button = st.button("Is this book ready for publishing")

    if publish_button:
        if uploaded_files:
            rows = []
            criteria = [
                "Storytelling", "Storyline", "Writing quality", "Tone", "Pacing", "Heartwarming", "Insight", "Honesty",
                "Empathy", "Writing style", "Enthralling", "Inspirational story", "Thought provoking", "Readability",
                "Warmth", "Sadness", "Comfort level", "Mental health issue", "Health", "Romance", "Humor",
                "Educational value", "Value", "Boredom"
            ]
            columns = ["Book_ID"] + criteria  # Initial columns excluding "Publish"

            for uploaded_file in uploaded_files:
                st.write(f"Processing {uploaded_file.name}...")
                row = label_story_criteria(uploaded_file)  # Extract story features
                
                if row:
                    rows.append(row)

            if rows:
                # Create a DataFrame with the extracted rows
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df)

                # Add "Publish" column
                df["Publish"] = ""  # Placeholder for predictions

                # Prepare features for classification without permanently dropping 'Book_ID'
                feature_columns = [col for col in df.columns if col not in ["Book_ID", "Publish"]]

                # Load the pre-trained model
                model_filepath = "LR.pkl"  # Path to the pre-trained model
                LR_model = load_model(model_filepath)

                # Classify the story and update the "Publish" column
                for index, row in df.iterrows():
                    # Convert row to a DataFrame to match the model's expected input
                    X_new = pd.DataFrame([row[feature_columns]], columns=feature_columns)
                    predictions, suggestions = classify_story(LR_model, X_new)

                    # Update the "Publish" column in the DataFrame
                    df.at[index, "Publish"] = predictions

                    # Display suggestions
                    st.write(f"Suggestions for Book ID {row['Book_ID']}: {suggestions}")

                # Append to existing Excel file or create a new one
                if existing_excel:
                    # Create a temporary output file path
                    output_file = "updated_books.xlsx"
                    output_file = append_to_excel(existing_excel, df, output_file)
                    st.download_button(
                        label="Download Updated Excel File",
                        data=open(output_file, "rb").read(),
                        file_name="updated_books.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    # Save as a new Excel file
                    df.to_excel("new_books.xlsx", index=False)
                    st.download_button(
                        label="Download New Excel File",
                        data=open("new_books.xlsx", "rb").read(),
                        file_name="new_books.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.error("Please upload PDF files.")   

# def append_to_excel(existing_excel, new_data, output_file):
#     # Load the existing Excel file
#     with pd.ExcelFile(existing_excel) as xl:
#         # Read the existing data from the Excel sheet
#         existing_df = xl.parse(xl.sheet_names[0])
        
#     # Append the new rows to the existing DataFrame
#     updated_df = pd.concat([existing_df, new_data], ignore_index=True)

#     # Save the updated DataFrame back to the Excel file
#     updated_df.to_excel(output_file, index=False)
#     return output_file


# # Streamlit App UI
# st.title("Book Critic - Story Features Evaluation")

# # Upload files
# uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# # Upload an existing Excel file to update
# existing_excel = st.file_uploader("Upload existing Excel file (optional)", type=["xlsx"])

# if st.button("Evaluate Story For Features"):
#     if uploaded_files:
#         rows = []
#         criteria = [
#         "Storytelling", "Storyline", "Writing quality", "Tone", "Pacing", "Heartwarming", "Insight", "Honesty",
#         "Empathy", "Writing style", "Enthralling", "Inspirational story", "Thought provoking", "Readability", 
#         "Warmth", "Sadness", "Comfort level", "Mental health issue", "Health", "Romance", "Humor", 
#         "Educational value", "Value", "Boredom"
#         ]
#         columns = ["Book_ID"] + criteria # Add Book_ID as the first column
#         for uploaded_file in uploaded_files:
#             st.write(f"Processing {uploaded_file.name}...")
#             row = label_story_criteria(uploaded_file)
#             if row:
#                 rows.append(row)
        
#         if rows:
#             # Create DataFrame and display
#             df = pd.DataFrame(rows, columns=columns)
#             st.dataframe(df)

#         # If an existing Excel file is uploaded, append new data
#         if existing_excel:
#             # Create a temporary output file path
#             output_file = "updated_books.xlsx"
#             output_file = append_to_excel(existing_excel, df, output_file)
#             st.download_button(
#                 label="Download Updated Excel File",
#                 data=open(output_file, "rb").read(),
#                 file_name="updated_books.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#         else:
#             # If no existing file, save the new data as a new file
#             df.to_excel("new_books.xlsx", index=False)
#             st.download_button(
#                 label="Download New Excel File",
#                 data=open("new_books.xlsx", "rb").read(),
#                 file_name="new_books.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#     else:
#         st.error("Please upload PDF files.")





            
            

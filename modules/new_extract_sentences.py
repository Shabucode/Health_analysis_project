import pandas as pd
import re
import string
# import pdfplumber

# def extract_text_from_pdf(pdf_path):
#     """
#     Extracts text from a PDF file.
#     """
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             return ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
#     except Exception as e:
#         print(f"Error extracting text from the PDF: {e}")
#         return ""

def clean_text(text):
    """
    Cleans text by splitting it into sentences, removing punctuation, and extra spaces.
    """
    def remove_extra_spaces(sentence):
        """Remove leading/trailing spaces and replace multiple spaces with a single space."""
        return re.sub(r'\s+', ' ', sentence.strip())

    # Split text into sentences using regex
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    sentences = re.split(sentence_pattern, text)

    # Remove punctuation and extra spaces from each sentence
    cleaned_sentences = [
        remove_extra_spaces(sentence.translate(str.maketrans('', '', string.punctuation)))
        for sentence in sentences
    ]
    cleaned_sentences = filter_sentences_by_length(cleaned_sentences)

    return cleaned_sentences

def filter_sentences_by_length(sentences, min_words=3):
    """
    Filters sentences that contain more than `min_words` words.
    """
    return [sentence for sentence in sentences if len(sentence.split()) > min_words]

def process_text_and_return_dataframe(cleaned_sentences) -> pd.DataFrame:
    """
    Process the given text, clean its sentences, and return a DataFrame.
    
    Args:
        text (str): The input text to process.
        text_id (str): Identifier for the text, e.g., file name or unique ID.
    
    Returns:
        pd.DataFrame: A DataFrame containing cleaned sentences and their associated text ID.
    """
    if cleaned_sentences:
        # Assuming clean_sentences is a predefined function that cleans the text
        # cleaned_sentences = clean_sentences(text)
        df = pd.DataFrame({"questions": cleaned_sentences})
        return df
    else:
        print("No text provided. Please check the input.")
        return pd.DataFrame() # Return an empty DataFrame if no data is found
# # Step 1: Extract text
# extracted_text = extract_text_from_pdf(pdf_path)

# # Step 2: Clean text
# cleaned_sentences = clean_text(extracted_text)


# # Step 4: Create a DataFrame with sentences under the "Questions" column
# df = pd.DataFrame({"Questions": cleaned_sentences})

# # Print the first few rows of the DataFrame
# print(df[:50])

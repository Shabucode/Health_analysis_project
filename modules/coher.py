#Finding coherence using similarity score between pages

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pdfplumber
import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')
from sentence_transformers import SentenceTransformer, util

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text_per_page = []
        for page in pdf.pages:
            text_per_page.append(page.extract_text())
    return text_per_page



# text cleaning function
def clean_text(text):
    # Replace newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs (links)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove specific unwanted punctuation (like . , " " ... ; etc.)
    text = re.sub(r'[.,;!?()"\[\]{}<>@#&%=+_-]', '', text)
    
    # Remove ellipses (multiple dots) and single dots
    text = re.sub(r'\.{2,}', '', text)  # Remove ellipses "..." or ".."
    text = re.sub(r'\.', '', text)      # Remove single dots "."

    # Remove commas and other punctuation
    text = re.sub(r',', '', text)
    
    # Remove single-word texts (words with length 1)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text


# Function to compare coherence between pages
def compare_pages_for_coherence(page_text1, page_text2):
    # Split the pages into sentences
    sentences1 = page_text1.split('.')
    sentences2 = page_text2.split('.')

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Take the last 2 sentences of the first page and the first 2 sentences of the next page
    page1_end = ' '.join(sentences1[-2:]).strip()
    page2_start = ' '.join(sentences2[:2]).strip()
    
    # Encode the sentences into embeddings
    embedding1 = model.encode(page1_end, convert_to_tensor=True)
    embedding2 = model.encode(page2_start, convert_to_tensor=True)
    
    # Compute the cosine similarity between the two embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    
    return cosine_similarity.item()

def coher_function_call(pdf_path):
# Provide the path to PDF document
# pdf_path = r"F:\Rep_Well_Being\narrative stories\PDF Books For AI\Darkness_Visible__A_Memoir_of_Madness_-_William_Styron.pdf"  # Path to your PDF file
    text_per_page = extract_text_from_pdf(pdf_path)
    cleaned_text_per_page = [clean_text(page_text) for page_text in text_per_page]

    # Compute the coherence scores between consecutive pages and calculate the overall coherence
    similarity_scores = []
    for i in range(len(cleaned_text_per_page) - 1):
        page1_text = text_per_page[i]
        page2_text = text_per_page[i + 1]
        similarity = compare_pages_for_coherence(page1_text, page2_text)
        similarity_scores.append(similarity)

    # Calculate overall coherence
    overall_coherence = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    # print(f"Overall Coherence Score: {overall_coherence:.2f}")
    if overall_coherence <0.3:
        coher_summary = "The story has very less coherence"
      # print("The story has very less coherence")
    elif overall_coherence <0.6:
        coher_summary = "The story has moderate coherence"
      # print("The story has moderate coherence")
    else:
      coher_summary = "The story has very high coherence"
      # print("The story has very high coherence")
    return overall_coherence, coher_summary

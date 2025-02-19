import re
import nltk
import pandas as pd
from collections import Counter
import pdfplumber
from nltk.corpus import cmudict
import wordninja

    
# text = pdf_extract_text(pdf_path)
# Initialize the CMU Pronouncing Dictionary
# nltk.download('cmudict')
# pron_dict = cmudict.dict()
nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)

# Check if 'cmudict' exists
try:
    nltk.data.find('corpora/cmudict')
    print("CMU Pronouncing Dictionary found.")
except LookupError:
    print("CMU Pronouncing Dictionary not found. Please download it using nltk.download('cmudict')")

# Load cmudict only if available
try:
    pron_dict = cmudict.dict()
    print("CMU Pronouncing Dictionary loaded successfully.")
except LookupError:
    print("Failed to load cmudict. Make sure it is downloaded.")

# Function to extract text from a PDF file
# pdf_path = "Down_Came_the_Rain_-_Brooke_Shields.pdf"
def pdf_extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
      text = ''
      for page in pdf.pages:
        text+=page.extract_text()
    return text

# Function to count syllables in a word
def count_syllables(word):
    word = word.lower()
    if word in pron_dict:
        # print([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] )
        # print(max([len(list(y)) for y in pron_dict[word]]))
        syllables = max([len(list(y for y in x if y[-1].isdigit())) for x in pron_dict[word.lower()]]) # Max syllables in the word
    else:
        syllables = len(re.findall(r'[aeiouy]+', word))
    
    return word, syllables 


# Function to perform text analysis
def analyze_text(text):
    # Tokenize sentences
    sentences = re.split(r'[.!?]', text)
    
    # Calculate total word count and average sentence length
    words = re.findall(r'\b\w+\b', text) #using word edges concept 
    #using wordninja to split the adjoined words
    # Split compound words using wordninja and flatten the list
    words = [item for sublist in [wordninja.split(word) for word in words] for item in sublist]

    total_word_count = len(words)
    average_sentence_length = int(total_word_count / len(sentences) if len(sentences) > 0 else 0)
    
    # Calculate average word length
    total_characters = sum(len(word) for word in words)
    average_word_length = int(total_characters / total_word_count if total_word_count > 0 else 0)
    
    # Calculate syllable statistics
    syllable_counts = []
    max_syllables = 0
    word_with_max_syllables = ""

    for word in words:
        word, syllables = count_syllables(word)
        syllable_counts.append(syllables)
        
        # Check if this word has more syllables than the current max
        if syllables > max_syllables:
            max_syllables = syllables
            word_with_max_syllables = word
    
    syllable_distribution = Counter(syllable_counts)

    # Create a dictionary for the analysis
    table_data = {
        "Statistic": [
            "Total Word Count",
            "Average Sentence Length",
            "Average Word Length",
            "Maximum Syllables per Word (Word)",
            "Words with 1 Syllable",
            "Words with 2 Syllables",
            "Words with 3 Syllables",
            "Words with 4+ Syllables"
        ],
        "Value": [
        str(total_word_count),
        str(average_sentence_length),
        str(average_word_length),
        f"{max_syllables} ({word_with_max_syllables})",
        str(syllable_distribution[1]),
        str(syllable_distribution[2]),
        str(syllable_distribution[3]),
        str(sum(v for k, v in syllable_distribution.items() if k >= 4))
        ]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(table_data)

    return df


# text_stats_df = analyze_text(text)

# # Print the DataFrame
# print(text_stats_df)

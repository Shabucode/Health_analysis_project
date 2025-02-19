import wordninja
import re

# Preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    words = [item for sublist in [wordninja.split(word) for word in words] for item in sublist]
    return words

def get_vocabulary(words):
    # words = preprocess_text(text)
    vocabulary = set(words)  # Set removes duplicates
    return vocabulary

# Calculate the Type-Token Ratio (TTR) - diversity of vocabulary
def type_token_ratio(words):
    unique_words = set(words)  # Set removes duplicates
    ttr = len(unique_words) / len(words) if len(words) > 0 else 0
    return ttr

def print_ttr_feedback(ttr):
    if ttr < 0.05:
        print("This Type-Token Ratio (TTR) is very low. The text may have an overuse of certain words.")
    elif 0.05 <= ttr < 0.15:
        print("This Type-Token Ratio (TTR) is typical for narrative or fiction. It suggests a moderate variety of vocabulary.")
    elif 0.15 <= ttr < 0.30:
        print("This Type-Token Ratio (TTR) is typical for formal or complex documents, indicating a richer vocabulary.")
    else:
        print("This Type-Token Ratio (TTR) is high, suggesting a very diverse vocabulary, typical for academic or sophisticated writing.")


def vocab_analysis(text):
    words = preprocess_text(text)
    print("Total Word Count:", len(words))

    # Get the vocabulary
    vocabulary = get_vocabulary(words)
    # Calculate TTR for the vocabulary
    ttr = type_token_ratio(words)
    # Print the vocabulary
    print("Vocabulary length is ",len(vocabulary),  "\nVocabulary:", vocabulary)
    print("Type-Token Ratio:", ttr)# Function to print TTR feedback based on score
    # Print feedback based on the TTR
    print_ttr_feedback(ttr)
    print(f"Vocabulary: ", set(words))
    return


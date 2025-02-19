import nltk
import spacy

nltk_data_path = "nltk_data" 
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Punkt not found. Please download it using nltk.download('punkt')")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("taggers not found. Please download it using nltk.download('averaged_perceptron_tagger')")
# nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')
# import en_core_web_sm
# nlp = en_core_web_sm.load()

from transformers import pipeline
from nltk.tokenize import sent_tokenize
# import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def labels_and_classifier():
    
    # Define candidate labels for rhetorical devices
    candidate_labels = ['Ethos', 'Pathos', 'Logos', 'Metaphor', 'Hyperbole', 'Repetition']

    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Set a threshold for the confidence score (you can adjust as needed)
    threshold = 0.1
    return candidate_labels, classifier, threshold

# Function to process sentences in batches
def classify_batch(sentences, batch_size=16):

    candidate_labels, classifier, threshold = labels_and_classifier()
    # Split sentences into batches
    batched_sentences = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

    results = []

    for batch in batched_sentences:
        # Get predictions for the entire batch
        predictions = classifier(batch, candidate_labels)

        # Process predictions for each sentence in the batch
        for sentence, prediction in zip(batch, predictions):
            # Get the label with the highest score
            label = prediction['labels'][0]
            score = prediction['scores'][0]

            # If the score is below the threshold, assign "No match"
            if score < threshold:
                label = "No match"

            results.append((sentence, label, score))

    return results



# # Print the relevant results
# for sentence, label, score in results:
#     if label != "No match":
#         print(f"Sentence: {sentence}")
#         print(f"Rhetorical Category: {label} (Score: {score})\n")


def rhetorical_classification(sentences):


    # Process sentences in batches and output results
    results = classify_batch(sentences)
    rhetorical_sentences = 0
    rhetorical_scores = []

    # Print the relevant results
    # print("Analysis of Rhetorical Device in the Story:\n")
    rhe_dev = []
    for sentence, label, score in results:
        if score > 0.5:  # Adjusted threshold for educational relevance
            rhetorical_sentences += 1
            rhetorical_scores.append(score)
            rhe_dev.append({
                'Sentence': sentence,
                'Rhetorical Device Category': label,
                'Rhetorical Device Score': score
            })

            # print(f"Sentence: {sentence}")
            # print(f"Educational Value Category: {label} (Score: {score:.2f})\n")
    rhe_dev_df = pd.DataFrame(rhe_dev)
    # Calculate and display the contribution
    total_sentences = len(sentences)
    rhetorical_contribution = (rhetorical_sentences / total_sentences) * 100 if total_sentences else 0
    average_score = np.mean(rhetorical_scores) if rhetorical_scores else 0

    # Define grading system
    if rhetorical_contribution > 25:
        grade = """A - Highly effective use of rhetorical devices. This suggests a strong literary or persuasive quality that engages readers and conveys deep meaning.
        Recommended for literary fiction, persuasive essays, speeches, or opinion pieces."""
    elif 10 <= rhetorical_contribution <= 25:
        grade = """B - Good use of rhetorical devices, but the story might still prioritize narrative flow, character development, or clarity.
        Suitable for short stories, personal essays, or informative content where emotional resonance and persuasive elements are important but not dominant"""
    elif 5 <= rhetorical_contribution < 10:
        grade = """C - Moderate use of rhetorical devices, It suggests that the writing may benefit from further emphasis on persuasion or emotional appeal."""
    elif 0 < rhetorical_contribution < 5:
        grade = """D - Very limited use of rhetorical devices, which may indicate that the story could lack emotional depth, persuasion,
        or engagement.."""
    elif rhetorical_contribution == 0:
        grade = """E - The story lacks rhetorical devices entirely, making it ineffective for publication unless the purpose of the writing is purely factual or technical.
        This is generally not recommended for creative writing or stories meant to engage emotionally or persuasively."""


    # Define the rhetorical devices and their meanings
    rhetorical_data = {
        "Rhetorical Device": ["Ethos", "Pathos", "Logos", "Metaphors & Similes", "Hyperbole", "Repetition"],
        "Meaning": [
            "Credibility and trustworthiness",
            "Emotional appeal",
            "Logical reasoning and facts",
            "Comparisons for vivid imagery",
            "Exaggeration for emphasis",
            "Repeating for emphasis"
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(rhetorical_data)
    # Print the DataFrame
    print(df)
    # Display the results
    rhe_dev_summary = f"""\nSummary Report:\n
    Total Sentences: {total_sentences}
    Rhetorical Sentences: {rhetorical_sentences}
    Rhetorical Contribution: {rhetorical_contribution:.2f}%
    Average Confidence Score: {average_score:.2f}
    Grade: {grade}"""
    
    return df, rhe_dev_summary, rhe_dev_df

# # Define candidate labels for rhetorical devices
# candidate_labels = ['Ethos', 'Pathos', 'Logos', 'Metaphor', 'Hyperbole', 'Repetition']

# # PDF text extraction function
# pdf_path = "/content/Darkness_Visible__A_Memoir_of_Madness_-_William_Styron (1).pdf"
# def pdf_extract_text(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text

# # Extract text from PDF
# text = pdf_extract_text(pdf_path)

# # Tokenize the text into sentences
# sentences = sent_tokenize(text)

# # Initialize the zero-shot classifier
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# # Set a threshold for the confidence score (you can adjust as needed)
# threshold = 0.1

# # Function to process sentences in batches
# def classify_batch(sentences, batch_size=16):
#     # Split sentences into batches
#     batched_sentences = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

#     results = []

#     for batch_index, batch in enumerate(batched_sentences):
#         # Get predictions for the entire batch
#         predictions = classifier(batch, candidate_labels)

#         # Process predictions for each sentence in the batch
#         for sentence, prediction in zip(batch, predictions):
#             # Get the label with the highest score
#             label = prediction['labels'][0]
#             score = prediction['scores'][0]

#             # If the score is below the threshold, assign "No match"
#             if score < threshold:
#                 label = "No match"

#             # Append the result with the batch index
#             results.append((batch_index, sentence, label, score))

#     return results

# # Process sentences in batches and output results
# results = classify_batch(sentences)

# # Analyze rhetorical devices and their educational value
# rhetorical_sentences = 0
# rhetorical_scores = []

# # Print the relevant results
# print("Analysis of Rhetorical Device in the Story:\n")
# for batch_index, sentence, label, score in results:
#     if score > 0.5:  # Adjusted threshold for educational relevance
#         rhetorical_sentences += 1
#         rhetorical_scores.append(score)
#         print(f"Batch {batch_index + 1} | Sentence: {sentence}")
#         print(f"Educational Value Category: {label} (Score: {score:.2f})\n")

# # Calculate and display the contribution
# total_sentences = len(sentences)
# rhetorical_contribution = (rhetorical_sentences / total_sentences) * 100 if total_sentences else 0
# average_score = np.mean(rhetorical_scores) if rhetorical_scores else 0

# # Define grading system
# if rhetorical_contribution > 25:
#     grade = """A - Highly effective use of rhetorical devices. This suggests a strong literary or persuasive quality that engages readers and conveys deep meaning.
#     Recommended for literary fiction, persuasive essays, speeches, or opinion pieces."""
# elif 10 <= rhetorical_contribution <= 25:
#     grade = """B - Good use of rhetorical devices, but the story might still prioritize narrative flow, character development, or clarity.
#      Suitable for short stories, personal essays, or informative content where emotional resonance and persuasive elements are important but not dominant"""
# elif 5 <= rhetorical_contribution < 10:
#     grade = """C - Moderate use of rhetorical devices, It suggests that the writing may benefit from further emphasis on persuasion or emotional appeal."""
# elif 0 < rhetorical_contribution < 5:
#     grade = """D - Very limited use of rhetorical devices, which may indicate that the story could lack emotional depth, persuasion,
#     or engagement.."""
# elif rhetorical_contribution == 0:
#     grade = """E - The story lacks rhetorical devices entirely, making it ineffective for publication unless the purpose of the writing is purely factual or technical.
#      This is generally not recommended for creative writing or stories meant to engage emotionally or persuasively."""

# # Create a pie chart showing the distribution of rhetorical devices across batches
# rhetorical_device_count = {i: 0 for i in range(len(sentences) // 16 + 1)}

# for batch_index, sentence, label, score in results:
#     if label != "No match":
#         rhetorical_device_count[batch_index] += 1

# # Create a pie chart showing the distribution of rhetorical devices across batches
# labels = [f"Batch {i+1}" for i in rhetorical_device_count.keys()]
# sizes = list(rhetorical_device_count.values())

# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
# plt.title("Distribution of Rhetorical Devices in Batches")
# plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
# plt.show()

# # Print the summary report
# print("\nSummary Report:\n")
# print(f"Total Sentences: {total_sentences}")
# print(f"Rhetorical Sentences: {rhetorical_sentences}")
# print(f"Rhetorical Contribution: {rhetorical_contribution:.2f}%")
# # print(f"Average Confidence Score: {average_score:.2f}")
# print(f"Grade: {grade}")

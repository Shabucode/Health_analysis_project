#Finding educational value in a story using zero-shot classification

from transformers import pipeline
# import pdfplumber
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nltk_data_path = "nltk_data" 
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Punkt not found. Please download it using nltk.download('punkt')")
# nltk.download('stopwords')
# Check if 'stopwords' exists
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Stopwords not found. Please download it using nltk.download('stopwords')")

def label_and_model():
    # Define the candidate labels for educational value
    candidate_labels = [
        "Recovery", "Mental Pain Reduced", "Coping Strategies", "Healing", "Emotional Relief",
        "Self-Care", "Stress Reduction", "Mindfulness", "Therapeutic Techniques", "Comfort",
        "Support", "Positive Coping", "Mental Health Improvement", "Motivation", "Resilience Building",
        "Therapeutic Impact", "Pain Relief", "Mindset Shift", "Healing Activities", "Hope",
        "Positive Change", "Distraction", "Gratitude", "Relaxation", "Escape", "Uplifting Experience",
        "Empowerment", "Supportive Environment", "Therapeutic Activity", "Respite", "Rejuvenation",
        "Emotional Release", "Stress Management", "Healing Journey", "Self-Reflection", "Perspective Shift",
        "Overcoming Adversity", "Adaptive Strategies", "Emotional Wellness", "Constructive Thinking",
        "Positive Affirmations", "Balance Restoration", "Mind-Body Connection", "Inner Peace",
        "Mindfulness Practices", "Motivational Support", "Psychological Growth", "Renewed Strength",
        "Emotional Wellbeing", "Self-Discovery", "Therapeutic Relationships", "Life Satisfaction",
        "Breakthrough", "Hopeful Outlook", "Positive Reinforcement", "Healing Mindset", "Personal Growth",
        "Adaptive Thinking", "Mental Clarity", "Self-Improvement", "Stress-Free Living", "Emotional Balance",
        "Mental Reset", "Therapeutic Support", "Self-Empowerment", "Constructive Reflection", "Restorative Practices"
    ]
    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Set a threshold for the confidence score (you can adjust as needed)
    threshold = 0.2

    return classifier, threshold, candidate_labels

# Function to process sentences in batches
def classify_batch(sentences, batch_size=16):

    classifier, threshold, candidate_labels = label_and_model()

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


def educational_value_category(sentences):
    # Process sentences in batches and output results
    results = classify_batch(sentences)

    educational_sentences = 0
    educational_scores = []

    # Print the relevant results
    print("Analysis of Educational Value in the Story:\n")
    edu_data = []
    for sentence, label, score in results:
        if score > 0.25:  # Adjusted threshold for educational relevance
            educational_sentences += 1
            educational_scores.append(score)
            edu_data.append({
                "Sentence": sentence,
                "Label": label,
                "Score": score,
            })

            print(f"Sentence: {sentence}")
            print(f"Educational Value Category: {label} (Score: {score:.2f})\n")
    edu_data_df = pd.DataFrame(edu_data)
    # Calculate and display the contribution
    total_sentences = len(sentences)
    educational_contribution = (educational_sentences / total_sentences) * 100 if total_sentences else 0
    average_score = np.mean(educational_scores) if educational_scores else 0

    # Define grading system
    if educational_contribution > 20:
        grade = "A - Highly Educational"
    elif 10 <= educational_contribution <= 20:
        grade = "B - Moderately Educational"
    elif 5 <= educational_contribution < 10:
        grade = "C - Some Educational Value"
    elif 0 < educational_contribution < 5:
        grade = "D - Minimal Educational Content"
    elif educational_contribution == 0:
        grade = "F - No Educational Content"

    # Display the results
    edu_summary = f"""\nSummary Report:\n
                        Total Sentences: {total_sentences}
                        Educational Sentences: {educational_sentences}
                        Educational Contribution: {educational_contribution:.2f}%
                        Average Confidence Score: {average_score:.2f}
                        Grade: {grade}"""

    return edu_summary, edu_data_df


# # Print the relevant results
# for sentence, label, score in results:
#     if label != "No match":
#         print(f"Sentence: {sentence}")
#         print(f"Educational Value Category: {label} (Score: {score})\n")

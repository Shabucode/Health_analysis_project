
#Analysis of coping mechanisms in the narrative story
#No data/No training - Zero shot learning

import pandas as pd
from transformers import pipeline
# from new_extract_sentences import clean_text
from modules.text_statistics import *
from pathlib import Path

def initialize_classifier(model_name="facebook/bart-large-mnli"):
    """Initialize the zero-shot-classification pipeline."""
    return pipeline("zero-shot-classification", model=model_name)


def coping_classification(senten_c):

    # senten_c = clean_sentences(text)
    # Initialize the classifier
    classifier = initialize_classifier()
    # Define the candidate labels for adaptive and maladaptive coping mechanisms
    adaptive_coping_labels = [
        "Deep breathing", "Meditation", "Exercise", "Journaling", "Talking with a friend",
        "Positive thoughts", "Taking a bath", "Reading a book", "Aromatherapy"  # Adaptive coping
    ]

    maladaptive_coping_labels = [
        "Substance abuse", "Avoidance and denial", "Self-harm", "Negative thoughts and negative self-talk",
        "Emotional eating or binge eating", "Isolation", "Procrastination(Denying and Ignoring)",
        "Overworking", "Aggression and Anger outbursts", "Excessive screen time"  # Maladaptive coping
    ]

    # Combine adaptive and maladaptive labels
    candidate_labels = adaptive_coping_labels + maladaptive_coping_labels

    # Initialize counters for adaptive and maladaptive coping
    adaptive_count = 0
    maladaptive_count = 0
    # Set a threshold for confidence 
    threshold = 0.6

    # Initialize a list to store sentence and coping mechanism pairs
    data = []
    print(f"sentence count: { len(senten_c)}")
    # Iterate over the sentences to classify them
    for sentence in senten_c:
        # Get predictions from the classifier
        predictions = classifier(sentence, candidate_labels)

        # Get the label with the highest score
        label = predictions['labels'][0]
        score = predictions['scores'][0]

        # If the score is below the threshold, assign "No match"
        if score < threshold:
            label = "No match"

        # If the label is not "No match", store it in the data list
        if label != "No match":
            print(f"Sentence: {sentence}")
            print(f"Assigned coping mechanism: {label} (Score: {score})\n")
            # Store sentence and the corresponding coping mechanism in the data list
            if label in adaptive_coping_labels:
                data.append([sentence, label, None])  # Adaptive coping in the second column
                adaptive_count += 1
            elif label in maladaptive_coping_labels:
                data.append([sentence, None, label])  # Maladaptive coping in the third column
                maladaptive_count += 1

    # calculate a ratio of adaptive to maladaptive coping
    if adaptive_count + maladaptive_count > 0:
        adaptive_ratio = adaptive_count / (adaptive_count + maladaptive_count)
        mal_adaptive_ratio = maladaptive_count / (adaptive_count + maladaptive_count)
        coping_ratio = (adaptive_count + maladaptive_count) / len(senten_c)
        # Coping measure (sum of scores): {coping_measure}\n
        coping_summary = f"""From Struggle to Strength: Detecting Coping Mechanisms in the Story\n
        Total adaptive coping sentences: {adaptive_count}\n
        Total maladaptive coping sentences: {maladaptive_count}\n

        Coping Mechanisms: Proportional Analysis\n
        Proportion of Adaptive Coping to Overall Coping Strategies: {adaptive_ratio:.2f}\n
        Proportion of Maldaptive Coping to Overall Coping Strategies: {mal_adaptive_ratio:.2f}\n
        Proportion of Coping Mechanisms in the Story: {coping_ratio:.2f}\n"""
    else:
        coping_summary = f"No coping sentences found."

    # Create a DataFrame
    df1 = pd.DataFrame(data, columns=["Sentence", "Adaptive Coping", "Maladaptive Coping"])

    # Save the DataFrame to an Excel file
    # df1.to_excel('copings_mechanisms1.xlsx', index=False)
    return coping_summary, df1


  
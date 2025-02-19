# from coher import .
# from ner_keywrd import .
# from ner_keywrd_sent import .
# from text_statistics import .
# from readability_check import .
from modules.new_extract_sentences import *

#Empathy and Emotional Intensity  #Representation and Inclusivity Check
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiments(questions):
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Initialize variables for expressiveness calculation
    positive_score = 0
    negative_score = 0
    neutral_score = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Step 2: Analyze each sentence
    for sentence in questions:
        vs = analyzer.polarity_scores(sentence)
        compound = vs['compound']
        
        # Categorize the sentiment
        if compound > 0.05:  # Positive
            positive_score += compound
            positive_count += 1
        elif compound < -0.05:  # Negative
            negative_score += compound
            negative_count += 1
        else:  # Neutral
            neutral_score += compound
            neutral_count += 1

    # Step 3: Calculate the average scores for each category
    total_count = len(questions)
    if total_count > 0:
        avg_positive = positive_score / positive_count if positive_count > 0 else 0
        avg_negative = negative_score / negative_count if negative_count > 0 else 0
        avg_neutral = neutral_score / neutral_count if neutral_count > 0 else 0
        positive_content = (positive_count / total_count) * 100
        negative_content = (negative_count / total_count) * 100
        neutral_content = (neutral_count / total_count) * 100
        
        # Combine intensities into a single output
        combined_intensity = []

        if avg_positive > 0.8:
            combined_intensity.append("High positive emotional intensity")
        elif avg_positive > 0.4:
            combined_intensity.append("Moderate positive emotional intensity")

        if avg_negative < -0.8:
            combined_intensity.append("High negative emotional intensity")
        elif avg_negative < -0.4:
            combined_intensity.append("Moderate negative emotional intensity")

        Rep_Inc= f"""Representation & Inclusivity Check\n
        Total number of sentences in the story: {total_count}
        The story has {positive_content:.2f}% positive expressions with {positive_count} sentences, {negative_content:.2f}% negative expressions with {negative_count} sentences, and {neutral_content:.2f}% of neutral expressions with {neutral_count} sentences.\n
        Empathy and Emotional Intensity of the narrative story\n
        Positive expressiveness score: {avg_positive:.2f}
        Negative expressiveness score: {avg_negative:.2f}
        Neutral expressiveness score: {avg_neutral:.2f}\n
        Overall Emotional Intensity: {" and ".join(combined_intensity)}\n
        Emotional expressiveness score scale range: -1.0 (most negative) to +1.0 (most positive)"""  
    else:
        Rep_Inc = "No questions found."
    return Rep_Inc

# Main function to process text and analyze sentiments
def emotional_intensity(cleaned_sentences):
    """
    Processes the text, extracts questions, and performs sentiment analysis.

    Parameters:
        text (str): The input text to process.
    """
    # Assuming `process_text_and_return_dataframe` is defined in extract_sentences.py
    from modules.new_extract_sentences import process_text_and_return_dataframe

    # Step 1: Process text and extract sentences
    df = process_text_and_return_dataframe(cleaned_sentences)
    questions = df['questions'].tolist()

    Rep_Inc_Emo_Intensity = analyze_sentiments(questions)
    return Rep_Inc_Emo_Intensity
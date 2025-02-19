#Readability analysis using the Readability library in Python

import pandas as pd
import nltk
from readability.readability import Readability
import pdfplumber
# nltk.download('punkt')

def analyze_readability(text):
    try:
        r = Readability(text)

        fk = r.flesch_kincaid()
        f = r.flesch()
        gf = r.gunning_fog()
        cl = r.coleman_liau()
        dc = r.dale_chall()
        ari = r.ari()
        lw = r.linsear_write()
        # sg = r.smog()
        sg = r.smog(all_sentences=True)
        sc = r.spache()


        # Define the data with corrected lengths for all columns
        data = {
            "Score Type": [
                "Flesch Reading Ease",
                "Dale Chall Score",
                "ARI Score",
                "Coleman Liau Score",
                "Gunning Fog Score",
                "Smog Score"
            ],
            "Explanation": [
                "The Flesch reading ease score, which indicates how easy a text is to read and the grade levels based on the Flesch readability score. Higher levels indicate more difficult texts.",
                "The Dale-Chall readability score, which is based on 3000 common words, with a higher score indicating greater complexity.",
                "The ARI score, which estimates the grade level needed to understand a text based on characters per word and words per sentence and the ARI grade level, indicating the grade level of the text based on the ARI formula.",
                "The Coleman-Liau index score, which is based on characters per word and sentences per text and the grade level based on the Coleman-Liau index.",
                "The Gunning Fog index, which estimates the years of formal education needed to understand a text and the grade level based on the Gunning Fog formula.",
                "The SMOG index, which is used to estimate the grade level based on syllables in a text and the grade level based on the SMOG index."
            ],
            "Grade Level": [
                str(f.grade_levels), str(dc.grade_levels), str(ari.grade_levels), f"{cl.grade_level} years of formal education", str(gf.grade_level), f"{sg.grade_level} years of formal education"
            ],
            "Score Value": [
                f.score, dc.score, ari.score, cl.score, gf.score, sg.score
            ]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Apply the mapping function to 'Scale of Difficulty' column
        df['Scale of Difficulty'] = df.apply(map_difficulty1, axis=1)
        # Apply the mapping to the 'Scale & Difficulty' column
        df['Scale & Difficulty'] = df['Score Type'].apply(map_difficulty)
        # Create the styled DataFrame with more information on scale and levels
        df_style = df.style.set_properties(subset=["Score Type", "Explanation", "Grade Level", "Score Value", "Scale of Difficulty", "Scale & Difficulty"],
                                        **{'text-align': 'left'})
        # # Access the single string in f.grade_levels list
        # grade_level_text = f.grade_levels[0]
        grade_level_text = f.grade_levels[0] if isinstance(f.grade_levels, list) else f.grade_levels
        summary_text = display_summary(f, grade_level_text)

        return df_style, summary_text
    
    except Exception as e:
        print(f"Error in readability analysis: {e}")
        return "Error"

# Map difficulty levels to the 'Scale & Difficulty' column
def map_difficulty(score_type):
    if score_type == "Flesch Reading Ease":
        return "Scale: 0-100 | Level: 90-100 is Very Easy, 60-89 is Easy, 30-59 is Difficult, 0-29 is Very Difficult"
    elif score_type == "Dale Chall Score":
        return "Scale: 0-12 | Level: 0-4.9 is Very Easy, 5-6.9 is Easy, 7-8.9 is Average, 9-10.9 is Difficult, 11+ is Very Difficult"
    elif score_type == "ARI Score":
        return "Scale: 1-12 | Level: 1-5 is Easy, 6-8 is Standard, 9-11 is Difficult, 12+ is Very Difficult"
    elif score_type == "Coleman Liau Score":
        return "Scale: 0-100 | Level: (Customized scale)"
    elif score_type == "Gunning Fog Score":
        return "Scale: 5-16+ | Level: 5-8 is Easy to Read, 9-12 is Moderate Difficulty, 13-15 is Difficult, 16+ is Very Difficult"
    elif score_type == "Smog Score":
        return "Scale: 3-16+ | Level: 3-7 is Easy, 8-11 is Moderate, 12-15 is Difficult, 16+ is Very Difficult"


# Define a mapping function based on score type and score value
def map_difficulty1(row):
    try:
        score_type = row['Score Type']
        score_value = row['Score Value']

        if score_type == "Flesch Reading Ease":
            if score_value >= 90:
                return "Very Easy (90-100)"
            elif score_value >= 60:
                return "Easy (60-89)"
            elif score_value >= 30:
                return "Difficult (30-59)"
            else:
                return "Very Difficult (0-29)"

        elif score_type == "Dale Chall Score":
            if score_value < 5:
                return "Very Easy (0-4.9)"
            elif score_value < 7:
                return "Easy (5-6.9)"
            elif score_value < 9:
                return "Average (7-8.9)"
            elif score_value < 11:
                return "Difficult (9-10.9)"
            else:
                return "Very Difficult (11+)"

        elif score_type == "ARI Score":
            if score_value <= 5:
                return "Easy (1-5)"
            elif score_value <= 8:
                return "Standard (6-8)"
            elif score_value <= 11:
                return "Difficult (9-11)"
            else:
                return "Very Difficult (12+)"

        elif score_type == "Coleman Liau Score":
            return "(Customized scale based on Score Type)"

        elif score_type == "Gunning Fog Score":
            if score_value < 8:
                return "Easy to Read (5-8)"
            elif score_value <= 12:
                return "Moderate Difficulty (9-12)"
            elif score_value <= 15:
                return "Difficult (13-15)"
            else:
                return "Very Difficult (16+)"

        elif score_type == "Smog Score":
            if score_value < 8:
                return "Easy (3-7)"
            elif score_value <= 11:
                return "Moderate (8-11)"
            elif score_value <= 15:
                return "Difficult (12-15)"
            else:
                return "Very Difficult (16+)"

        return "No Scale Available"
    
    except Exception as e:
        print(f"Error in mapping based on score type and score value: {e}")
        return "Error"

# # Apply the mapping function to 'Scale of Difficulty' column
# df['Scale of Difficulty'] = df.apply(map_difficulty1, axis=1)
# # Apply the mapping to the 'Scale & Difficulty' column
# df['Scale & Difficulty'] = df['Score Type'].apply(map_difficulty)

# # Create the styled DataFrame with more information on scale and levels
# df_style = df.style.set_properties(subset=["Score Type", "Explanation", "Grade Level", "Score Value", "Scale of Difficulty", "Scale & Difficulty"],
#                                    **{'text-align': 'left'})
from IPython.display import HTML, display
HTML_COLOR_MAP = {
    "very easy": "green",
    "easy": "green",
    "difficult": "red",
    "very difficult": "red",
    "5th Grade": "green",
    "8th Grade": "green",
    "12th Grade": "green",
    "college": "red"
}
def get_html_score_color(score):
    try:
        if score >= 90:
            return HTML_COLOR_MAP[" very easy"]
        elif score >= 60:
            return HTML_COLOR_MAP["easy"]
        elif score >= 30:
            return HTML_COLOR_MAP["difficult"]
        else:
            return HTML_COLOR_MAP["very difficult"]
    except Exception as e:
        print(f"Error in getting score color mapping: {e}")

def get_html_difficulty_color(difficulty):
    return HTML_COLOR_MAP.get(difficulty, "black")

# 
def display_summary(f, grade_level_text):
    try:
        # Adjusted HTML code
        summary_text = (
            f"The readability analysis shows an overall Flesch Reading Ease score of "
            f"<span style='color:{get_html_score_color(f.score)};'>{f.score}</span>.<br>"
            f"Indicating a difficulty level of '<span style='color:{get_html_difficulty_color(f.ease)};'>{f.ease}</span>' "
            f"with an estimated grade level requirement of '<span style='color:{HTML_COLOR_MAP.get(grade_level_text, 'black')};'>{grade_level_text}</span>'.<br>"
            "Additional readability scores contribute to a comprehensive understanding of text complexity."
        )
        # return HTML(summary_text)
        return summary_text
    except Exception as e:
        print(f"Error in summary display: {e}")
        return "Summary generation failed."


# display(HTML(summary_text))
# # Display the table with styling
# print(HTML(summary_text))
# print(df)



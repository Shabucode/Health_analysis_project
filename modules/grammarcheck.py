import requests
import pdfplumber
# from fpdf import FPDF


# # Function to extract text from the PDF
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text

# Function to check grammar using LanguageTool
def check_grammar_with_languagetool(text):
    url = "https://api.languagetool.org/v2/check"
    data = {
        "language": "en-US",
        "text": text
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to calculate grammar correctness score
def calculate_grammar_correctness_score(text, grammar_check_result):
    if not grammar_check_result or "matches" not in grammar_check_result:
        return 100  # No issues found, 100% correct

    total_issues = len(grammar_check_result["matches"])
    total_words = len(text.split())

    # A simple heuristic: (number of issues / total words) * 100 to get a percentage of incorrectness
    incorrectness_percentage = (total_issues / total_words) * 100
    correctness_percentage = 100 - incorrectness_percentage

    return round(correctness_percentage, 2)

# Function to process the PDF and check grammar in smaller chunks
def process_pdf_for_grammar(text, chunk_size=5000):
    # Step 1: Extract text from the PDF
    # text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split text into smaller chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    total_issues = 0
    total_words = 0

    # Step 3: Process each chunk separately
    for chunk in chunks:
        # Check grammar using LanguageTool
        grammar_check_result = check_grammar_with_languagetool(chunk)
        
        # Calculate the grammar correctness score for this chunk
        grammar_score = calculate_grammar_correctness_score(chunk, grammar_check_result)
        # print(f"Grammar Correctness Score for chunk: {grammar_score}%")
        
        # Step 4: Show grammar issues (if any)
        # if grammar_check_result:
        #     for match in grammar_check_result.get("matches", []):
        #         print(f"Error: {match['message']}")
        #         print(f"Context: {match['context']}")
        #         print(f"Suggested correction: {', '.join([s['value'] for s in match.get('replacements', [])])}")
        #         print("-" * 40)
        
        # Track total issues and words for final score
        if grammar_check_result:
            total_issues += len(grammar_check_result["matches"])
        total_words += len(chunk.split())

    # Final score based on all chunks
    overall_score = 100 - ((total_issues / total_words) * 100)
    # print(f"Overall Grammar Correctness Score: {round(overall_score, 2)}%")
    return overall_score
# pdf_path = "Down_Came_the_Rain_-_Brooke_Shields.pdf"
# process_pdf_for_grammar(pdf_path)

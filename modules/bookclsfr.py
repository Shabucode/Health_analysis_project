import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load and preprocess data from the given Excel file."""
    data = pd.read_excel(filepath)
    X = data.drop(['Publish', 'Book_ID'], axis=1)  # Features
    y = data['Publish']  # Label
    return X, y

def train_model(X, y, model_path, test_size=0.2, random_state=42):
    """Train a logistic regression model and save it to a file."""
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """Load a trained model from a file."""
    return joblib.load(model_path)

def classify_story(LR_model, X_new, threshold=3, excluded_features=None):
    """Classify a story and provide improvement suggestions based on feature scores."""
    if excluded_features is None:
        excluded_features = ['Boredom', 'Humor', 'Romance', 'Comfort level', 'Health', 'Sadness']

    # Make predictions
    predictions = LR_model.predict(X_new)
    print("Predictions:", predictions)

    # Identify features below the threshold
    low_features = [
        feature for feature in X_new.columns[X_new.iloc[0] < threshold].tolist()
        if feature not in excluded_features
    ]

    # Generate suggestions based on predictions and feature scores
    if predictions[0] == 1:
        if low_features:
            suggestions = (
                f"The story is predicted to be ready for publishing. However, the following features scored below the standard score, {threshold} "
                f"and could still be improved: {', '.join(low_features)}."
            )
        else:
            suggestions = "All features scored above the threshold. The story is ready for publishing."
    else:
        if low_features:
            suggestions = (
                f"The story could be improved for publishing. \nThe following features scored below the standard score, {threshold} for the scale of (0 to 4) "
                f"and need improvement: \n{', '.join(low_features)}."
            )
        else:
            suggestions = "The story could be improved for publishing."

    # Display suggestions
    print(suggestions)
    return predictions[0], suggestions

def main():
    # File paths
    data_filepath = "b1.xlsx"
    model_filepath = "LR.pkl"
     # Check if the model already exists
    if not os.path.exists(model_filepath):
        print("Model not found. Training a new model...")
        X, y = load_data(data_filepath)
        train_model(X, y, model_filepath)
    else:
        print("Model already exists. Skipping training...")

    # # Train and save the model
    # X, y = load_data(data_filepath)
    # train_model(X, y, model_filepath)

    # Perform inference on new data
    column_names = [
        "Storytelling", "Storyline", "Writing quality", "Tone", "Pacing", "Heartwarming", "Insight", "Honesty",
        "Empathy", "Writing style", "Enthralling", "Inspirational story", "Thought provoking", "Readability",
        "Warmth", "Sadness", "Comfort level", "Mental health issue", "Health", "Romance", "Humor",
        "Educational value", "Value", "Boredom"
    ]
    
    # Accept X_new as user input
    print("Enter feature values for the new story in CSV format (comma-separated):")
    user_input = input()
    feature_values = [float(value.strip()) for value in user_input.split(',')]
    X_new = pd.DataFrame([feature_values], columns=column_names)

    # Load the trained model
    LR_model = load_model(model_filepath)

    # Classify and generate suggestions
    classify_story(LR_model, X_new)

# if __name__ == "__main__":
#     main()

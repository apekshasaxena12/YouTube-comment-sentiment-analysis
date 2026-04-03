#vader + transformer

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

nltk.download("vader_lexicon")

INPUT_FILE = "5_youtube_comments_raw_ed.csv"
OUTPUT_FILE = "5_hybrid.csv"
TEXT_COLUMN = "comment"

LABELS = [
    "Supportive",
    "Suggestive",
    "Sarcastic",
    "Normal",
    "Negative political",
    "Negative misogynistic",
    "Negative personal attack",
    "Negative normal"
]


# INITIALIZE MODELS

vader = SentimentIntensityAnalyzer()

# Hate / attack detection
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    truncation=True
)

# Political detection (zero-shot)
political_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# CLASSIFICATION FUNCTION

def classify_comment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Normal"

    text = str(text)


    scores = vader.polarity_scores(text)
    compound = scores["compound"]


    # POSITIVE / NEUTRAL
    if compound >= 0.4:
        return "Supportive"

    if 0.05 <= compound < 0.4:
        return "Suggestive"

    if -0.05 < compound < 0.05:
        return "Normal"


    # NEGATIVE PIPELINE
    if compound <= -0.1:

        # Toxicity / misogyny / attack 
        tox = toxicity_classifier(text[:512])[0]

        if tox["label"] in ["toxic", "severe_toxic", "insult", "threat"] and tox["score"] > 0.6:
            if tox["label"] == "insult":
                return "Negative personal attack"
            else:
                return "Negative misogynistic"

        # Political negativity 
        political_result = political_classifier(
            text[:512],
            candidate_labels=["politics", "entertainment", "sports", "education"]
        )

        if political_result["labels"][0] == "politics" and political_result["scores"][0] > 0.6:
            return "Negative political"

        # Fallback 
        return "Negative normal"

    return "Normal"


df = pd.read_csv(INPUT_FILE)

df["label"] = df[TEXT_COLUMN].apply(classify_comment)

df.to_csv(OUTPUT_FILE, index=False)

print("Classification complete → vader.csv")

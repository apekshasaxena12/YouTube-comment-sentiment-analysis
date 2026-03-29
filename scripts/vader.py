import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

INPUT_FILE = "5_youtube_comments_raw_ed.csv"
OUTPUT_FILE = "5_vader.csv"
TEXT_COLUMN = "comment"   

sia = SentimentIntensityAnalyzer()

def has_excessive_punctuation(text):
    return bool(re.search(r"[!?.]{2,}", text))

def has_all_caps(text):
    words = text.split()
    if len(words) == 0:
        return False
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    return len(caps_words) / len(words) > 0.3


# CLASSIFICATION FUNCTION

def classify_vader(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Normal"

    text = str(text)
    scores = sia.polarity_scores(text)

    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    neu = scores["neu"]

    intensity = has_excessive_punctuation(text) or has_all_caps(text)

    # SUPPORTIVE
    if compound >= 0.4 and pos > neg:
        return "Supportive"

    # SUGGESTIVE (neutral-positive with intent)
    if 0.05 <= compound < 0.4 and neu > 0.4:
        return "Suggestive"

    # SARCASTIC (mixed polarity + intensity)
    if abs(pos - neg) < 0.1 and intensity and compound < 0:
        return "Sarcastic"

    # NEGATIVE CLASSES (coarse-grained)
    if compound <= -0.4:
        if intensity:
            return "Negative personal attack"
        else:
            return "Negative normal"

    if -0.4 < compound <= -0.1:
        return "Negative normal"

    # NORMAL
    return "Normal"

df = pd.read_csv(INPUT_FILE)

df["label"] = df[TEXT_COLUMN].apply(classify_vader)

df.to_csv(OUTPUT_FILE, index=False)

print("✅ VADER-based classification complete → vader.csv")

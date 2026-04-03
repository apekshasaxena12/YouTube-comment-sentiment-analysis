import pandas as pd
from google import genai
from tqdm import tqdm
import time

GEMINI_API_KEY = "YOUR API KEY"

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-2.5-flash"
BATCH_SIZE = 20

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

# LOAD COMMENTS
df = pd.read_csv("5_youtube_comments_raw_ed.csv")

comment_col = next(col for col in df.columns if "comment" in col.lower())
df = df.rename(columns={comment_col: "comment"})

comments = df["comment"].astype(str).tolist()


# CLASSIFICATION FUNCTION
def classify_batch(batch_comments):
    prompt = f"""
You are an expert content moderator.

Classify each YouTube comment into ONE of the following labels:
{LABELS}

Rules:
- Choose exactly one label per comment
- Output ONLY the label
- Keep order same as input
- One label per line

Comments:
"""
    for i, c in enumerate(batch_comments, 1):
        prompt += f"{i}. {c}\n"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    labels = response.text.strip().splitlines()

    if len(labels) != len(batch_comments):
        labels = ["Normal"] * len(batch_comments)

    return labels


# BATCHED INFERENCE 
predicted_labels = []

for i in tqdm(range(0, len(comments), BATCH_SIZE), desc="Classifying with Gemini"):
    batch = comments[i:i + BATCH_SIZE]
    predicted_labels.extend(classify_batch(batch))
    time.sleep(12)  # free-tier rate limit safety


# SAVE OUTPUT
output_df = pd.DataFrame({
    "comment": comments,
    "label": predicted_labels
})

output_df.to_csv("Gemini.csv", index=False)

print(" Gemini classification completed")
print("Output saved as Gemini.csv")

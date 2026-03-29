import pandas as pd 
import torch 
from transformers import pipeline 
from tqdm import tqdm 
# ----------------------------- 
# CONFIG 
# ----------------------------- 

INPUT_FILE = "5_youtube_comments_raw_ed.csv" 
OUTPUT_FILE = "5_roberta_ed.csv"
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

TEXT_COLUMN = "comment" 

# ----------------------------- 
# LOAD DATA 
# ----------------------------- 
df = pd.read_csv(INPUT_FILE) 
df = df.dropna(subset=[TEXT_COLUMN]) 
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str) 

print("Total comments:", len(df)) 

# ----------------------------- 
# DEVICE SELECTION 
# ----------------------------- 
if torch.cuda.is_available(): 
    device = 0 
elif torch.backends.mps.is_available(): 
    device = "mps" 
else: device = -1 

print("Using device:", device) 

# ----------------------------- 
# LOAD ZERO-SHOT MODEL (STABLE) 
# ----------------------------- 
classifier = pipeline( 
    "zero-shot-classification", 
    model="facebook/bart-large-mnli", 
    device=device, batch_size=8 
) 

# ----------------------------- 
# CLASSIFICATION FUNCTION 
# ----------------------------- 
def classify_comment(text): 
    try: 
        result = classifier( 
            text, 
            candidate_labels=LABELS, 
            multi_label=False 
        ) 
        return result["labels"][0] 
    except Exception: 
        return "Error" 
    
# ----------------------------- 
# APPLY MODEL 
# ----------------------------- 
tqdm.pandas() 
df["label"] = df[TEXT_COLUMN].progress_apply(classify_comment) 

# ----------------------------- 
# SAVE OUTPUT 
# ----------------------------- 
df.to_csv(OUTPUT_FILE, index=False) 
print("Saved classified comments to:", OUTPUT_FILE)

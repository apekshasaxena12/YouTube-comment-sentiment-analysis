# YouTube Comment Sentiment Analysis: A Multi-Method Comparative Study

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research-orange?style=flat-square)
![Methods](https://img.shields.io/badge/Methods-4-purple?style=flat-square)

> A research project comparing four sentiment/comment classification methods — VADER (Lexicon), Hybrid (VADER + RoBERTa), RoBERTa (Transformer), and Gemini 2.5 Flash (LLM) — across 1,250 YouTube comments from five diverse video categories.

---

## Abstract

This study evaluates the performance of four natural language processing approaches for fine-grained YouTube comment classification. Comments are collected from five distinct video genres (music, politics, podcast, gaming, and education), manually labeled, and then classified by each method independently. Results are benchmarked against human-annotated ground truth using Accuracy, Precision, Recall, and F1-Score.

**Key Finding:** LLM-based classification (Gemini 2.5 Flash) significantly outperforms all other methods, achieving up to 92% accuracy — compared to ~25–63% for traditional and transformer-based approaches.

---

## Repository Structure

```
youtube-comment-sentiment/
│
├── data/
│   ├── raw/
│   │   ├── 1_youtube_comments_raw_song.csv
│   │   ├── 2_youtube_comments_raw_politics.csv
│   │   ├── 3_youtube_comments_raw_doac.csv
│   │   ├── 4_youtube_comments_raw_game.csv
│   │   └── 5_youtube_comments_raw_ed.csv
│   │
│   └── classified/
│       ├── {1..5}_comments_class.csv      ← Ground truth (human labels)
│       ├── {1..5}_vader.csv               ← VADER predictions
│       ├── {1..5}_hybrid.csv              ← Hybrid predictions
│       ├── {1..5}_roberta.csv             ← RoBERTa predictions
│       └── {1..5}_Gemini.csv              ← Gemini predictions
│
├── scripts/
│   ├── comments.py         ← YouTube API data collection
│   ├── vader.py            ← VADER lexicon classifier
│   ├── roberta.py          ← RoBERTa transformer classifier
│   ├── hybrid.py           ← Hybrid (VADER + RoBERTa) classifier
│   ├── gemini.py           ← Gemini 2.5 Flash LLM classifier  ⭐ Main method
│   ├── compare.py          ← Evaluation & metric computation
│   ├── results.py          ← Results aggregation & charts
│   └── mismatch.py         ← Error/mismatch analysis
│
├── outputs/
│   ├── performance_comparison.png     ← Line chart: Accuracy/P/R/F1 per method
│   └── accuracy_heatmap.png           ← Heatmap: Accuracy per video × method
│
├── explanation.txt         ← Notes on methodology & design decisions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| # | Video Category | Raw CSV | Comments |
|---|---------------|---------|----------|
| 1 | Music / Song | `1_youtube_comments_raw_song.csv` | 250 |
| 2 | Politics | `2_youtube_comments_raw_politics.csv` | 250 |
| 3 | Podcast | `3_youtube_comments_raw_doac.csv` | 250 |
| 4 | Gaming | `4_youtube_comments_raw_game.csv` | 250 |
| 5 | Education | `5_youtube_comments_raw_ed.csv` | 250 |
| | **Total** | | **1,250** |

Comments were collected using the **YouTube Data API v3**.

---

## Label Schema

All methods classify each comment into one of **8 categories**:

| Label | Description |
|-------|-------------|
| `Supportive` | Positive, encouraging, or agreeable comments |
| `Suggestive` | Constructive feedback or suggestions |
| `Sarcastic` | Irony or sarcasm |
| `Normal` | Neutral, conversational comments |
| `Negative political` | Politically hostile or inflammatory |
| `Negative misogynistic` | Gender-based hate or harassment |
| `Negative personal attack` | Direct attacks on individuals |
| `Negative normal` | General negativity or criticism |

---

## Methods

### 1. VADER — Lexicon-Based (`vader.py`)
Rule-based sentiment analysis using the VADER (Valence Aware Dictionary and Sentiment Reasoner) lexicon. Compound scores are mapped to the 8-label schema.

### 2. Hybrid — VADER + RoBERTa (`hybrid.py`)
A two-stage approach: VADER provides coarse polarity, and RoBERTa refines the classification. Labels are resolved via a weighted decision logic.

### 3. RoBERTa — Transformer-Based (`roberta.py`)
Uses a fine-tuned `cardiffnlp/twitter-roberta-base-sentiment` model for contextual sentiment classification.

### 4. Gemini 2.5 Flash — LLM-Based (`gemini.py`) ⭐
**Primary method.** Uses Google's `gemini-2.5-flash` model via batch prompting (20 comments/batch) with structured zero-shot instruction. Achieves highest accuracy across all video categories.

#### Gemini Pipeline

```
YouTube Data API
      ↓
Raw Comment Storage (CSV)
      ↓
Data Loading & Column Standardization
      ↓
Minimal Preprocessing
      ↓
Batch Segmentation (20 comments/batch)
      ↓
Structured Prompt Construction
      ↓
Gemini API Inference (gemini-2.5-flash)
      ↓
Response Parsing & Validation
      ↓
Final CSV Output
```

---

## Results

### Overall Performance (Averaged Across All 5 Videos)

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| VADER | ~38% | ~40% | ~39% | ~37% |
| Hybrid | ~40% | ~47% | ~40% | ~40% |
| RoBERTa | ~35% | ~41% | ~35% | ~34% |
| **Gemini** | **~82%** | **~85%** | **~82%** | **~82%** |

### Per-Video Accuracy

| Video | VADER | Hybrid | RoBERTa | Gemini |
|-------|-------|--------|---------|--------|
| 1 — Song | 0.62 | 0.63 | 0.51 | **0.92** |
| 2 — Politics | 0.25 | 0.36 | 0.48 | **0.87** |
| 3 — Podcast | 0.28 | 0.29 | 0.21 | **0.83** |
| 4 — Gaming | 0.39 | 0.38 | 0.31 | **0.78** |
| 5 — Education | 0.35 | 0.36 | 0.18 | **0.70** |

---

## Getting Started

### Prerequisites

```bash
python >= 3.9
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/youtube-comment-sentiment.git
cd youtube-comment-sentiment
pip install -r requirements.txt
```

### Running the Pipeline

**Step 1 — Collect comments (requires YouTube Data API key):**
```bash
python scripts/comments.py
```

**Step 2 — Run classifiers:**
```bash
python scripts/vader.py
python scripts/roberta.py
python scripts/hybrid.py
python scripts/gemini.py      # Requires Gemini API key
```

**Step 3 — Compare & evaluate:**
```bash
python scripts/compare.py
python scripts/results.py
```

**Step 4 — Analyze mismatches:**
```bash
python scripts/mismatch.py
```

---

## API Keys

This project requires two API keys. **Never commit them to the repository.**

Create a `.env` file in the root directory:

```env
YOUTUBE_API_KEY=your_youtube_data_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Then load them in the scripts via `os.getenv("GEMINI_API_KEY")`.

> ⚠️ The `.gitignore` already excludes `.env` files. If you have hardcoded any API keys in your scripts, **rotate them immediately** before pushing to GitHub.

---

## Requirements

```
google-generativeai
pandas
tqdm
vaderSentiment
transformers
torch
scikit-learn
matplotlib
seaborn
python-dotenv
google-api-python-client
```

Install all via:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) — C.J. Hutto & E.E. Gilbert
- [CardiffNLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) — Cardiff NLP
- [Google Gemini API](https://ai.google.dev/) — Google DeepMind
- [YouTube Data API v3](https://developers.google.com/youtube/v3) — Google

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{yourname2026ytsentiment,
  title     = {YouTube Comment Sentiment Analysis: A Multi-Method Comparative Study},
  author    = {Your Name},
  year      = {2026},
  url       = {https://github.com/YOUR_USERNAME/youtube-comment-sentiment}
}
```

---

*For questions or collaboration, open an issue or reach out via GitHub.*

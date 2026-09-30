# 🏆 Business Entity Resolution — Amazon ML Challenge 2026

> **Team zippy** — Final submission scored **macro F0.5 = 0.9747** on the public leaderboard and **0.9827** on validation.

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-green.svg)](https://github.com/microsoft/LightGBM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Given ~2 million Source-1 (S1) reference business entities and ~10 million noisy Source-2/3 (S2/S3) records,
our solution determines which S2/S3 records refer to the same real-world business as each S1 entity.

The pipeline handles multi-script text (Devanagari, Gujarati, Bengali, Tamil, Telugu, Kannada), legal-form
variations, leetspeak typos, address inconsistencies, and an **entirely unseen country (France)** at test time.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE AT A GLANCE                                      │
│                                                                                     │
│  Raw TSVs ──► Normalise ──► Block (stage A + B) ──► GBDT Matcher ──► Cross-Encoder  │
│                                                          │                │         │
│                                                          └───► Blend ◄────┘         │
│                                                                  │                  │
│                                                            One-to-One               │
│                                                                  │                  │
│                                                         Global Threshold            │
│                                                                  │                  │
│                                                          matching_results.tsv        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem Statement

The [Amazon ML Challenge 2026](docs/problem_statement.md) poses a large-scale **entity resolution** task:

| Aspect | Detail |
|---|---|
| **Input** | S1 (reference), S2 and S3 (noisy copies + distractors), with business name, address, country |
| **Output** | For each S1 entity, a list of matching S2/S3 record IDs |
| **Metric** | **Macro F0.5** per S1 entity (precision-heavy: false merges hurt more than missed matches) |
| **Train** | 2.21M S1, 5.03M S2, 5.29M S3 records — US and India |
| **Test** | 1.73M S1 entities — US (38%), India (47%), **France (15%, unseen in training)** |
| **Constraints** | Models must be MIT/Apache licensed, ≤ 8B parameters; no external data or lookups |

### Key Data Characteristics We Exploited

- **Every S2/S3 record belongs to at most one S1 entity** → one-to-one post-processing
- **5.6% of S1 entities are singletons** (no matches) → predicting nothing earns a perfect score for these
- **~27% of S2/S3 records are distractors**; test has even more (5.8 vs 4.7 records per S1)
- **Sibling look-alikes** in test: *X Exports / X Group / X Ventures / X Holdings*
- **Noise types**: legal-form swaps, leetspeak, Indic scripts, DBA constructions, appended phone numbers,
  house-number edits, missing addresses, reordered address components
- **12% of normalised S1 names are not unique** → name-only matching is insufficient
- **Country always matches** → blocking is per-country

---

## Solution Architecture

```
raw TSVs ─► prepare.py ─► normalised parquet parts (+ learned token map)
              ▼
         block.py   stage A: hashed multi-key inverted index per country, IDF key-overlap score
         block2.py  stage B: pool (top-30 + side slots + query expansion) → LightGBM ranker
              │             → global budget: 10 candidates / S1        ═► candidate_pairs.tsv
              ▼
         train.py   ~120 pair features → LightGBM matcher (+ leave-one-country-out model)
         predict.py test scoring
              ▼
         kaggle_ce*.py / ce_score_local.py   cross-encoders (e5-base, e5-large)
              ▼
         blend_ce.py 0.5·GBDT + 0.5·mean(CE) → one-to-one → global threshold
                                                                         ═► matching_results.tsv
```

---

## Pipeline Walkthrough

### Step 1 — Normalisation

**Files:** [`src/common.py`](src/common.py) (`Normalizer` class), [`src/prepare.py`](src/prepare.py)

Two records that a human would call "the same string, written differently" should become the same tokens.

Each record is normalised into structured fields:
- `nm` — full normalised name
- `core` — name without legal forms or filler words
- `ad` — normalised address
- `nums` — extracted house numbers, PIN/ZIP codes
- `alph` — alphabetic-only address tokens

**Normalisation steps:**

| Technique | Example |
|---|---|
| Case + Unicode accent strip | `Café → cafe` |
| Junk prefix removal | `M/s ##COMPANY → company` |
| Leetspeak repair | `0bsidian → obsidian`, `a1ltime → alltime` |
| Dotted acronym collapse | `S.A.R.L. → sarl`, `P.V.T. → pvt` |
| Legal-form canonicalisation | `Private Limited / Pvt Ltd / LLC → pvtltd` |
| Address abbreviations | `Street → st`, `Road → rd`, `Avenue → ave` |
| State name mapping | `Maharashtra → MH`, `महाराष्ट्र → MH` |
| French article removal | `de, du, des, la, le, les` stripped from address |
| **Learned Indic token map** | Script tokens matched to their Latin equivalents from training ground-truth pairs |
| Rule-based Brahmic romaniser | Fallback for every Indic Unicode block |

> **Why learn the token map?** The data has its own spelling habits ("Shri"/"Shree", "Laxmi"/"Lakshmi").
> The matched training pairs show us exactly which Latin spelling each native-script word maps to. This also
> respects the "no external data" rule.

Data is stored as 1M-row parquet part files so the pipeline runs with ~16 GB RAM.

---

### Step 2 — Candidate Generation (Blocking)

With 1.7M S1 entities and ~10M S2/S3 records, brute-force all-pairs comparison is infeasible. Blocking
decides which pairs are even considered — a true match lost here is lost forever.

#### Stage A — Multi-key Inverted Index ([`src/block.py`](src/block.py))

Nine types of hashed blocking keys are built per record, with a per-country inverted index:

| Key | What it captures |
|---|---|
| `c:` concatenated core name | Exact name matches including domain-style names |
| `s:` sorted unique core tokens | Order-invariant name overlap |
| `p:` unordered 4-char prefix pairs | Typo and word-order robustness |
| `u:` single rare token prefix | Long-tail rare names |
| `n:` core name × locality | Disambiguates common names by location |
| `m:` name-prefix × locality | Catches partial name matches |
| `a:` house number × street/city | Address-based matching |
| `x:` name-prefix × house number | Cross-modal disambiguation |
| `z:` ZIP/PIN × name token | Postal code anchoring |

Each shared key contributes `idf = log(1 + N/df)`, and keys with `df > 300` are discarded as too common.
Selection: top-12 by total score + top-3 name-only + top-3 address-only side slots.

**Result:** ~13 candidates per S1, 95.6% pair recall on held-out training data.

#### Stage B — Learned Candidate Ranker ([`src/block2.py`](src/block2.py))

Instead of asking "how many candidates per entity?", we asked **"which candidates, across all entities,
are worth keeping?"**

1. **Pool** — Top-30 + side slots + **query expansion**: the top-3 candidates are used as extra queries,
   and their best 5 neighbours join the pool
2. **LightGBM ranker** — Trained on 150K non-validation S1 entities, using group/competition features
   (score, rank, gap to best, channel scores, expansion flags)
3. **Global cutoff** — One threshold on raw ranker score so easy entities keep 3–4 candidates and hard
   ones keep 20+

| Method | Pair Recall | Candidates/S1 |
|---|---|---|
| Stage A alone (top-12 + sides) | 95.6% | 13 |
| Stage A with K=40 | 97.1% | 40.6 |
| **Stage A + B (final)** | **97.09%** | **10** |

> **Query expansion** is the secret sauce here: if S1 "A" easily finds copy "B", then B's own nearest
> neighbours are often the harder copies of A (transliterated, or with a missing address). Asking B for
> its neighbours reaches records that share no key with A at all.

---

### Step 3 — Pair Scoring (LightGBM Matcher)

**Files:** [`src/features.py`](src/features.py), [`src/train.py`](src/train.py), [`src/predict.py`](src/predict.py)

- **Training data:** Candidates of 700K S1 entities, ~9.08M pairs (25.5% positive)
- **Validation:** 40K held-out S1 entities (fixed `RandomState(42)` permutation)
- **Model:** LightGBM binary classifier — 255 leaves, lr 0.05, early stopping, up to 3,000 trees

See the [Feature Engineering](#feature-engineering) section below for the full ~120 feature breakdown.

---

### Step 4 — Cross-Encoder Ensemble

The GBDT sees hand-designed features. A cross-encoder reads two records as raw text and can discover
patterns we did not anticipate — paraphrases, unusual abbreviations, mixed-script names.

| Tag | Model (MIT) | Platform | Pairs Scored |
|---|---|---|---|
| **ce1** | `multilingual-e5-base` (278M), 800K pairs | Kaggle 2×T4 + local GPU | All val + test |
| **ce2** | `multilingual-e5-large` (560M), 1M pairs | Kaggle 2×T4 | Val + uncertain test |
| **ce4** | `e5-large` continued from ce2 (500K new pairs, lr 1e-5) | Kaggle 2×T4 | Val + uncertain test |

**Input format:** `name | address` for each record, fed as two segments to the cross-encoder.

Only "uncertain" pairs (0.002 < p < 0.998 from ce1) are scored by the expensive large models — free Kaggle
GPU time is limited, so we spend it where a second opinion matters.

**Files:** [`src/kaggle_ce.py`](src/kaggle_ce.py) (all stages: ce1, ce2, ce4, extra),
[`src/ce_score_local.py`](src/ce_score_local.py)

---

### Step 5 — Decision: Blend + One-to-One + Threshold

**File:** [`src/blend_ce.py`](src/blend_ce.py)

1. **Blend:** `pb = 0.5 × p_gbdt + 0.5 × mean(ce1, ce2, ce4)`
2. **Unseen countries (France):** GBDT-only with leave-one-country-out probabilities rescaled so one
   threshold applies universally
3. **One-to-one:** Each S2/S3 record is kept only for the S1 that scores it highest
4. **Global threshold:** Accept pair if score ≥ 0.65 (grid-searched for macro F0.5 on validation)

---

## Feature Engineering

**~120 features** in [`src/features.py`](src/features.py), computed per country:

### A. Vector Similarities (Hashed TF-IDF, 2²⁰ dims, cosine)

- Name char-2/3/4-grams: `cos_c2_nm`, `cos_c3_nm`, `cos_c4_nm`
- Core char-3 and word: `cos_c3_core`, `cos_w_core`
- Address char-3 and word: `cos_c3_ad`, `cos_w_ad`
- Combined `name_x_addr`

### B. Set & Token Features (61 columns)

| Category | Key Features |
|---|---|
| **Name overlap** | `core_jac`, `core_ovl_min`, `concat_eq`, `concat_sub`, `first_tok_eq`, `core_cov1/2` |
| **Legal forms** | `legal_common`, `legal_conflict` |
| **House numbers** | `num_common`, `first_num_eq`, `first_num_sim`, `best_num_sim` (1-digit edits, truncation, leading zeros) |
| **ZIP/Address** | `zip_eq`, `zip_conflict`, `alph_jac`, `alph_fuzzy_cov1`, `ad_cov2`, `ad_empty2` |
| **Edit distances** | Levenshtein, Jaro-Winkler, token-sort ratio, partial ratio (via `rapidfuzz`) |
| **Phonetic** | Soundex coverage `sdx_cov1/2` — catches "Shri/Shree", "Laxmi/Lakshmi" |

### C. Rarity Features (Biggest Single Gain: +0.0066 F0.5)

"Same name" means very different things for *"Obsidian Quartz Fabricators"* and *"Sri Ganesh Stores"*.
For a rare name, a name match is near-proof. For a common one, the address is essential.

- `nmfreq_s1_in_s1`, `nmfreq_s1_in_c`, `nmfreq_c_in_c`, `nmfreq_c_in_s1`
- `adfreq_s1_in_c`, `adfreq_c_in_c`, `adkey_eq`

### D. Sibling-Word Features

When two names differ by one extra word, the word itself tells you a lot:
- **Supervised table:** P(non-match | extra word) — *exports: 0.99, ventures: 0.96, group: 0.83* vs
  *services: 0.31, center: 0.46, dba: 0.09*
- **Unsupervised address-agreement table:** Do records differing by this word share the same house number?
  Sibling words almost never share the address. **Works for France without labels.**

### E. Context & Competition Features

- Blocking scores and ranks (`bscore`, `brank`, `b_rel`)
- Competition: how many S1 entities claim this candidate (`n_s1_for_cand`, `cand_gap_best_other`)
- Group-relative cosine differences and ranks

**Top LightGBM importances:** `len_ratio_nm`, `nm_sh_idf_max`, `nm_sh_idf`, `b_rel`, `cos_c2_nm`,
`ad_un1_idf`, `name_x_addr_rel1`, `bname`, `cos_c3_ad_rel1`, `bscore`

---

## Handling Unseen Countries (France)

France has **zero training labels**. Our strategy:

1. **Statistics without labels:** IDF tables, address-agreement tables, and name/address frequencies are
   built from each country's own corpus — no labels needed.
2. **Leave-one-country-out (LOCO) model:** A strongly regularised GBDT (31 leaves, min_child_samples 300)
   trained by hiding one country at a time:
   - Train on US → test on India: F0.5 = 0.9594
   - Train on India → test on US: F0.5 = 0.9777
3. **Stricter threshold (0.7875):** Both LOCO directions prefer a stricter threshold for unseen countries.
   Probabilities are rescaled as `p' = p × thr / thr_unseen` so one global threshold applies.
4. **France stays GBDT-only:** Cross-encoders were never trained on French text.

---

## Results & Ablation Study

| Step Added | Validation F0.5 | Public Leaderboard |
|---|---|---|
| Baseline: normalisation + token map, multi-key blocking, ~70 features, LightGBM, one-to-one | 0.9738 | 0.9613 |
| + Rarity, sibling-word, edit-distance, phonetic features; LOCO model | 0.9783 | 0.9652 |
| + e5-base cross-encoder, 50/50 blend | 0.9786 | 0.9710 |
| + Cross-encoder ensemble (e5-base + e5-large) | 0.9790 | 0.9719 |
| + Learned candidate ranker (stage B): 10 cand/S1, 97.1% recall | 0.9820 | — |
| **+ Cross-encoder ensemble on new candidates (final)** | **0.9827** | **0.9747** |

The validation ceiling with perfect classification of our candidates is **0.9907**.

**Key insight:** The two biggest jumps came from *rarity features* (+0.0066) and *better candidates* (+0.0030).
Neither is a bigger model — both came from reading our own errors.

---

## Error Analysis

### Remaining False Positives
- Same name at a nearby address with one house-number digit changed (921 → 924)
- Sibling businesses (X Exports vs X)
- Generic chain names appearing at multiple locations

### Remaining False Negatives
- Empty address + common name (no signal to match on)
- Fully replaced alias names (no token overlap)
- Multiple simultaneous typos exceeding fuzzy-match thresholds

### Ceiling Analysis
Even a perfect classifier on our candidates reaches only 0.9907 on validation — more than half the
remaining gap to 1.0 is recall that blocking never recovers.

---

## Project Structure

```
AmazonML/
├── README.md                   ← This file
├── requirements.txt            ← Pinned Python 3.14 dependencies
├── Documentation_template.md   ← Official submission documentation
├── .gitignore
│
├── src/                        ← All source code
│   ├── common.py               ← IO, Normalizer, token-map learning, Brahmic romaniser, F0.5 metric
│   ├── data.py                 ← Data loading, id ↔ int conversion, ground-truth loader
│   ├── prepare.py              ← Token map learning + parallel normalisation → parquet parts
│   ├── block.py                ← Stage A: multi-key inverted-index blocking
│   ├── block2.py               ← Stage B: candidate pool, LightGBM ranker, global-budget selection
│   ├── features.py             ← ~120 pair features, context, competition, sibling-word tables
│   ├── train.py                ← LightGBM matcher + LOCO model, threshold tuning
│   ├── predict.py              ← Test scoring, unseen-country rescaling, writes both TSVs
│   ├── blend_ce.py             ← GBDT + cross-encoder blend, one-to-one, final threshold
│   ├── kaggle_ce.py            ← Cross-encoder training & scoring (all stages: ce1, ce2, ce4, extra)
│   ├── ce_score_local.py       ← Score stage-B-only pairs with ce1 locally
│   ├── make_v7_extra.py        ← Build pair list for second-round cross-encoders
│   ├── analyze_misses.py       ← Blocking miss analysis (guided stage B design)
│   └── run/
│       ├── run_final.bat       ← Windows driver for the full local pipeline
│       └── kaggle_ce_notebook.ipynb  ← Kaggle notebook cell for cross-encoders
│
├── utils/
│   └── validate_submission.py  ← Official submission validator
│
└── docs/
    ├── problem_statement.md    ← Original Amazon ML Challenge problem statement
    └── solution_details.md     ← Detailed solution writeup from submission
```

**Recommended reading order:** `common.py` → `block.py` → `block2.py` → `features.py` → `train.py` → `blend_ce.py`

---

## Setup & Installation

### Prerequisites

- **Python 3.14** (the version used for development and submission)
- **~60 GB disk space** for the work directory (intermediate parquet files)
- **GPU** (optional): Only cross-encoder steps require one (Kaggle 2×T4 GPUs used)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU-accelerated PyTorch (CUDA 12.6):
```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

### Dataset

Download the dataset from Kaggle:

👉 **[Amazon ML Challenge 2026 — BER Dataset](https://www.kaggle.com/datasets/omkarprabhuuuu/ber-data)**

```bash
# Using Kaggle CLI
kaggle datasets download -d omkarprabhuuuu/ber-data
unzip ber-data.zip -d dataset/
```

Or download manually and place it at:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## How to Reproduce

All paths below use the Windows layout. The same commands work on Linux with `/` paths.
Set these variables to match your setup:

```
DATA = path to dataset/ folder
WORK = scratch folder (≈ 60 GB)
KOUT = folder holding cross-encoder score files
```

### Full Pipeline (Steps 1–6)

```bash
# 1. Normalise all records (+ learn Indic→Latin token map from train ground truth)
python src/prepare.py --data DATA --work WORK --workers 10

# 2. Stage A: multi-key blocking (also produces cross-encoder training pairs)
python src/block.py --work WORK --split train --workers 6
python src/block.py --work WORK --split test  --workers 6

# 3. Stage B: learned blocking — pool → ranker → global budget of 10 candidates/S1
python src/block2.py pool  --work WORK --split train --workers 6
python src/block2.py pool  --work WORK --split test  --workers 6
python src/block2.py fit   --work WORK --data DATA --workers 10
python src/block2.py apply --work WORK --data DATA --split train --workers 10 --budget 10
python src/block2.py apply --work WORK --data DATA --split test  --workers 10 --budget 10

# 4. LightGBM matcher (+ LOCO model for unseen countries) and test scoring
python src/train.py   --work WORK --data DATA --workers 10 \
       --blocks blocks_train_v7 --K 1000 --side 0 --loco 1 --iters 3000
python src/predict.py --work WORK --data DATA --out OUT_V7 --workers 8 \
       --blocks blocks_test_v7

# 5. Cross-encoders — see "Cross-Encoders on Kaggle" section below
#    Produces: KOUT/ce_val.parquet, ce_test_*.parquet, ce2_*/ce4_* files
python src/ce_score_local.py --work WORK --ce_dir KOUT
python src/make_v7_extra.py KOUT

# 6. Blend: 0.5×GBDT + 0.5×mean(ce1,ce2,ce4) → one-to-one → threshold
echo 0.5 > KOUT/blend_w.txt
python src/blend_ce.py --work WORK --art WORK --data DATA \
       --ce_dir KOUT --out OUT_FINAL --tags ce1,ce2,ce4
cp OUT_V7/candidate_pairs.tsv OUT_FINAL/

# Validate
python utils/validate_submission.py \
       --matching OUT_FINAL/matching_results.tsv \
       --candidate OUT_FINAL/candidate_pairs.tsv \
       --test-dir DATA/test
```

Or use the Windows batch driver:
```bash
# Edit ROOT at the top of the file first
src/run/run_final.bat
```

### Expected Output

| Step | Expected Log Line |
|---|---|
| `block2.py apply` | `ranker recall 0.9709 at 10 candidates/S1` |
| `train.py` | `matcher validation F0.5 0.98195` |
| `blend_ce.py` | `BEST blend: F0.5=0.98273 w_gbdt=0.5 thr=0.65 CE=ce1+ce2+ce4` |

Small differences in the last digit are normal (thread ordering, GPU non-determinism).

---

## Cross-Encoders on Kaggle

The cross-encoders are trained and scored on **Kaggle GPU (2×T4, Internet on)**:

1. Upload `src/*.py` and the competition `dataset/` as a private Kaggle dataset
2. Run the cell in [`src/run/kaggle_ce_notebook.ipynb`](src/run/kaggle_ce_notebook.ipynb) — it runs
   `prepare.py` → `block.py` → `kaggle_ce.py --stage all` (ce1, ce2, ce4 in sequence)
3. If a Kaggle session times out (12h cap), re-run with `--prev DIR` pointing to the earlier output —
   completed stages are auto-skipped
4. For extra pairs from stage B: `kaggle_ce.py --stage extra --prev DIR --extra v7_extra_pairs.parquet`
5. Download the output `.parquet` files into `KOUT`

> **Tip:** Use *Save Version → Save & Run All (Commit)* rather than interactive sessions — committed runs
> persist when the browser is closed, and output files remain accessible.

Model checkpoints (~1.1–2.2 GB each) are **not included** in this repo. They are fully reproducible from
the training data with fixed seeds using the commands above.

---

## Runtime & Hardware

| Step | Time | Hardware |
|---|---|---|
| `prepare.py` + `block.py` + `block2.py` | ~3 hours | CPU (16 GB RAM) |
| `train.py` (LightGBM, 700K S1, 3,000 trees) | ~4 hours | CPU |
| `predict.py` (17.3M test pairs) | ~1.5 hours | CPU |
| Cross-encoders (ce1, ce2, ce4) | ~5–6 hours each | Kaggle 2×T4 GPU |
| `blend_ce.py` | ~5 minutes | CPU |

---

## Key Design Decisions

1. **Learned candidate budget, not fixed K** — One global cutoff on a ranker score gives higher recall
   with fewer pairs (97.1% at 10/S1 vs 95.6% at 13/S1).

2. **GBDT + cross-encoder blend** — The GBDT carries hand-crafted signals (rarity, house numbers, sibling
   words). Cross-encoders read both records jointly and catch things we didn't think of.

3. **One-to-one assignment** — Every S2/S3 record has at most one owner in the ground truth. Enforcing
   this removes many false merges for free.

4. **Simple decision rule** — One global threshold (0.65) on blended score. One parameter means it
   transfers better to the harder test distribution.

5. **Country as open set** — Blocking, IDF, and address-agreement tables are built per country without
   labels. Unseen countries get a regularised LOCO model with a stricter threshold.

6. **Learned only from provided data** — The transliteration map, extra-token table, and all models are
   learned from competition data only. No external lookups.

---

## What We Learned

- **Look at the misses, not at the score.** The learned ranker exists because we categorised the lost
  matches. "95.6% recall" told us nothing — "61% of misses have a common name" told us what to build.

- **The metric tells you what to fear.** F0.5 punishes false merges more than missed matches. Most of
  our post-processing is a direct answer to that.

- **Understand how the data was made.** Once we saw that true copies keep the house number while decoys
  change it or add a sibling word, the fuzzy house-number features and sibling-word tables followed
  naturally.

- **Respect the unseen country.** Everything that requires labels is at risk on France. Statistics
  that need no labels (IDF, address agreement, frequencies) were the safest gains.

---

## Compliance

| Component | License | Size |
|---|---|---|
| LightGBM | MIT | — |
| scikit-learn | BSD | — |
| rapidfuzz | MIT | — |
| `intfloat/multilingual-e5-base` | MIT | 278M params |
| `intfloat/multilingual-e5-large` | MIT | 560M params |

- ✅ All models far below the 8B parameter limit
- ✅ No code from other teams was used
- ✅ No external data, APIs, or lookups — all normalisation tables are static rules; every learned map
  comes from the provided training data only

---

## Team

**Team zippy** — Amazon ML Challenge 2026

| Member | Institution | Role |
|---|---|---|
| **Omkar Prabhu** | M.S. Ramaiah University of Applied Sciences | Team Leader |
| **Shreyas Poojari** | BMS Institute of Technology and Management, Bangalore | Member |
| **Mann Bhardwaj** | M.S. Ramaiah University of Applied Sciences | Member |

**Submitted:** 2 October 2026

---

<p align="center">
  <i>Built with ☕ and a lot of error analysis.</i>
</p>

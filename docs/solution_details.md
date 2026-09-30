# Business Entity Resolution — Amazon ML Challenge 2026 — Team **zippy**

**Team:** Omkar Prabhu (leader, M.S. Ramaiah University of Applied Sciences) · Shreyas Poojari (BMS Institute of
Technology and Management) · Mann Bhardwaj (M.S. Ramaiah University of Applied Sciences)

**Submitted:** 2 October 2026

**Final result (best submission):** public leaderboard macro F0.5 **0.974667**; validation **0.98273** on 40k
held-out training entities (singletons included).

---

## 0. Summary

For every Source-1 (S1) business we find its copies among the Source-2/3 (S2/S3) records in five steps.

1. **Normalise** every record: legal forms, abbreviations, accents, "leetspeak", and Indic scripts transliterated
   with a token map learned from the training ground truth.
2. **Generate candidates** in two stages:
   - A. A multi-key inverted index proposes a pool with query expansion.
   - B. A learned LightGBM ranker keeps **10 candidates per S1 on average at 97.1 % pair recall**, using a single
     global budget, so easy entities keep few candidates and hard ones many.
3. **Score pairs** with a LightGBM matcher on ~120 similarity, rarity, sibling-word and context features.
4. **Blend** 50/50 with an ensemble of fine-tuned multilingual cross-encoders (e5-base, e5-large ×2, MIT licence).
5. **Decide**: every S2/S3 record goes to at most one S1 (one-to-one, as in the ground truth), and a pair is
   accepted if its blended score passes one global threshold tuned for macro F0.5 on validation. France, which is
   absent from training, is scored by a regularised leave-one-country-out model.

```
raw TSVs ─► prepare.py ─► normalised parquet parts (+ learned token map)
              ▼
         block.py   stage A: hashed multi-key inverted index per country, IDF key-overlap score
         block2.py  stage B: pool (top-30 + side slots + query expansion) → LightGBM ranker
              │             → global budget: 10 candidates / S1        ═► output/candidate_pairs.tsv
              ▼
         train.py   ~120 pair features → LightGBM matcher (+ leave-one-country-out model for unseen countries)
         predict.py test scoring
              ▼
         kaggle_ce.py / ce_score_local.py    cross-encoders ce1 (e5-base), ce2 (e5-large), ce4 (e5-large, round 2)
              ▼
         blend_ce.py 0.5·GBDT + 0.5·mean(CE) → one-to-one → global threshold
                                                                         ═► output/matching_results.tsv
```

**The idea in one picture.** The example below is illustrative (made up in the style of the data), but it is
exactly the situation the pipeline is built for:

```
S1   Shree Laxmi Traders Pvt Ltd | 921 MG Road, Bengaluru 560001

S2   Shri Lakshmi Traders Private Limited | 921, M.G. Rd, Bangalore 560001     ← true copy (spelling, legal form)
S3   श्री लक्ष्मी ट्रेडर्स | 0921 MG Road Bengaluru                               ← true copy (script, leading zero)
S2   Shree Laxmi Traders | 924 MG Road, Bengaluru 560001                        ← decoy (house number 921 → 924)
S3   Shree Laxmi Traders Exports Pvt Ltd | 12 Brigade Road, Bengaluru           ← decoy (sibling business)
```

Normalisation makes the first two look alike. The pair models learn that a changed house number ("924") and a
sibling word ("Exports") are bad signs, and the one-to-one rule gives each S2/S3 record to a single owner.

---

## 1. The problem

- **Input.** Source 1 (S1) is the reference list of businesses. Sources 2 and 3 (S2/S3) are noisy copies plus distractors.
- **Task.** For every S1 entity, output the S2/S3 records that are the same business.
  - `output/matching_results.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated, may be empty).
  - `output/candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids` (our blocking output; a smaller candidate set is rewarded).
  - Both are checked by `student_resource/utils/validate_submission.py`.
- **Metric: macro F0.5 per S1 entity** (β = 0.5, so precision counts twice as much as recall).
  - S1 entities with no true match (singletons) score 1 only if we predict nothing for them.
  - Consequence: a false merge hurts more than a missed match.
- **Rules.**
  - Models must be MIT/Apache licensed and ≤ 8B parameters.
  - No external data or lookups.
  - Country is an open set: **France appears only in test.**
- **Deliverable zip.**
  - `output/`
  - `code/business_entity_resolution/{src, README.md, requirements.txt}`
  - `Documentation_template.md`

### Data facts found in EDA (they drive every design choice)

We looked hard at the data before building anything, and it paid off. Almost every component below exists because
of one line in this list.

- **Train size:** 2.21M S1, 5.03M S2 and 5.29M S3 records (US + India).
- **Test size:** 1.73M S1 entities.

  | Country | S1 entities | Share of test |
  |---|---|---|
  | India | 809,986 | 46.8% |
  | US | 663,106 | 38.3% |
  | France | 259,452 | 15.0% |

- **Matches per entity:**
  - An S1 entity has 3.5 matches on average (range 0–13).
  - 5.6% of S1 entities are singletons.
- **Every S2/S3 record belongs to at most one S1 entity** (7.64M matches, all unique). This gives the one-to-one post-processing.
- **Distractors:**
  - About 27% of S2/S3 records in train are unmatched distractors.
  - Test has 5.8 records per S1 entity (train 4.7), so it has more distractors.
  - Test contains many **"sibling" look-alike businesses**: *X Exports / X Group / X Ventures / X Holdings*, and in France *Groupe X / X Holding / X Participations / X Développement / X International / X Distribution*.
- **Country:** matches always share the country, so blocking is done per country.
- **Noise types:**
  - legal-form swaps and permutations ("Shree Ltd Pvt Entertainment")
  - junk prefixes ("M/s", "--", "<<", "##")
  - DBA / F/K/A constructions and domain-style names ("capitalholding.com")
  - leetspeak ("0bsidian", "a1ltime")
  - appended phone numbers and filler words ("Center", "Services")
  - names in Devanagari, Gujarati, Bengali, Tamil, Telugu or Kannada script, and native-script state names
  - house-number edits (dropped or edited digits, leading zeros) and reordered address components
  - missing addresses (3.4% of S2)
  - generated hard negatives: the same name at a nearby address
- **Name reuse:** 12% of normalised S1 names are not unique, so name-only blocking keys are too frequent and blocking must combine name and locality.
- **No shortcuts:** we checked that entity ids and row order carry no signal about matches.

---

## 2. Results

Validation = macro F0.5 on 40k held-out training S1 entities (singletons included); leaderboard = public score.

| Step added | Validation | Leaderboard |
|---|---|---|
| Baseline: normalisation + token map, multi-key blocking, ~70 features, LightGBM, one-to-one, tuned threshold | 0.9738 | 0.9613 |
| + rarity, sibling-word, edit-distance and phonetic features; leave-one-country-out model for France | 0.9783 | 0.9652 |
| + e5-base cross-encoder, 50/50 blend | 0.9786 | 0.9710 |
| + cross-encoder ensemble (e5-base + e5-large) | 0.9790 | 0.9719 |
| + learned candidate ranker (stage B): 10 candidates/S1 at 97.1 % recall, matcher retrained on it | 0.9820 | – |
| **+ cross-encoder ensemble on the new candidates (final submission)** | **0.9827** | **0.9747** |

The validation ceiling (perfect classification of our candidates) is 0.9907.

**How to read this table.**

- Each row is the previous row plus one idea, so the difference between two rows is what that idea was worth.
- The two biggest jumps came from *features that describe rarity* (row 2) and from *better candidates* (row 5).
  Neither is a bigger model. Both came from reading our own errors.
- The leaderboard is about 0.01 below validation throughout. Two reasons: France has no training labels, and the
  test set has more distractors per entity than train.

---

## 3. Method in detail

### 3.1 Normalisation — `src/common.py` (`Normalizer`), `src/prepare.py`

The goal here is simple: two records that a human would call "the same string, written differently" should
become the same tokens before any model sees them.

Each record is turned into several fields: `nm` (full normalised name), `core` (name without legal
forms or filler), `ad` (address), `nums` (house numbers, PIN/ZIP), `alph` (alphabetic address tokens)
and flags `is_dom` / `has_dba`.

- **Case and characters:** lowercase, Unicode accent strip, punctuation cleanup, junk-prefix removal.
- **Leetspeak repair:** digits inside words are mapped back to letters (0→o, 1→l/i, 3→e, 4→a, 5→s, 7→t).
- **Dotted acronyms:** `S.A.R.L.` → `sarl`, `P.V.T.` → `pvt`.
- **Legal-form map:**
  - Pvt Ltd / Private Limited / LLC / Inc / Corp / SARL / SAS / SA / GmbH … are mapped to canonical tokens.
  - They are stripped from `core` but kept for the legal-agreement features.
- **Address abbreviations and states:**
  - street → st, road → rd, avenue → ave …
  - US and Indian state names ↔ codes, including native-script state names.
- **French address articles** (de, du, des, la, le, les …) are removed from `alph`.
- **Indic transliteration:**
  - (a) A **token map learned from training ground-truth pairs**. When a native-script token aligns with a Latin token in matched pairs, we learn the mapping. Stored in `token_map.json` and learned only from provided train data.
  - (b) A **rule-based Brahmic romaniser** that covers every Indic Unicode block as a fallback.
- **Storage:** data is split into 1M-row parquet part files (`*.p0.parquet …`) so 16 GB of RAM is enough.

Why learn the token map instead of using a transliteration library? Because the data has its own spelling habits
("Shri/Shree", "Laxmi/Lakshmi"), and the matched training pairs show us exactly which Latin spelling each
native-script word was generated from. It also keeps us inside the "no external data" rule.

### 3.2 Candidate generation, stage A: multi-key inverted index — `src/block.py`

With 1.7M S1 entities and about 10M S2/S3 records, comparing everything with everything is impossible. Blocking
decides which pairs are even looked at, so a true match lost here is lost for good.

- **Keys:** per record, a set of hashed keys (32-bit hash packed with the row index into `uint64`), with a per-country inverted index.

  | Key | Meaning |
  |---|---|
  | `c:` | concatenated core name ("capital holding" == "capitalholding.com") |
  | `s:` | sorted unique core tokens |
  | `p:` | unordered pairs of 4-character token prefixes (typo and word-order robust) |
  | `u:` | single token prefix (only effective when rare) |
  | `n:` | concatenated core × locality token |
  | `m:` | name-prefix × locality token |
  | `a:` | house number × street/city token prefix |
  | `x:` | name-prefix × house number |
  | `z:` | ZIP/PIN × name token |

- **Score:** each shared key adds `idf = log(1 + N/df)`. Keys with df > 300 are ignored as too common. Scores are kept separately for the name, address and mixed channels.
- **Selection:** the top-12 by total score, plus the top-3 by name-only score, plus the top-3 by address-only score. This rescues aliases and address-less records.
  - Result: ≈ 12.98 candidates per S1 and 28.6M train pairs. Test: 22.8M pairs.
- **Recall on held-out train (stage A alone):**

  | Setting | Recall | Avg candidates per S1 |
  |---|---|---|
  | top-12 only | 95.3% | 12 |
  | **top-12 + side slots (used)** | **95.6%** | **13** |
  | top-20 + 3 side | 96.5% | 20.7 |
  | top-40 + 5 side | 97.1% | 40.6 |

  Stage B (below) raises recall to 97.1% with only 10 candidates per S1.
- **Memory:** queries are processed in chunks sized so that each chunk expands to a bounded number of pairs, and the index is built from the 1M-row part files, so memory stays flat.

### 3.3 Candidate generation, stage B: learned candidate ranker — `src/block2.py`

The table above shows the dilemma: to get 97% recall with stage A alone we would need 40 candidates per entity,
and the challenge rewards a small candidate set. So we asked a different question: instead of "how many
candidates per entity?", "which candidates, across all entities, are worth keeping?".

- **Why.** A miss analysis of stage A (`src/analyze_misses.py`) showed the lost true matches were mostly
  (a) S1 entities with a very common name (61% of misses), (b) matches ranked 13–60 by the IDF key score,
  (c) records sharing only frequent keys. Simply taking more raw candidates (K = 20/40) costs precision.
- **Pool** (`pool` stage): per S1, top-30 by key score + 3 name-only + 3 address-only side slots +
  **query expansion** (the top-3 candidates are used as extra queries; their best 5 neighbours join the pool).
- **Ranker** (`fit` stage): group/competition features for every pooled pair (score, rank, gap to best,
  name/address channel scores, number of keys, expansion flags, candidate-side competition) → LightGBM
  trained on 150k non-validation train S1 to predict "is a true match".
- **Selection** (`apply` stage): one **global cutoff** on the ranker's *raw* score such that the average is
  10 candidates per S1 (minimum 1 per S1). Easy S1 keep 3–4 candidates, hard S1 keep 20+.
- **Result:** recall 95.6% at 13/S1 (stage A) → **97.09% at 10/S1** (stage B). Validation ceiling 0.987 → 0.9907.
  Fewer candidates *and* higher recall, because the ranker spends the budget where it is needed.
- **Ranking uses the raw ranker score** (`ranker_score`), which is numerically stable for the global cutoff.
- The matcher is trained on these blocks (`--K 1000 --side 0`); the ranker outputs `rp`, `esc`, `from_exp`,
  `rrank` are added as context features.

Query expansion deserves one more sentence. If S1 "A" finds copy "B" easily, then B's own nearest neighbours are
often the harder copies of A (a transliterated one, or one with a missing address). Asking B for its neighbours
reaches records that share no key with A at all.

### 3.4 Pair features — `src/features.py` (≈ 115 columns)

Features are computed per country. IDF tables are fitted on that country's own corpus, so France
gets French IDFs without labels.

**A. Vector similarities** (hashed TF-IDF, 2^20 dims, cosine):
- name char-2/3/4-grams: `cos_c2_nm`, `cos_c3_nm`, `cos_c4_nm`
- core char-3 and word: `cos_c3_core`, `cos_w_core`
- address char-3 and word: `cos_c3_ad`, `cos_w_ad`
- alphabetic-address char-4: `cos_c4_alph`
- combined `name_x_addr`

**B. Set and token features (`SET_COLS`, 61):**
- **Name overlap:** `core_jac`, `core_ovl_min`, `concat_eq`, `concat_sub`, `first_tok_eq`, `name_in_other`, token counts, `core_cov1/2`, `nm_digit_tok2`, `len_ratio_nm`
- **Legal forms:** `legal_common`, `legal_conflict`
- **Numbers:** `num_common`, `num_jac`, `first_num_eq`, `num_conflict_first`, and fuzzy house-number similarity `first_num_sim` / `best_num_sim` (1-digit edits, truncation, leading zeros). Also `num_cov2`, `n_nums1/2`
- **ZIP/PIN and address:** `zip_eq`, `zip_conflict`, `alph_jac`, `alph_fuzzy_cov1`, `ad_cov2`, `last_alph_eq`, address lengths, `ad_empty2`
- **Fuzzy token coverage:** `tok_fuzzy_cov1/2`
- **IDF-weighted shared and unshared mass:** `nm_sh_idf`, `nm_sh_idf_max`, `nm_un1_idf`, `nm_un2_idf`, `ad_sh_idf`, `ad_un1_idf`, `ad_un2_idf`
- **Edit distances** (rapidfuzz, MIT):
  - `lev_nm`: Levenshtein ratio of names
  - `jw_core`: Jaro-Winkler of cores
  - `tsr_nm`: token-sort ratio
  - `pr_nm`: partial ratio
  - `jw_first`: Jaro-Winkler of the first token
  - `lev_ad`, `tsr_ad`: address edit similarity
  - `lev_street`: Levenshtein of the extracted street part
- **Phonetic:** `sdx_cov1/2`, Soundex-code coverage (catches transliteration variants such as "Shri/Shree", "Laxmi/Lakshmi").
- **Sibling-word features** (target the test look-alike businesses):
  - `x1_*`, `x2_*` from a **supervised extra-token table** (`extra_tok.pkl`, learned on train pairs).
    - Scope: pairs whose names overlap ≥ 99% except for a few extra words.
    - It records P(non-match | a token appears in only one name).
    - Examples: exports 0.99, ventures 0.96, group 0.83 versus services 0.31, center 0.46, dba 0.09.
    - Outputs: max and mean sibling score, counts of sibling, noise and unknown extra words.
  - `x*_agree_min`, `x*_n_lowagree` from an **unsupervised address-agreement table** (`agree_{split}_{country}.pkl`).
    - For each extra word, it records how often the two records share the same house number.
    - Sibling words (group, holding, groupe, participations …) almost never share the address, while noise words usually do.
    - It needs no labels and is built from each split and country's own candidate pairs, **so it works for France**.
- **Rarity** (the biggest single gain, +0.0066 F0.5):
  - `nmfreq_s1_in_s1`, `nmfreq_s1_in_c`, `nmfreq_c_in_c`, `nmfreq_c_in_s1`: how many records share the exact core name
  - `adfreq_s1_in_c`, `adfreq_c_in_c`: how many share the exact address key
  - `adkey_eq`
- **Record flags:** `is_dom2`, `has_dba2`, `src` (S2 or S3)

**C. Context and competition features:**
- **From blocking:** `bscore`, `bname`, `baddr`, `nkeys`, `brank`, `rank_n`, `rank_a`
- **Relative to the S1's other candidates:** `b_rel` (score relative to the best candidate of the same S1), `n_cand_s1`
- **Competition:** `n_s1_for_cand` (how many S1 entities claim this candidate), `cand_rank_among_s1`, `cand_gap_best_other`
- **Group-relative similarity:** for `cos_c3_nm`, `cos_c3_ad`, `name_x_addr` and `cos_w_core`, the difference to the S1's best (`_rel1`), the difference to the candidate's best (`_relc`) and the rank within the S1 (`_rk1`)

Top LightGBM importances: `len_ratio_nm`, `nm_sh_idf_max`, `nm_sh_idf`, `b_rel`, `cos_c2_nm`,
`ad_un1_idf`, `name_x_addr_rel1`, `bname`, `cos_c3_ad_rel1`, `bscore`.

**Why rarity matters so much.** "Same name" means very different things for "Obsidian Quartz Fabricators" and for
"Sri Ganesh Stores". For a rare name, a name match is almost proof. For a common one, it is nearly worthless
without the address. Similarity scores alone cannot tell these two cases apart; the frequency counts can.

**The sibling-word tables in plain words.** When two names differ by one extra word, the word itself tells you a
lot. "Services" or "Center" is usually filler added by the noise generator. "Exports" or "Holdings" is usually a
different company. One table learns this from labels; the other learns it without labels, by checking whether
records that differ by that word tend to sit at the same house number. The second one is what carries the idea
over to French words we never saw labelled.

### 3.5 Matcher — `src/train.py`

- **Split:** 40k S1 entities are held out for validation, chosen with a `RandomState(42)` permutation. The Kaggle cross-encoder reproduces the same split.
- **Training data:** candidates of 700k other S1 entities, about 9.08M pairs with 25.5% positives, in the **full training regime** (no S1 removed, no candidate closure).
- **Model:** **LightGBM** binary classifier with 255 leaves, learning rate 0.05, early stopping on the validation logloss and up to 3,000 trees. If LightGBM is missing, it falls back to sklearn HistGradientBoosting.
- **Unseen-country model (LOCO)** for France:
  - A strongly regularised GBDT: 31 leaves, min_child_samples 300, column subsampling.
  - Leave-one-country-out: train on US and test on India (F0.5 0.9594), then the reverse (0.9777).
  - Both directions prefer a stricter threshold than the in-country model, so a new country gets a **higher threshold (0.7875)**.
  - At prediction time, countries not seen in training use this model, and its probabilities are rescaled as `p' = p · thr / thr_unseen` so one global threshold applies.
- **Decision (`postprocess`):**
  1. **One-to-one:** each S2/S3 candidate is kept only for the S1 entity that gives it the highest probability.
  2. Accept if p ≥ threshold.
  3. The threshold is grid-searched (0.20…0.95) for **macro F0.5 on the validation entities, singletons included**.

The France problem is worth a comment. We have no French labels, so we cannot measure anything on France
directly. What we *can* do is pretend: hide India, train on US, and see how the model behaves on a country it has
never seen, then swap. Both directions told the same story: a simpler model and a stricter threshold travel
better. That is the only evidence we trust for France.

### 3.6 Cross-encoder ensemble

The GBDT sees numbers that we designed. A cross-encoder reads the two records as text, side by side, and can pick
up things we did not think of: paraphrased names, unusual abbreviations, mixed scripts.

| Tag | Model (MIT) | Where | Pairs scored |
|---|---|---|---|
| ce1 | multilingual-e5-base, 800k pairs | Kaggle 2×T4 (+ local GPU for pairs proposed only by stage B, `ce_score_local.py`) | all val + all test pairs |
| ce2 | multilingual-e5-large, 1M pairs | Kaggle | val + test pairs where ce1 is not sure (0.002 < p < 0.998) + pairs proposed only by stage B |
| ce4 | e5-large continued from ce2 (500k new pairs, lr 1e-5) | Kaggle | val + uncertain test pairs |

- **Input format:** each record is written as `name | address`, and the pair is fed to the model as two segments.
- **Blend:** `pb = 0.5·p_gbdt + 0.5·mean(ce1, ce2, ce4)` (where a cross-encoder did not score a pair, the mean is
  over the ones that did). The subset ce1+ce2+ce4 was selected on validation (F0.5 0.98273).
- **w = 0.5** (equal weight) scored best on the public leaderboard: the cross-encoders help more on the
  decoy-heavy test set than validation shows.
- **France stays GBDT-only** (the cross-encoders were never trained on French).
- **Why only "uncertain" pairs for the large models?** Free Kaggle GPU time is limited (a weekly quota and a
  12-hour cap per session). The small model is already sure about most pairs, so the large ones spend their time
  only where a second opinion can change the answer.

### 3.7 Decision: one-to-one + global threshold — `src/blend_ce.py`

1. **Blend:** `pb = 0.5·p_gbdt + 0.5·mean(ce1, ce2, ce4)` for US and India.
2. **Unseen countries (France):** the leave-one-country-out GBDT only, with its probabilities rescaled as
   `p' = p · thr / thr_unseen`, so the same global threshold applies to every country.
3. **One-to-one:** each S2/S3 record is kept only for the S1 entity that scores it highest. Every S2/S3 record belongs
   to at most one S1 in the ground truth, so this removes false merges at no cost.
4. **Threshold:** a pair is accepted if its score is at least 0.65. The threshold is grid-searched for macro F0.5 on
   the 40k validation entities, singletons included, after the one-to-one step.

Validation: GBDT alone 0.98195 → blend with the cross-encoder ensemble **0.98273**. Public leaderboard: **0.974667**.

We kept the decision rule this simple on purpose. One threshold has a single number to tune, so it carries over to
the test set, which has more decoys than validation.

---

## 4. Key design decisions

1. **A learned candidate budget instead of a fixed K.** One global cutoff on a ranker score gives higher recall with
   fewer pairs (97.1 % at 10/S1 versus 95.6 % at 13/S1).
2. **One strong GBDT, blended with cross-encoders.** The GBDT carries the hand-made signals (rarity, house numbers,
   sibling words). The cross-encoders read both records jointly and catch paraphrases and transliterations.
3. **One-to-one assignment.** It follows from the ground truth (every S2/S3 record has at most one owner) and removes
   many false merges for free.
4. **A simple decision rule.** One global threshold on the blended score, tuned for macro F0.5 with singletons
   included. It has one parameter, so it transfers to the harder test set.
5. **Country is an open set.** Blocking, IDF and address-agreement tables are built per country without labels, and
   an unseen country gets a regularised model with a leave-one-country-out threshold.
6. **Learned only from the provided data.** The transliteration map, extra-token table and all models are learned
   from the training files. Test-side statistics (IDF, agreement, frequencies) use no labels.

### What we learned along the way

- **Look at the misses, not at the score.** The learned ranker exists because we took the lost matches and sorted
  them into categories. The number "95.6 % recall" told us nothing; "61 % of misses have a common name"
  told us what to build.
- **The metric tells you what to fear.** F0.5 per entity punishes a false merge more than a missed match, and it
  gives a singleton either 1 or 0. Most of our post-processing is a direct answer to that.
- **Understand how the data was made.** Once we saw that copies keep the house number and decoys change it or add
  a sibling word, the fuzzy house-number features and the sibling-word tables followed.
- **Respect the unseen country.** Everything that needs labels is at risk on France. Statistics that need no
  labels (IDF, address agreement, frequencies) were the safest gains we had.

### Where the remaining error is

- **False positives:** the same name at a nearby address with one house-number digit changed; sibling businesses
  (X Exports vs X); generic chain names.
- **False negatives:** an empty address together with a common name; fully replaced (alias) names; several typos
  at once.
- **Ceiling:** even a perfect classifier on our candidates would reach 0.9907 on validation, so more than half of
  the remaining gap to 1.0 is recall that blocking never recovers.

---

## 5. How to run

### 5.0 Before you start

- **Python:** 3.14 (the version we used). Install the pinned packages with `pip install -r requirements.txt`.
- **Disk:** about 60 GB free for the work folder.
- **GPU:** only the cross-encoder steps need one (Kaggle 2×T4, see 5.3).
- **Order matters, restarts do not.** Each step reads the files of the previous one from the work folder, so a
  step that fails can be re-run alone.

### 5.1 Reproduce the final submission

Paths below are the Windows layout (`C:\projects\AmazonML`). `DATA` = the competition `dataset/` folder,
`WORK` = a scratch folder (≈ 60 GB), `KOUT` = the folder holding the cross-encoder score files. The same commands
work on Linux with `/` paths.

```
pip install -r requirements.txt

REM 1. normalise all records (+ learn the Indic->Latin token map from train ground truth)
python src\prepare.py --data DATA --work WORK --workers 10

REM 2. first-stage multi-key blocking (its train blocks are also the cross-encoders' training pairs)
python src\block.py --work WORK --split train --workers 6
python src\block.py --work WORK --split test  --workers 6

REM 3. stage B learned blocking: pool -> ranker -> global budget of 10 candidates / S1
python src\block2.py pool  --work WORK --split train --workers 6
python src\block2.py pool  --work WORK --split test  --workers 6
python src\block2.py fit   --work WORK --data DATA --workers 10
python src\block2.py apply --work WORK --data DATA --split train --workers 10 --budget 10
python src\block2.py apply --work WORK --data DATA --split test  --workers 10 --budget 10

REM 4. LightGBM matcher (+ leave-one-country-out model for unseen countries) and test scoring
python src\train.py   --work WORK --data DATA --workers 10 --blocks blocks_train_v7 --K 1000 --side 0 --loco 1 --iters 3000
python src\predict.py --work WORK --data DATA --out OUT_V7 --workers 8 --blocks blocks_test_v7

REM 5. cross-encoders, trained and scored on Kaggle 2xT4 with one script (see 5.3):
REM    python kaggle_ce.py --work W --data D --out /kaggle/working --stage all
REM    ce1 = e5-base (all stage-A val/test pairs), ce2 = e5-large, ce4 = e5-large continued from ce2
REM    -> KOUT\ce_val.parquet, ce_test_*.parquet, ce2_val/ce2_test_*.parquet, ce4_val/ce4_test_0.parquet, models
python src\ce_score_local.py --work WORK --ce_dir KOUT     (ce1 on the pairs only stage B proposes -> ce_*_new.parquet)
python src\make_v7_extra.py KOUT                           (-> KOUT\v7_extra_pairs.parquet)
REM    on Kaggle: python kaggle_ce.py --work W --data D --out /kaggle/working --stage extra --prev <first run output> --extra v7_extra_pairs.parquet
REM    -> KOUT\ce2_extra.parquet

REM 6. blend: 0.5 * GBDT + 0.5 * mean(ce1, ce2, ce4); one-to-one; threshold tuned on validation; France GBDT-only
REM    -> final matching_results.tsv
echo 0.5> KOUT\blend_w.txt
python src\blend_ce.py --work WORK --art WORK --data DATA --ce_dir KOUT --out OUT_FINAL --tags ce1,ce2,ce4
copy OUT_V7\candidate_pairs.tsv OUT_FINAL\
python student_resource\utils\validate_submission.py --matching OUT_FINAL\matching_results.tsv --candidate OUT_FINAL\candidate_pairs.tsv --test-dir DATA\test
```

`src/run/run_final.bat` runs steps 1–4 and 6 on Windows; step 5 runs on Kaggle (5.3). Edit the `ROOT` path at
the top of the file first. It writes one log per step into `logs\`.

**A note on names.** The suffix `_v7` in `blocks_train_v7`, `OUT_V7` and `make_v7_extra.py` is just the internal
file name of the stage-B candidate set. It is not a different model.

**What you should see if everything went well:**

| Step | Line in the log |
|---|---|
| 3 (`block2.py apply`) | ranker recall 0.9709 at 10 candidates/S1 |
| 4 (`train.py`) | matcher validation F0.5 0.98195 |
| 6 (`blend_ce.py`) | `BEST blend: F0.5=0.98273 w_gbdt=0.5 thr=0.65 CE=ce1+ce2+ce4` |

Small differences in the last digit are normal (thread order in LightGBM, GPU non-determinism in the
cross-encoders).

### 5.2 Run times

| Step | Time |
|---|---|
| prepare + block + block2 (CPU) | ≈ 3 h |
| train (LightGBM, 700k S1, 3,000 trees, CPU) | ≈ 4 h |
| predict (17.3M test pairs, CPU) | ≈ 1.5 h |
| cross-encoders (Kaggle 2×T4) | ≈ 5–6 h each |
| blend | ≈ 5 min |

### 5.3 Cross-encoders on Kaggle (GPU T4 ×2, Internet on)

All cross-encoder work is done by one script, `src/kaggle_ce.py`, called from the single cell in
`src/run/kaggle_ce_notebook.ipynb`.

1. Upload `src/*.py` and the competition `dataset/` as one private Kaggle dataset; attach it with *Add Input*.
2. Run the notebook cell. It runs `prepare.py` → `block.py` (train, test) → `kaggle_ce.py --stage all`, which
   trains and scores the three models one after the other:

   | Stage | Model | Training | Scores | Output |
   |---|---|---|---|---|
   | ce1 | `intfloat/multilingual-e5-base` | 800k pairs, lr 3e-5 | all validation and test pairs | `ce_val.parquet`, `ce_test_*.parquet`, `ce_model/` |
   | ce2 | `intfloat/multilingual-e5-large` | 1M pairs, lr 2e-5 | validation pairs + test pairs where ce1 is not sure | `ce2_val.parquet`, `ce2_test_*.parquet`, `ce2_model/` |
   | ce4 | e5-large continued from `ce2_model` | 500k new pairs, lr 1e-5 | same pairs as ce2 | `ce4_val.parquet`, `ce4_test_0.parquet`, `ce4_model/` |

3. Download the parquet files and `ce_model/` into `KOUT`, then run `ce_score_local.py` and `make_v7_extra.py`
   (5.1 step 5).
4. Upload `v7_extra_pairs.parquet` and run `kaggle_ce.py --stage extra --prev <output of step 2> --extra
   v7_extra_pairs.parquet`. This only scores (no training) and writes `ce2_extra.parquet`. Download it into `KOUT`.

**Session limit.** The three stages together take longer than one 12-hour Kaggle session. The script writes a
`<stage>.done` marker after each stage and skips stages that are already done, so the same command is simply
repeated in a new session with the earlier output attached (`--prev DIR`, or `PREV` in the notebook cell). We ran
the stages in separate sessions for this reason.

Use *Save Version → Save & Run All (Commit)* rather than an interactive session: a committed run keeps going when
the browser is closed, and its output files stay available afterwards.

**Model weights are not in the zip.** The checkpoints (ce1 ≈ 1.1 GB, ce2/ce4 ≈ 2.2 GB each, plus LightGBM
models) are all produced by the commands above from the provided training data only, with fixed seeds.

### 5.4 What ends up in the work folder

| File | Written by | Used by |
|---|---|---|
| `*.p0.parquet …` (normalised parts), `token_map.json` | `prepare.py` | everything |
| `blocks_{split}.parquet` | `block.py` | `block2.py`, cross-encoder training pairs |
| `blocks_{split}_v7` | `block2.py apply` | `train.py`, `predict.py` |
| `extra_tok.pkl`, `agree_{split}_{country}.pkl` | `train.py` (train tables), `predict.py` (test agreement tables) | features |
| `models/ranker.pkl` | `block2.py fit` | `block2.py apply` |
| `models/gbdt_stage1.pkl`, `val_pairs.pkl`, `val_pred_stage1.npy` | `train.py` | `predict.py`, `blend_ce.py` |
| `test_scored_stage1.pkl` | `predict.py` | `blend_ce.py` |
| `ce*_val.parquet`, `ce*_test_*.parquet`, `ce2_extra.parquet` (in `KOUT`) | `kaggle_ce.py`, `ce_score_local.py` | `blend_ce.py` |

### 5.5 If something goes wrong

| Symptom | What to do |
|---|---|
| Out of memory in `block.py` / `block2.py` / `train.py` | Reduce `--workers`. The output is identical, only slower. |
| LightGBM is not installed | `pip install lightgbm`. Without it `train.py` silently falls back to sklearn's HistGradientBoosting and the scores will not match ours. |
| `blend_ce.py` cannot find cross-encoder files | Check that the parquet files from Kaggle are in `KOUT` with their original names. |
| `validate_submission.py` complains about a missing file | `candidate_pairs.tsv` is written by `predict.py` into `OUT_V7`; copy it next to `matching_results.tsv` (last lines of 5.1). |
| A step was interrupted | Re-run only that step. Earlier outputs in the work folder are reused. |

---

## 6. Files

| File | Purpose |
|---|---|
| `src/common.py` | IO, `Normalizer`, token-map learning, Brahmic romaniser, F0.5 metric, part-file helpers |
| `src/data.py` | Loading of normalised fields by split, source, id and country; id ↔ int conversion; ground-truth loader |
| `src/prepare.py` | Learns the token map; normalises every file in parallel into parquet parts |
| `src/block.py` | Stage A multi-key blocking → `blocks_{split}.parquet` |
| `src/block2.py` | Stage B: candidate pool with query expansion, LightGBM ranker, global-budget selection → `blocks_{split}_v7` (file suffix `_v7`) |
| `src/features.py` | Pair features, context and competition features, extra-token and agreement tables |
| `src/train.py` | Candidate cut, featurisation, LightGBM matcher, leave-one-country-out model, threshold tuning |
| `src/predict.py` | Test scoring per country, unseen-country rescaling, post-processing, writes both TSVs |
| `src/kaggle_ce.py` | Cross-encoder training and scoring on Kaggle GPUs, all stages (ce1, ce2, ce4, extra pairs) |
| `src/ce_score_local.py`, `src/make_v7_extra.py` | Score the pairs proposed only by stage B; build the pair list for the second-round cross-encoders |
| `src/blend_ce.py` | GBDT + cross-encoder blend, one-to-one, threshold tuning on validation; writes the final `matching_results.tsv` |
| `src/analyze_misses.py` | Blocking-miss analysis that guided stage B |
| `src/run/run_final.bat` | Windows driver for the whole local pipeline |
| `src/run/kaggle_ce_notebook.ipynb` | The Kaggle notebook cell for the cross-encoders |

If you want to read the code, a good order is `common.py` → `block.py` → `block2.py` → `features.py` →
`train.py` → `blend_ce.py`. The rest are drivers around these.

---

## 7. Compliance

- **Models:**
  - LightGBM (MIT); scikit-learn (BSD)
  - rapidfuzz (MIT)
  - `intfloat/multilingual-e5-base` (MIT, 278M) and `intfloat/multilingual-e5-large` (MIT, 560M), both far below the 8B limit and fine-tuned only on the provided training pairs.
- **No code from other teams** was used.
- **No external data, APIs or lookups.** All normalisation tables are static rules written in code. Every learned map (transliteration tokens, extra-token table) comes from the provided training data only. Test-side statistics (IDF, address agreement, name and address frequencies) use no labels.

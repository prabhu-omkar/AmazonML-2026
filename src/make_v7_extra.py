"""Pairs that only v7's candidate sets contain and on which the v1 cross-encoder was unsure (plus all such
validation pairs) -> kaggle_out/v7_extra_pairs.parquet, to be re-scored by the stronger cross-encoders on Kaggle."""
import sys
import numpy as np
import pandas as pd
ce = sys.argv[1]
t = pd.read_parquet(f"{ce}/ce_test_new.parquet")
u = np.minimum(t.ce.values, 1 - t.ce.values)
t = t.loc[u > 0.002, ["s1", "cand"]].assign(split="test")
v = pd.read_parquet(f"{ce}/ce_val_new.parquet")[["s1", "cand"]].assign(split="val")
e = pd.concat([v, t], ignore_index=True)
e.to_parquet(f"{ce}/v7_extra_pairs.parquet", index=False)
print("val", len(v), "test", len(t), "->", f"{ce}/v7_extra_pairs.parquet")

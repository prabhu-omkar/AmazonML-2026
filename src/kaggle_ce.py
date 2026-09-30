"""Cross-encoder training and scoring on Kaggle GPUs (2x T4) - all stages in one script.

Stages (run in this order by --stage all):
  ce1    multilingual-e5-base, 800k training pairs.  Scores every validation pair and every test pair.
         -> ce_val.parquet, ce_test_{j}.parquet, ce_model/
  ce2    multilingual-e5-large, 1M training pairs.  Scores the validation pairs and the test pairs on which
         ce1 is not sure (min(p, 1-p) > --band).
         -> ce2_val.parquet, ce2_test_{j}.parquet, ce2_model/
  ce4    e5-large continued from ce2_model on 500k new pairs with a lower learning rate.  Same pairs as ce2.
         -> ce4_val.parquet, ce4_test_0.parquet, ce4_model/
  extra  scoring only: ce2_model on an extra pair list (parquet with s1, cand, split in {val, test}); used for
         the pairs that only the stage-B candidate sets contain (see ce_score_local.py, make_v7_extra.py).
         -> ce2_extra.parquet

Assumes prepare.py + block.py have produced WORK/norm_* and WORK/blocks_{train,test}.

  python kaggle_ce.py --work W --data D --out /kaggle/working --stage all
  python kaggle_ce.py --work W --data D --out /kaggle/working --stage all --prev DIR[,DIR]    (continue)
  python kaggle_ce.py --work W --data D --out /kaggle/working --stage extra --prev DIR --extra v7_extra_pairs.parquet

A stage whose outputs already exist in --out or in a --prev folder is skipped, so the same command can be
repeated in a new Kaggle session (12 h cap) with the earlier output attached as --prev.
"""
import argparse
import glob
import os
import time

from common import *
from data import load_fields, gt_int
from train import cut_candidates

# per-stage settings (exactly the settings of the three runs behind the submission)
STAGES = {
    "ce1": dict(prefix="ce", model_dir="ce_model", model="intfloat/multilingual-e5-base", init=None,
                n_train=800000, sample_seed=0, perm_seed=0, lr=3e-5, pct_start=0.06, train_hours=None,
                parallel_train=False, test="all", max_test=None, chunk=3_000_000, dtype=np.float16),
    "ce2": dict(prefix="ce2", model_dir="ce2_model", model="intfloat/multilingual-e5-large", init=None,
                n_train=1000000, sample_seed=1, perm_seed=0, lr=2e-5, pct_start=0.06, train_hours=2.6,
                parallel_train=True, test="uncertain", max_test=5_000_000, chunk=2_000_000, dtype=np.float32,
                cast32=True),
    "ce4": dict(prefix="ce4", model_dir="ce4_model", model="intfloat/multilingual-e5-large", init="ce2_model",
                n_train=500000, sample_seed=2, perm_seed=2, lr=1e-5, pct_start=0.03, train_hours=2.2,
                parallel_train=True, test="uncertain", max_test=None, chunk=None, dtype=np.float32,
                cast32=False),
}
ORDER = ["ce1", "ce2", "ce4"]


def texts_for(work, split, ids_s1, ids_c):
    s1f = load_fields(work, split, [1], set(np.unique(ids_s1)), cols=["entity_id", "nm", "ad"])
    cf = load_fields(work, split, [2, 3], set(np.unique(ids_c)), cols=["entity_id", "nm", "ad"])
    f = lambda n, a: f"{n} | {a}" if a else n
    t1 = pd.Series([f(n, a) for n, a in zip(s1f.nm.values, s1f.ad.values)], index=s1f.entity_id.values)
    t2 = pd.Series([f(n, a) for n, a in zip(cf.nm.values, cf.ad.values)], index=cf.entity_id.values)
    return t1.reindex(ids_s1).to_numpy(dtype=object), t2.reindex(ids_c).to_numpy(dtype=object)


def find(a, name):
    """path of an earlier output (file or folder): --out first, then the --prev folders; None if absent"""
    for d in [a.out] + [p for p in a.prev.split(",") if p]:
        if os.path.exists(f"{d}/{name}"):
            return f"{d}/{name}"
    return None


def need(a, name):
    p = find(a, name)
    assert p is not None, f"{name} not found in --out / --prev (run the earlier stage first)"
    return p


def train_pairs(a, n_train, sample_seed):
    """training pairs (never from the validation S1 entities) with labels; also returns the cut train pairs"""
    gt = gt_int(f"{a.data}/train/train_ground_truth.tsv")
    s1_meta = load_fields(a.work, "train", [1], None, None, cols=["entity_id", "country"])
    rng = np.random.RandomState(42)                       # identical split to train.py
    val_ids = s1_meta.entity_id.values[rng.permutation(len(s1_meta))][:a.n_val]
    pairs = cut_candidates(load_df(f"{a.work}/blocks_train"), 12, 3)
    vmask = pairs.s1.isin(val_ids).values
    V = pairs.loc[vmask, ["s1", "cand"]].reset_index(drop=True)
    T = pairs.loc[~vmask, ["s1", "cand"]]
    T = T.sample(n=min(n_train, len(T)), random_state=sample_seed).reset_index(drop=True)
    T["y"] = np.array([c in gt.get(s, ()) for s, c in zip(T.s1.values, T.cand.values)], dtype=np.int64)
    return T, V


def uncertain_test_pairs(a, cast32, max_test):
    """test pairs on which ce1 is not sure"""
    d = os.path.dirname(need(a, "ce_val.parquet"))
    files = sorted(f for f in glob.glob(f"{d}/ce_test_*.parquet") if f[-9:-8].isdigit())
    assert files, f"no ce_test_*.parquet next to {d}/ce_val.parquet"
    C1 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ce = C1.ce.values.astype(np.float32) if cast32 else C1.ce.values
    u = np.minimum(ce, 1 - ce)
    band = u > a.band
    Tt = C1.loc[band, ["s1", "cand"]].copy()
    Tt["u"] = u[band]
    n_all = len(C1)
    del C1
    if max_test and len(Tt) > max_test:
        Tt = Tt.nlargest(max_test, "u")
    Tt = Tt[["s1", "cand"]].reset_index(drop=True)
    log(f"test pairs to re-score {len(Tt)} of {n_all}")
    return Tt


def make_scorer(a, tok, net, dtype):
    import torch

    def score(A, B, name):
        out = np.zeros(len(A), dtype=dtype)
        order = np.argsort([len(x) + len(z) for x, z in zip(A, B)])
        t1 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for k, i in enumerate(range(0, len(A), a.score_bs)):
                ix = order[i:i + a.score_bs]
                enc = tok(list(A[ix]), list(B[ix]), truncation=True, max_length=a.max_len, padding=True,
                          return_tensors="pt").to("cuda")
                out[ix] = torch.softmax(net(**enc).logits.float(), -1)[:, 1].cpu().numpy()
                if k % 1000 == 0:
                    log(f"  {name}: {i + len(ix)}/{len(A)} ({(i + len(ix)) / (time.time() - t1 + 1e-6):.0f} pairs/s)")
        return out

    return score


def run_stage(a, name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    c = STAGES[name]
    px = c["prefix"]
    if find(a, f"{px}.done"):
        log(f"[{name}] already done, skipped")
        return
    log(f"[{name}] start")
    dev = "cuda"
    T, V = train_pairs(a, c["n_train"], c["sample_seed"])
    log(f"[{name}] train pairs {len(T)} pos-rate {T.y.mean():.3f}; val pairs {len(V)}")
    if c["test"] == "all":
        # the same cut as the training pairs, so the candidates are identical
        Tt = cut_candidates(load_df(f"{a.work}/blocks_test"), 12, 3)[["s1", "cand"]].reset_index(drop=True)
        log(f"[{name}] test pairs {len(Tt)}")
    else:
        V = pd.read_parquet(need(a, "ce_val.parquet"))[["s1", "cand"]]
        Tt = uncertain_test_pairs(a, c["cast32"], c["max_test"])
    ta, tb = texts_for(a.work, "train", T.s1.values, T.cand.values)

    src = need(a, c["init"]) if c["init"] else c["model"]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2).to(dev)
    for n_, p_ in model.named_parameters():       # freeze word embeddings (70% of params)
        if "word_embeddings" in n_:
            p_.requires_grad = False
    multi = torch.cuda.device_count() > 1
    net = torch.nn.DataParallel(model) if (multi and c["parallel_train"]) else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.01)
    steps = len(T) // a.bs + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=c["pct_start"])
    scaler = torch.cuda.amp.GradScaler()
    idx = np.random.RandomState(c["perm_seed"]).permutation(len(T))
    y = T.y.values
    net.train()
    t0 = time.time()
    log(f"[{name}] training {len(T)} pairs from {src}")
    for k, i in enumerate(range(0, len(idx), a.bs)):
        ix = idx[i:i + a.bs]
        enc = tok(list(ta[ix]), list(tb[ix]), truncation=True, max_length=a.max_len, padding=True,
                  return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(net(**enc).logits.float(), torch.tensor(y[ix], device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if k % 500 == 0:
            log(f"step {k}/{steps} loss {loss.item():.4f} {(i + len(ix)) / (time.time() - t0):.0f} pairs/s")
        if c["train_hours"] and time.time() - t0 > c["train_hours"] * 3600:
            log(f"time budget reached at step {k}/{steps}")
            break
    del ta, tb, T
    model.save_pretrained(f"{a.out}/{c['model_dir']}")
    tok.save_pretrained(f"{a.out}/{c['model_dir']}")
    if multi and not c["parallel_train"]:
        net = torch.nn.DataParallel(model)
    net.eval()
    score = make_scorer(a, tok, net, c["dtype"])

    va, vb = texts_for(a.work, "train", V.s1.values, V.cand.values)
    V["ce"] = score(va, vb, "val")
    V.to_parquet(f"{a.out}/{px}_val.parquet", index=False)
    log(f"[{name}] val scored")
    del va, vb
    step = c["chunk"] or max(len(Tt), 1)          # test in chunks (memory) - write each chunk
    for j, st in enumerate(range(0, len(Tt), step)):
        C = Tt.iloc[st:st + step].reset_index(drop=True)
        A, B = texts_for(a.work, "test", C.s1.values, C.cand.values)
        C["ce"] = score(A, B, f"test{j}")
        C.to_parquet(f"{a.out}/{px}_test_{j}.parquet", index=False)
        log(f"[{name}] test chunk {j} saved")
    open(f"{a.out}/{px}.done", "w").write("ok\n")
    del net, model, opt, scaler
    torch.cuda.empty_cache()
    log(f"[{name}] done")


def run_extra(a):
    """scoring only: ce2 on the extra pair list"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    assert a.extra, "--stage extra needs --extra PAIRS.parquet"
    src = need(a, "ce2_model")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2).to("cuda")
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    net.eval()
    score = make_scorer(a, tok, net, np.float32)
    E = pd.read_parquet(a.extra)
    outs = []
    for split, sp in (("val", "train"), ("test", "test")):
        C = E[E.split == split][["s1", "cand"]].reset_index(drop=True)
        if len(C):
            A, B = texts_for(a.work, sp, C.s1.values, C.cand.values)
            C["ce"] = score(A, B, f"extra-{split}")
            C["split"] = split
            outs.append(C)
    pd.concat(outs, ignore_index=True).to_parquet(f"{a.out}/ce2_extra.parquet", index=False)
    log("[extra] extra pairs scored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", help="all (= ce1,ce2,ce4), or a comma list of ce1, ce2, ce4, extra")
    ap.add_argument("--prev", default="", help="comma list of folders with outputs of earlier runs")
    ap.add_argument("--extra", default="", help="pair list for --stage extra")
    ap.add_argument("--n_val", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--score_bs", type=int, default=512)
    ap.add_argument("--band", type=float, default=0.002)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stages = ORDER if a.stage == "all" else [s for s in a.stage.split(",") if s]
    for s in stages:
        assert s in STAGES or s == "extra", f"unknown stage {s}"
    for s in stages:
        if s == "extra":
            run_extra(a)
        else:
            run_stage(a, s)
    log("ALL DONE")


if __name__ == "__main__":
    main()

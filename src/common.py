"""Shared utilities: IO, text normalisation, transliteration, blocking keys, metrics.

Everything here is pure python / numpy / pandas so it runs anywhere.
"""
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                     keep_default_na=False, na_filter=False, encoding="utf-8")
    return df.astype(object)


def save_df(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path + ".parquet", index=False)
    except ImportError:
        df.to_pickle(path + ".pkl")


def _part_no(x):
    return int(re.search(r"\.p(\d+)\.(?:pkl|parquet)$", x).group(1))


def load_df(path, columns=None):
    import glob
    if os.path.exists(path + ".parquet"):
        return pd.read_parquet(path + ".parquet", columns=columns)
    if os.path.exists(path + ".pkl"):
        df = pd.read_pickle(path + ".pkl")
        return df[columns] if columns is not None else df
    parts = sorted(glob.glob(path + ".p*.parquet"), key=_part_no)
    if parts:
        return pd.concat([pd.read_parquet(x, columns=columns) for x in parts], ignore_index=True)
    parts = sorted(glob.glob(path + ".p*.pkl"), key=_part_no)
    out = []
    for x in parts:
        df = pd.read_pickle(x)
        out.append(df[columns] if columns is not None else df)
    if not out:
        raise FileNotFoundError(path)
    return pd.concat(out, ignore_index=True)


def exists_df(path):
    import glob
    return (os.path.exists(path + ".parquet") or os.path.exists(path + ".pkl")
            or bool(glob.glob(path + ".p*.parquet")) or bool(glob.glob(path + ".p*.pkl")))


def src_of(ids):
    """'S2-123' -> 2"""
    return np.array([int(x[1]) for x in ids], dtype=np.int8)


def read_ground_truth(path):
    gt = read_tsv(path)
    out = {}
    for s1, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
        out[s1] = [x for x in m.split(",") if x]
    return out


# --------------------------------------------------------------------------------------
# Transliteration of Brahmic scripts (all Indic blocks share the Devanagari layout)
# --------------------------------------------------------------------------------------
_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_IND_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri",
               0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
               0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}


def _indic_off(ch):
    o = ord(ch)
    if 0x0900 <= o < 0x0D80:
        base = o & ~0x7F
        return o - base
    return None


def translit_indic(word):
    """Rough rule-based romanisation of one Indic-script word (fallback only)."""
    out = []
    i, n = 0, len(word)
    while i < n:
        off = _indic_off(word[i])
        if off is None:
            out.append(word[i]); i += 1; continue
        if off in _CONS:
            c = _CONS[off]
            j = i + 1
            if j < n and _indic_off(word[j]) == _NUKTA:
                j += 1
            nxt = _indic_off(word[j]) if j < n else None
            if nxt == _VIRAMA:
                out.append(c); i = j + 1
            elif nxt in _MATRAS:
                out.append(c + _MATRAS[nxt]); i = j + 1
            else:
                # inherent 'a' (dropped at word end: schwa deletion)
                out.append(c + ("a" if j < n and nxt is not None else "")); i = j
        elif off in _IND_VOWELS:
            out.append(_IND_VOWELS[off]); i += 1
        elif off in _MATRAS:
            out.append(_MATRAS[off]); i += 1
        elif off in _SIGNS:
            out.append(_SIGNS[off]); i += 1
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66)); i += 1
        else:
            i += 1
    return "".join(out)


_NONLATIN_RE = re.compile(r"[ऀ-ൿ]")


def has_indic(s):
    return _NONLATIN_RE.search(s) is not None


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch) or _indic_off(ch) is not None)


LEGAL_MAP = {
    "private": "pvt", "pvt": "pvt", "pvt.": "pvt", "prv": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltd.": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp", "pllc": "pllc", "ltee": "ltd",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "ei": "ei", "scop": "scop", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "nv": "nv", "public": "public", "opc": "opc", "sprl": "sprl", "selarl": "selarl",
}
LEGAL_FORMS = set(LEGAL_MAP.values())
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l",
             "m", "s", "ms", "mr", "mrs", "for", "a", "an", "en", "au", "aux", "&"}
HONORIFIC = {"m/s", "ms", "mr", "mrs", "messrs", "shri", "sri", "smt"}
DBA_RE = re.compile(r"\b(?:d/b/a|dba|t/a|trading as|doing business as|f/k/a|fka|formerly known as|"
                    r"formerly|aka|a/k/a|also known as)\b")
DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|co|biz|info|fr|us|io|org\.in|net\.in)\b")

ADDR_MAP = {
    "street": "st", "str": "st", "st.": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avn": "ave", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "suite": "ste", "apartment": "apt",
    "building": "bldg", "floor": "fl", "flr": "fl", "number": "no", "nr": "near", "opp": "opposite",
    "opposite": "opposite", "sector": "sec", "sect": "sec", "nagar": "ngr", "colony": "col",
    "rue": "rue", "r": "rue", "chemin": "ch", "che": "ch", "impasse": "imp", "allee": "all",
    "route": "rte", "rte": "rte", "faubourg": "fbg", "saint": "st", "sainte": "ste", "ste": "ste",
    "mount": "mt", "fort": "ft", "first": "1", "second": "2", "third": "3", "fourth": "4",
    "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "po": "po", "p.o.": "po", "box": "box", "house": "h", "hno": "h", "h.no": "h",
    "plot": "plot", "dist": "district", "distt": "district", "district": "district",
    "tq": "taluk", "taluka": "taluk", "tal": "taluk",
    "bangalore": "bengaluru", "calcutta": "kolkata", "culcutta": "kolkata", "bombay": "mumbai",
    "madras": "chennai", "gurgaon": "gurugram", "orissa": "odisha", "pondicherry": "puducherry",
    "kolkatta": "kolkata",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "or": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tg": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "ut": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk",
    "jammu & kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "andaman and nicobar islands": "an", "lakshadweep": "ld", "bengal": "wb", "tamil": "tn",
}
_STATE_RE_CACHE = {}


def _multiword_state_re(table):
    key = id(table)
    if key not in _STATE_RE_CACHE:
        multi = sorted([k for k in table if " " in k], key=len, reverse=True)
        _STATE_RE_CACHE[key] = re.compile(r"\b(" + "|".join(re.escape(m) for m in multi) + r")\b")
    return _STATE_RE_CACHE[key]


ADDR_ARTICLES = {"bis", "ter", "de", "du", "des", "la", "le", "les", "l", "d", "au", "aux", "et", "the", "of", "and", "en", "sur"}
_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u0D7F]", re.UNICODE)
_ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th|er|e|eme|ème)$")
_PIN_SPLIT_RE = re.compile(r"\b(\d{3})\s(\d{3})\b")
_ALNUM_SPLIT_RE = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_WS_RE = re.compile(r"\s+")


_DOTTED_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


class Normalizer:
    """Normalises names/addresses.  `tok_map` is a learned native-script->latin token map."""

    def __init__(self, tok_map=None):
        self.tok_map = tok_map or {}

    def _translit_tokens(self, s):
        if not has_indic(s):
            return s
        out = []
        for t in s.split():
            if has_indic(t):
                core = t.strip(".,;:()-#")
                m = self.tok_map.get(core)
                out.append(m if m is not None else translit_indic(core))
            else:
                out.append(t)
        return " ".join(out)

    def base(self, s):
        s = s.lower()
        s = self._translit_tokens(s)
        s = strip_accents(s)
        return s

    # ---------------- names ----------------
    def name(self, raw):
        """returns (clean_string, tokens, core_tokens, flags)"""
        s = self.base(raw)
        is_dom = 0
        m = DOMAIN_RE.search(s)
        if m:
            is_dom = 1
            s = DOMAIN_RE.sub(lambda mm: " " + mm.group(1) + " ", s)
        s = _DOTTED_RE.sub(lambda mm: mm.group(0).replace(".", ""), s)
        has_dba = 1 if DBA_RE.search(s) else 0
        s = s.replace("&", " and ").replace("+", " and ").replace("m/s", " ").replace("@", " at ")
        s = _PUNCT_RE.sub(" ", s)
        toks = []
        for t in s.split():
            if not t.isdigit() and not t.isalpha():
                nd = sum(ch.isdigit() for ch in t)
                if nd <= 2 and len(t) - nd >= 3:
                    t = t.translate(_LEET)
            t = LEGAL_MAP.get(t, t)
            toks.append(t)
        # drop leading honorific
        while toks and toks[0] in HONORIFIC and len(toks) > 1:
            toks = toks[1:]
        core = [t for t in toks if t not in LEGAL_FORMS and t not in NAME_STOP
                and not (t.isdigit() and len(t) >= 4)]
        if not core:
            core = [t for t in toks if t not in NAME_STOP] or toks
        return " ".join(toks), toks, core, is_dom, has_dba

    # ---------------- addresses ----------------
    def address(self, raw):
        """returns (clean_string, tokens, numbers, alpha_tokens)"""
        if not raw:
            return "", [], [], []
        s = self.base(raw)
        s = s.replace("n°", " no ").replace("nº", " no ").replace("#", " ")
        s = _PIN_SPLIT_RE.sub(r"\1\2", s)
        s = s.replace("-", " ").replace("/", " / ")
        for tbl in (US_STATES, IN_STATES):
            s = _multiword_state_re(tbl).sub(lambda mm: tbl[mm.group(1)], s)
        s = _PUNCT_RE.sub(" ", s)
        s = _ALNUM_SPLIT_RE.sub(" ", s) if False else s
        toks, nums, alph = [], [], []
        for t in s.split():
            mo = _ORD_RE.match(t)
            if mo:
                t = mo.group(1)
            t = ADDR_MAP.get(t, t)
            t = US_STATES.get(t, IN_STATES.get(t, t))
            toks.append(t)
            if t.isdigit():
                nums.append(t.lstrip("0") or "0")
            elif any(c.isdigit() for c in t):
                d = re.sub(r"\D", "", t)
                if d:
                    nums.append(d.lstrip("0") or "0")
                alph.append(t)
            elif t not in ADDR_ARTICLES:
                alph.append(t)
        return " ".join(toks), toks, nums, alph


# --------------------------------------------------------------------------------------
# Learn native-script token -> latin token map from training ground truth
# --------------------------------------------------------------------------------------

def _lev_ratio(a, b):
    # small pure-python similarity used only while learning the map
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def _simple_tokens(s):
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    return [strip_accents(t) if not has_indic(t) else t for t in s.split()]


def learn_token_map(pairs, min_count=3, min_ratio=0.3):
    """pairs: iterable of (latin_text, other_text).  Learns other-script token -> latin token."""
    co = defaultdict(Counter)
    cnt = Counter()
    for lat, oth in pairs:
        if not has_indic(oth):
            continue
        lt = set(t for t in _simple_tokens(lat) if not has_indic(t))
        ot_all = _simple_tokens(oth)
        ot_lat = set(t for t in ot_all if not has_indic(t))
        cands = lt - ot_lat
        for t in set(t for t in ot_all if has_indic(t)):
            cnt[t] += 1
            for l in cands:
                co[t][l] += 1
    out = {}
    for t, c in cnt.items():
        if c < min_count:
            continue
        tr = translit_indic(t)
        best, best_s = None, -1
        for l, k in co[t].most_common(8):
            r = k / c
            if r < min_ratio:
                break
            sc = r * (0.35 + _lev_ratio(tr, l))
            if sc > best_s:
                best, best_s = l, sc
        if best is not None:
            out[t] = best
    return out


class Ensemble:
    """average of several fitted binary classifiers (sklearn-style predict_proba)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


# --------------------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------------------

def f05_entity(pred, true):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, gt_map, ids):
    return float(np.mean([f05_entity(pred_map.get(i, []), gt_map.get(i, [])) for i in ids]))


def write_id_lists(path, s1_ids, mapping, colname):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{colname}\n")
        for s in s1_ids:
            lst = mapping.get(s, [])
            seen, out = set(), []
            for x in lst:
                if x not in seen:
                    seen.add(x); out.append(x)
            f.write(s + "\t" + ",".join(out) + "\n")

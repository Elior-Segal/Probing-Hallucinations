"""
Fast automatic hallucination checker.

Install:
pip install sentence-transformers python-dateutil regex
"""

import re
import unicodedata
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer, util

# Load once (global)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast, small
EMB_THRESHOLD = 0.80

# common abstain phrases (lowercased)
ABSTAIN_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "can't answer", "cannot answer", "no idea", "unsure", "i have no information",
    "insufficient information", "i'm unable to", "i am unable to", "unknown",
    "cannot determine", "don't know"
]

NUMERIC_REGEX = re.compile(r"[-+]?\d[\d,\.]*")  # crude numeric finder


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # remove surrounding quotes
    s = s.strip(' "\'`')
    return s


def contains_abstain(output: str) -> (bool, str):
    """
    Checks whether the model output that it doesn't know
    :param output: The output of the model
    :return: whether the model output that it doesn't know
    REMOVED BECAUSE THE MODEL WASN'T FINE-TUNED FOR THIS OR CHAT THEREFOR IT DIDN'T CATCH ANY REAL SCENARIOS
    """
    out = normalize_text(output)
    for phrase in ABSTAIN_PHRASES:
        if phrase in out:
            return True, phrase
    # fuzzy-ish fallback: "i don't have" / "i have no"
    if re.search(r"\b(i (do not|don't) have|i have no|no information)\b", out):
        return True, "fuzzy-abstain"
    return False, ""


def is_exact_match(gold: str, output: str) -> bool:
    """
    Checks whether the model's answer is an exact match to the expected answer
    :param gold: The expected answer
    :param output: The output of the model
    :return: whether the model's answer is an exact match to the expected answer
    """
    g = normalize_text(gold)
    o = normalize_text(output)
    return g != "" and g in o


def numeric_compare(gold: str, candidate: str, abs_tol=1e-6, rel_tol=1e-3) -> bool:
    """Return True if numeric values match within tolerance.
       Handles integers and decimals with commas.
       REMOVED BECAUSE IT WAS TOO NOISY
    """

    def parse_num(s):
        if s is None:
            return None
        s = re.sub(r"[,\s]", "", s)  # remove thousands separators and spaces
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except:
            try:
                return float(s)
            except:
                return None

    # find first number in each
    m1 = NUMERIC_REGEX.search(str(gold))
    m2 = NUMERIC_REGEX.search(str(candidate))
    if not m1 or not m2:
        return False
    n1 = parse_num(m1.group(0))
    n2 = parse_num(m2.group(0))
    if n1 is None or n2 is None:
        return False
    # compare
    if isinstance(n1, (int, float)) and isinstance(n2, (int, float)):
        if abs(n1 - n2) <= abs_tol:
            return True
        # relative tolerance
        denom = max(abs(n1), abs(n2), 1e-9)
        if abs(n1 - n2) / denom <= rel_tol:
            return True
    return False


def date_compare(gold: str, candidate: str) -> bool:
    """
    Try to parse dates and compare normalized date (year-month-day when possible).
    :param gold: The expected answer
    :param candidate: Model's output
    :return: Whether Model's output and the expected answer are the same date.
    """
    try:
        d1 = dateparser.parse(gold, fuzzy=True)
        d2 = dateparser.parse(candidate, fuzzy=True)
        if not d1 or not d2:
            return False
        # Compare date parts: year-month-day if available
        return (d1.year == d2.year) and (d1.month == d2.month) and (d1.day == d2.day)
    except Exception:
        return False


def semantic_match(gold: str, candidates: list[str], model=EMBED_MODEL, threshold=EMB_THRESHOLD):
    """
    Compute semantic similarity between gold and each candidate string.
    :param gold: The expected answer
    :param candidates: Model's outputs
    :param model: The embedding model to use in order to compute semantic similarity.
    :param threshold: The threshold to use in order to determine whether the gold and candidate are close enough.
    :return: (best_score, best_candidate, pass_bool)
    """
    gold_n = normalize_text(gold)
    if gold_n == "":
        return 0.0, "", False
    # encode all at once
    embeddings = model.encode([gold_n] + [normalize_text(c) for c in candidates], convert_to_tensor=True)
    gold_emb = embeddings[0:1]
    cand_embs = embeddings[1:]
    sims = util.cos_sim(gold_emb, cand_embs).cpu().numpy()[0]
    best_idx = int(sims.argmax()) if len(sims) else -1
    best_score = float(sims[best_idx]) if best_idx >= 0 else 0.0
    best_candidate = candidates[best_idx] if best_idx >= 0 else ""
    return best_score, best_candidate, best_score >= threshold


def extract_candidates(output: str, max_sent_len=200):
    """
    Split output into candidate spans to compare to gold.
    Short sentences, lines, or comma-separated chunks.
    """
    out = output.strip()
    # split into sentences and lines
    lines = re.split(r'[\n\r]+', out)
    sentences = []
    for L in lines:
        # split by sentence terminator, but keep short pieces
        parts = re.split(r'(?<=[\.\?\!])\s+', L)
        for p in parts:
            p = p.strip()
            if p:
                if len(p) > max_sent_len:
                    # split long by commas
                    for chunk in p.split(','):
                        if chunk.strip():
                            sentences.append(chunk.strip())
                else:
                    sentences.append(p)
    # fallback: whole output
    if not sentences:
        sentences = [out]
    # trim to reasonable number
    return sentences[:30]


def check_example(gold_answer: str, model_output: str,
                  emb_threshold=EMB_THRESHOLD, numeric_abs_tol=0.0, numeric_rel_tol=1e-3):
    """
    Checks whether the model outputted the expected answer.
    :param gold_answer: The expected answer
    :param model_output: The output of the model.
    :param emb_threshold: The threshold to use for the semantic comparison.
    Returns dict: { 'pass': bool, 'reason': str, 'score': float (where applicable) }
    pass == True means NOT hallucination (either correct or abstain).
    """
    out_norm = normalize_text(model_output)
    gold_norm = normalize_text(gold_answer)

    # 1) abstain detection REMOVED BECAUSE THE MODEL WASN'T FINE-TUNED FOR THIS OR CHAT THEREFOR IT DIDN'T CATCH ANY REAL SCENARIOS
    # abstain, phrase = contains_abstain(model_output)
    # if abstain:
    #     return {'pass': True, 'reason': f'abstain_detected ({phrase})', 'score': None}

    # 2) exact substring match
    if is_exact_match(gold_answer, model_output):
        return {'pass': True, 'reason': 'exact_match', 'score': 1.0}

    # # 3) numeric compare REMOVED FOR BEING TOO NOISY
    # if numeric_compare(gold_answer, model_output, abs_tol=numeric_abs_tol, rel_tol=numeric_rel_tol):
    #     return {'pass': True, 'reason': 'numeric_match', 'score': 1.0}

    # 4) date compare
    if date_compare(gold_answer, model_output):
        return {'pass': True, 'reason': 'date_match', 'score': 1.0}

    # 5) semantic matching across candidate spans
    candidates = extract_candidates(model_output)
    best_score, best_candidate, pass_sem = semantic_match(gold_answer, candidates, threshold=emb_threshold)
    if pass_sem:
        return {'pass': True, 'reason': 'semantic_match', 'score': float(best_score), 'matched_span': best_candidate}

    # 6) if nothing matched -> hallucination
    return {'pass': False, 'reason': 'no_match', 'score': float(best_score) if 'best_score' in locals() else 0.0}


# Batch evaluation helper
def evaluate_batch(pairs, emb_threshold=EMB_THRESHOLD):
    """
    pairs: iterable of (gold_answer, model_output)
    returns: dict with counts and list of per-example results
    """
    results = []
    num_hall = 0
    for gold, out in pairs:
        r = check_example(gold, out, emb_threshold=emb_threshold)
        results.append(r)
        if not r['pass']:
            num_hall += 1
    total = len(pairs)
    rate = num_hall / total if total else 0.0
    return {'total': total, 'hallucination_count': num_hall, 'hallucination_rate': rate, 'details': results}

"""Parse autoresearch/notes.md into a clean per-hypothesis CSV for the paper figures.

Extracts (hypothesis_id, phase, accuracy, delta_vs_previous, outcome, category, change_type)
from the markdown research log. Writes to paper-b/figures/hypotheses.csv.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "log" / "research_log.md"
OUT = Path(__file__).resolve().parents[1] / "log" / "hypotheses.csv"

HEADER_RE = re.compile(r"^###\s+(H\d+[a-z]?|P2-H\d+|P3-H\d+|IN-H\d+|H3-\d+):\s*(.*?)(?:\s+\u2014|$)")
SECTION_END = re.compile(r"^#{2,}\s")  # any markdown header terminates the current entry's accumulation
RESULT_RE = re.compile(r"\*\*Result:?\**\s*.*?([0-9]+\.[0-9]+)\s*%")
STATUS_RE = re.compile(r"(NEW CHAMPION|PUSHED\s*[\u2713\u2714\u2705]?|NOT pushed|DISCARDED|DISCARD\b|RUNNING|PENDING|ongoing)", re.I)

# Manual overrides for entries that don't follow the standard "**Result: XX.XX%**" line
# (e.g. the agent recorded only a delta vs the previous champion, not the raw accuracy).
OVERRIDES = {
    "H67":   {"acc": 96.37, "status": "DISCARDED"},  # title-line only; delta -0.22pp vs H64=96.59
    "H3-18": {"acc": 77.39, "status": "DISCARDED"},  # delta -0.26pp vs IN-B100=77.65
}


def assign_phase(hid: str) -> str:
    if hid.startswith("P3-H") or hid.startswith("IN-H") or hid.startswith("H3-"):
        return "3"
    if hid.startswith("P2-H"):
        return "2"
    if hid.startswith("H"):
        n = int(re.match(r"H(\d+)", hid).group(1))
        return "1b" if n >= 43 else "1"
    return "?"


CATEGORY_KEYWORDS = [
    (re.compile(r"MLP|FFN|SwiGLU|GLU", re.I), "mlp_ffn"),
    (re.compile(r"LR|warmup|weight decay|BASE_LR|WARMUP", re.I), "training_recipe"),
    (re.compile(r"head|multi-head", re.I), "attention_heads"),
    (re.compile(r"CPE|position|pair bias", re.I), "position_encoding"),
    (re.compile(r"drop\s*path|DROP_PATH", re.I), "regularization"),
    (re.compile(r"depth|DEPTH|stage-\d", re.I), "depth_topology"),
    (re.compile(r"DW shortcut|DW_SHORTCUT", re.I), "shortcut"),
    (re.compile(r"Layer\s*Scale|RMSNorm|LayerNorm|QK-?Norm|norm", re.I), "normalisation"),
    (re.compile(r"SiLU|activation", re.I), "activation"),
    (re.compile(r"patch\s*embed", re.I), "patch_embed"),
    (re.compile(r"Mixup|CutMix|RandAugment|RandomErase|label smoothing|augmentation|aug\b", re.I), "augmentation"),
    (re.compile(r"aux|auxiliary|deep supervision", re.I), "aux_supervision"),
    (re.compile(r"cross-?stage|fusion|DuoFormer", re.I), "cross_stage"),
    (re.compile(r"sparse|top-?k|group|factorized|differential|sigmoid|register\s*token|value residual|dual-?scale|gated", re.I), "attention_variant"),
    (re.compile(r"kernel|QKV", re.I), "kernel_size"),
]


def classify(title: str) -> str:
    for pat, label in CATEGORY_KEYWORDS:
        if pat.search(title):
            return label
    return "other"


# Code-level innovations vs config toggles.
# A change is labelled "code" if it introduced a new module/mechanism rather than
# flipping an existing config knob. We err on the side of "config" when ambiguous.
CODE_IDS = {
    # Phase 1 / 1b
    "H30", "H42",  # Layer Scale, QK-Norm (code modules added)
    "H43", "H45", "H46", "H47", "H48", "H50", "H51", "H52", "H53", "H54", "H55",
    "H57", "H59", "H63", "H64", "H65", "H66", "H67",
    # Phase 2
    "P2-H3", "P2-H4", "P2-H8", "P2-H9", "P2-H10", "P2-H16", "P2-H17", "P2-H18",
    "P2-H19", "P2-H20", "P2-H21", "P2-H22", "P2-H23", "P2-H24", "P2-H25",
    "P2-H26", "P2-H27", "P2-H29", "P2-H30", "P2-H31",
    # Phase 3
    "H3-1", "H3-2",
}


def parse() -> list[dict]:
    text = NOTES.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rows = []
    cur = None
    for line in lines:
        # Any markdown section header (## or more) terminates the current entry's content.
        # This prevents an entry without an explicit "**Result:**" line from bleeding into
        # the next section (e.g. H67 was previously absorbing P2-B100's accuracy).
        if SECTION_END.match(line):
            if cur and cur.get("acc") is not None:
                rows.append(cur)
            cur = None
            m = HEADER_RE.match(line)
            if m:
                cur = {"id": m.group(1), "title": m.group(2).strip().strip(")").strip("("), "acc": None, "status": None}
            continue
        if cur is None:
            continue
        rm = RESULT_RE.search(line)
        if rm and cur.get("acc") is None:
            cur["acc"] = float(rm.group(1))
        sm = STATUS_RE.search(line)
        if sm:
            tok = sm.group(1).strip().upper()
            # Normalise synonyms so a later "DISCARDED" overrides an earlier "PUSHED"
            # only if it came later in the block.
            if tok.startswith("PUSHED") or tok == "NEW CHAMPION":
                cur["status"] = "PUSHED"
            elif tok.startswith("DISCARD") or tok.startswith("NOT PUSHED"):
                cur["status"] = "DISCARDED"
            elif tok in {"RUNNING", "PENDING", "ONGOING"}:
                if cur.get("status") is None:
                    cur["status"] = tok
    if cur and cur.get("acc") is not None:
        rows.append(cur)

    # Inject manual overrides for entries the parser cannot recover.
    by_id = {r["id"]: r for r in rows}
    for hid, fix in OVERRIDES.items():
        if hid in by_id:
            by_id[hid].update(fix)
        else:
            # Synthesise a row from the override; title pulled from the markdown header.
            title = ""
            for line in lines:
                m = HEADER_RE.match(line)
                if m and m.group(1) == hid:
                    title = m.group(2).strip().strip(")").strip("(")
                    break
            rows.append({"id": hid, "title": title, **fix})

    # Dedup: keep last occurrence of any ID.
    seen = {}
    for r in rows:
        seen[r["id"]] = r
    rows = list(seen.values())

    # Sort by natural order (phase, numeric index).
    def sort_key(r):
        hid = r["id"]
        if hid.startswith("P2-H"):
            return (2, int(hid.split("H")[1]))
        if hid.startswith("P3-H"):
            return (3, int(hid.split("H")[1]))
        if hid.startswith("IN-H"):
            return (3, 100 + int(hid.split("H")[1]))
        if hid.startswith("H3-"):
            return (3, 200 + int(hid.split("-")[1]))
        n = int(re.match(r"H(\d+)", hid).group(1))
        return (1 if n <= 42 else 1.5, n)

    rows.sort(key=sort_key)

    for r in rows:
        r["phase"] = assign_phase(r["id"])
        r["category"] = classify(r["title"])
        r["change_type"] = "code" if r["id"] in CODE_IDS else "config"

    # Outcome from status (PUSHED = accepted, else rejected).
    # Running champion computed only over accepted hypotheses.
    def scale_key(phase):
        return {"1": "c10", "1b": "c10", "2": "c100", "3": "in1k"}.get(phase, "unknown")

    champion = {}
    for r in rows:
        sk = scale_key(r["phase"])
        prev = champion.get(sk)
        r["prev_champion"] = prev
        st = (r.get("status") or "").upper()
        if st == "PUSHED":
            r["outcome"] = "success"
            champion[sk] = max(prev or 0.0, r["acc"])
        elif st in {"RUNNING", "PENDING", "ONGOING"}:
            r["outcome"] = "pending"
        else:
            r["outcome"] = "failure"
        r["delta"] = (r["acc"] - prev) if prev is not None else 0.0

    # Mark baselines as distinct if known.
    baselines = {
        "Baseline-C10": {"id": "Baseline-C10", "title": "CIFAR-10 baseline", "acc": 69.67,
                          "phase": "1", "category": "baseline", "change_type": "baseline",
                          "delta": 0.0, "outcome": "baseline", "status": None, "prev_champion": None},
        "Baseline-C100": {"id": "Baseline-C100", "title": "CIFAR-100 baseline (P2-H1)", "acc": 81.34,
                           "phase": "2", "category": "baseline", "change_type": "baseline",
                           "delta": 0.0, "outcome": "baseline", "status": None, "prev_champion": None},
        "Baseline-IN1K": {"id": "Baseline-IN1K", "title": "ImageNet-1K baseline (IN-B100)", "acc": 77.65,
                           "phase": "3", "category": "baseline", "change_type": "baseline",
                           "delta": 0.0, "outcome": "baseline", "status": None, "prev_champion": None},
    }
    # Insert baselines at the front of each phase group.
    out = [baselines["Baseline-C10"]]
    for r in rows:
        if r["phase"] == "2" and "Baseline-C100" in baselines:
            out.append(baselines.pop("Baseline-C100"))
        if r["phase"] == "3" and "Baseline-IN1K" in baselines:
            out.append(baselines.pop("Baseline-IN1K"))
        out.append(r)
    return out


def main():
    rows = parse()
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "phase", "acc", "delta", "outcome",
                                            "category", "change_type", "title", "status"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"wrote {len(rows)} rows to {OUT}")
    # Quick sanity summary.
    from collections import Counter
    ph = Counter(r["phase"] for r in rows)
    succ = Counter((r["phase"], r["outcome"]) for r in rows)
    print("per-phase rows:", dict(ph))
    print("per-phase outcome counts:", dict(succ))


if __name__ == "__main__":
    main()

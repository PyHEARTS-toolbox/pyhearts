"""
Generate a clean per-record metadata CSV for the MGH WFDB dataset.

Inputs:
  - WFDB headers: *.hea (contain age/sex/diagnoses + narrative clinical sections)
  - WFDB annotations: *.ari (beat/event symbols + aux notes)

Output:
  - One CSV row per record (mgh###), suitable for joining with PyHEARTS outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_header_first_line(line: str) -> dict[str, Any]:
    """
    Example:
      mgh050 8 360/0.476 1436400 15:47:45 19/07/1991
      mgh001 8 360/0.476 1376831
    """
    parts = line.strip().split()
    out: dict[str, Any] = {}
    if not parts:
        return out

    out["record_name_hdr"] = parts[0]
    if len(parts) >= 2:
        try:
            out["n_sig"] = int(parts[1])
        except Exception:
            out["n_sig"] = ""
    if len(parts) >= 3:
        fs_token = parts[2]
        try:
            out["fs_hz"] = float(fs_token.split("/", 1)[0])
        except Exception:
            out["fs_hz"] = ""
    if len(parts) >= 4:
        try:
            out["n_samples"] = int(parts[3])
        except Exception:
            out["n_samples"] = ""
    if len(parts) >= 5:
        out["start_time"] = parts[4]
    if len(parts) >= 6:
        out["start_date"] = parts[5]
    return out


_RE_AGE_SEX = re.compile(
    r"^#<age>:\s*(?P<age>[^<]+?)\s*<sex>:\s*(?P<sex>[^<]+?)\s*(?:<diagnoses>:\s*(?P<dx>.*))?$"
)
_RE_SECTION = re.compile(r"^#\s*([A-Z][A-Z0-9 \-/&()]+):\s*$")


def parse_age_sex_dx(lines: list[str]) -> dict[str, Any]:
    """
    Parses:
      #<age>: 80 <sex>: F <diagnoses>: Carotid endartarectomy
    Also handles the rare case where the diagnoses tag is missing but
    diagnoses are listed as indented lines immediately below.
    """
    age = sex = dx_tag = None
    dx_fallback_lines: list[str] = []

    for i, raw in enumerate(lines):
        line = raw.strip()
        m = _RE_AGE_SEX.match(line)
        if not m:
            continue
        age = m.group("age").strip()
        sex = m.group("sex").strip()
        dx_tag = (m.group("dx") or "").strip() or None

        # If dx_tag missing, harvest a few indented comment lines below as fallback
        if dx_tag is None:
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip("\n")
                if nxt.strip().startswith("#<") or _RE_SECTION.match(nxt.strip()):
                    break
                if not nxt.startswith("#"):
                    break
                content = nxt[1:].rstrip()
                if content.strip() and (content.startswith("   ") or content.startswith("\t") or content.startswith(" ")):
                    dx_fallback_lines.append(content.strip())
                    j += 1
                    continue
                # stop if we hit a non-indented, non-empty comment line
                if content.strip():
                    break
                j += 1
        break

    dx_fallback = "; ".join(dx_fallback_lines) if dx_fallback_lines else None
    return {
        "age_raw": age or "",
        "sex_raw": sex or "",
        "diagnoses_raw": dx_tag or "",
        "diagnoses_fallback_raw": dx_fallback or "",
        "has_age_sex": bool(age and sex),
        "has_diagnoses": bool(dx_tag or dx_fallback),
    }


def parse_section_headers(lines: list[str]) -> set[str]:
    sections: set[str] = set()
    for raw in lines:
        m = _RE_SECTION.match(raw.strip())
        if m:
            sections.add(m.group(1).strip())
    return sections


def extract_section_text(lines: list[str], section_names: set[str]) -> dict[str, str]:
    """
    Extract raw text content (joined) for selected section headers, e.g.:
      # ECG INTERPRETATION:
      #   Left ventricular hypertrophy

    Returns a mapping of section -> text (lowercased, whitespace-normalized).
    """
    want = {s.strip().upper() for s in section_names}
    out: dict[str, list[str]] = {s: [] for s in want}

    cur: str | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        m = _RE_SECTION.match(line.strip())
        if m:
            sec = m.group(1).strip().upper()
            cur = sec if sec in want else None
            continue
        if cur is None:
            continue
        if not line.startswith("#"):
            continue
        content = line[1:].strip()
        if not content:
            continue
        # Skip nested headers if any
        if _RE_SECTION.match(line.strip()):
            continue
        out[cur].append(content)

    # normalize
    norm: dict[str, str] = {}
    for k, vals in out.items():
        txt = " ".join(vals).strip().lower()
        txt = re.sub(r"\s+", " ", txt)
        norm[k] = txt
    return norm


def derive_ecg_narrative_labels(section_text: dict[str, str]) -> dict[str, Any]:
    """
    Derive coarse, binary cardiac labels from narrative text.
    Searches across relevant sections: ECG INTERPRETATION, PERTINENT HISTORY,
    UNDERLYING RHYTHM, RHYTHM DISTURBANCES, PACEMAKER DATA.
    """
    blob = " ".join([v for v in section_text.values() if v]).strip().lower()
    blob = re.sub(r"\s+", " ", blob)

    def has(pat: str) -> bool:
        return bool(re.search(pat, blob, flags=re.IGNORECASE))

    # Atrial fibrillation
    af = has(r"\batrial fibrillation\b|\bafib\b")

    # Hypertrophy
    lvh = has(r"\bleft ventricular hypertrophy\b|\blvh\b")
    lah = has(r"\bleft atrial hypertrophy\b|\blah\b")
    rvh = has(r"\bright ventricular hypertrophy\b|\brvh\b")

    # Conduction disease
    conduction_defect = has(
        r"\b(conduction defect|intraventricular conduction defect|ivcd|bundle branch block|bbb|lbbb|rbbb)\b"
    )
    # Heart block
    heart_block = has(r"\b(heart block|complete heart block|third degree heart block|3rd degree)\b")

    # Old infarct / MI
    old_mi = has(r"\b(old|previous|prior)\b.*\b(myocardial infarction|infarction)\b|\bold\b.*\bmi\b")

    # ST/T abnormalities (often ischemia proxy, but nonspecific)
    stt_abnormal = has(r"\b(st segment|st-segment|t wave|st-t)\b.*\b(abnormal|abnormalities)\b")

    # Pacemaker / pacing
    pacing = has(r"\b(pacemaker|pacing|paced)\b")

    return {
        "narr_afib": af,
        "narr_lvh": lvh,
        "narr_lah": lah,
        "narr_rvh": rvh,
        "narr_conduction_defect": conduction_defect,
        "narr_heart_block": heart_block,
        "narr_old_mi": old_mi,
        "narr_stt_abnormal": stt_abnormal,
        "narr_pacing": pacing,
    }


def slugify_section(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def sym_col(symbol: str) -> str:
    # make stable columns for odd WFDB symbol chars
    mapping = {
        "/": "slash",
        '"': "quote",
        "=": "eq",
        "~": "tilde",
        "?": "qmark",
        "+": "plus",
        ")": "rparen",
    }
    if symbol in mapping:
        return f"sym_{mapping[symbol]}"
    # keep alnum and underscore
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()
    cleaned = cleaned or "unknown"
    return f"sym_{cleaned}"


def aux_category_counts(aux_notes: list[str]) -> dict[str, int]:
    """
    Coarse, high-signal bucketing for aux notes (free text).
    Counts are case-insensitive.
    """
    counts = Counter()
    for raw in aux_notes:
        if not raw:
            continue
        a = raw.strip()
        if not a:
            continue
        al = a.lower()

        if al == "data on":
            counts["aux_data_on"] += 1
        if al == "data off":
            counts["aux_data_off"] += 1
        if "all data off" in al:
            counts["aux_all_data_off"] += 1
        if "all zero" in al:
            counts["aux_all_zero"] += 1
        if "artifact" in al:
            counts["aux_artifact"] += 1
        if "aberr" in al:
            counts["aux_aberrancy"] += 1
        if "damp" in al:
            counts["aux_damping"] += 1
        if "electrocautery" in al:
            counts["aux_electrocautery"] += 1
        if "no pap" in al:
            counts["aux_no_pap_signal"] += 1
        if "no cvp" in al:
            counts["aux_no_cvp_signal"] += 1
        if "ecg overflow" in al:
            counts["aux_ecg_overflow"] += 1
        if al == "start":
            counts["aux_start"] += 1
    return dict(counts)


def parse_ari_annotations(mgh_dir: Path, record: str) -> dict[str, Any]:
    import wfdb  # local import so the script can still show a useful error if missing

    ann = wfdb.rdann(str(mgh_dir / record), "ari")
    sym_list = list(ann.symbol) if ann.symbol is not None else []
    aux_list = list(ann.aux_note) if ann.aux_note is not None else []

    # normalize symbols: drop NaN
    norm_syms: list[str] = []
    for s in sym_list:
        if isinstance(s, float) and math.isnan(s):
            continue
        if s is None:
            continue
        norm_syms.append(str(s))

    aux_clean = [a.strip() for a in aux_list if isinstance(a, str) and a.strip()]

    sym_counts = Counter(norm_syms)
    aux_counts = Counter(aux_clean)

    top_aux_note = ""
    top_aux_note_count = 0
    if aux_counts:
        top_aux_note, top_aux_note_count = aux_counts.most_common(1)[0]

    out: dict[str, Any] = {
        "n_annotations": int(len(ann.sample)),
        "n_unique_symbols": int(len(sym_counts)),
        "n_unique_aux_notes": int(len(aux_counts)),
        "aux_top_note": top_aux_note,
        "aux_top_note_count": int(top_aux_note_count),
    }

    # symbol counts -> sym_* columns
    for sym, cnt in sym_counts.items():
        out[sym_col(sym)] = int(cnt)

    # coarse aux buckets
    out.update(aux_category_counts(aux_clean))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mgh_dir",
        type=Path,
        default=Path("data/mgh"),
        help="Directory containing mgh###.hea and mgh###.ari files",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/mgh_metadata_summary.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    mgh_dir: Path = args.mgh_dir
    out_csv: Path = args.out_csv

    hea_files = sorted([p for p in mgh_dir.glob("mgh*.hea")])
    if not hea_files:
        raise SystemExit(f"No headers found under {mgh_dir} (expected mgh*.hea).")

    # First pass: parse headers and track section universe
    header_rows: dict[str, dict[str, Any]] = {}
    all_sections: set[str] = set()
    extra_header_cols: set[str] = set()
    missing_ari: list[str] = []

    for hea in hea_files:
        rec = hea.stem
        lines = hea.read_text(errors="replace").splitlines()
        row: dict[str, Any] = {"record": rec}
        if lines:
            row.update(parse_header_first_line(lines[0]))
        row.update(parse_age_sex_dx(lines))

        secs = parse_section_headers(lines)
        all_sections |= secs
        row["_sections"] = secs

        # Narrative-derived labels from selected sections
        section_text = extract_section_text(
            lines,
            {
                "ECG INTERPRETATION",
                "ECG INTERPRETATIONS",
                "UNDERLYING RHYTHM",
                "RHYTHM DISTURBANCES",
                "RHYTHM CHANGES",
                "PERTINENT HISTORY",
                "PACEMAKER DATA",
                "PACEMAKER INFO",
                "PACEMAKER INFORMATION",
            },
        )
        row.update(derive_ecg_narrative_labels(section_text))

        header_rows[rec] = row
        extra_header_cols |= {k for k in row.keys() if k not in {"record", "_sections"}}

        if not (mgh_dir / f"{rec}.ari").exists():
            missing_ari.append(rec)

    # Second pass: parse annotations & track symbol columns universe
    ann_rows: dict[str, dict[str, Any]] = {}
    all_sym_cols: set[str] = set()
    all_aux_cols: set[str] = set()
    failed_ann: dict[str, str] = {}

    for rec in header_rows.keys():
        if not (mgh_dir / f"{rec}.ari").exists():
            continue
        try:
            ann_info = parse_ari_annotations(mgh_dir, rec)
        except Exception as e:
            failed_ann[rec] = repr(e)
            continue
        ann_rows[rec] = ann_info
        for k in ann_info.keys():
            if k.startswith("sym_"):
                all_sym_cols.add(k)
            if k.startswith("aux_"):
                all_aux_cols.add(k)

    # Build final column order
    base_cols = [
        "record",
        "record_name_hdr",
        "n_sig",
        "fs_hz",
        "n_samples",
        "start_time",
        "start_date",
        "age_raw",
        "sex_raw",
        "diagnoses_raw",
        "diagnoses_fallback_raw",
        "has_age_sex",
        "has_diagnoses",
        "n_annotations",
        "n_unique_symbols",
        "n_unique_aux_notes",
        "aux_top_note",
        "aux_top_note_count",
    ]

    # Add any extra header-derived columns (e.g., narrative labels) that aren't already in base_cols.
    for c in sorted(extra_header_cols):
        if c not in base_cols and not c.startswith("_"):
            base_cols.append(c)

    section_cols = [f"has_section_{slugify_section(s)}" for s in sorted(all_sections)]
    sym_cols = sorted(all_sym_cols)
    aux_cols = sorted(all_aux_cols)
    extra_cols = ["ari_missing", "ari_parse_failed", "ari_parse_error"]

    cols = base_cols + section_cols + sym_cols + aux_cols + extra_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for rec in sorted(header_rows.keys()):
            h = header_rows[rec]
            secs: set[str] = h.pop("_sections", set())
            row: dict[str, Any] = defaultdict(str)
            row.update(h)

            # section flags
            for s in all_sections:
                row[f"has_section_{slugify_section(s)}"] = (s in secs)

            # annotation fields
            ann = ann_rows.get(rec, {})
            row.update(ann)

            # status
            row["ari_missing"] = not (mgh_dir / f"{rec}.ari").exists()
            row["ari_parse_failed"] = rec in failed_ann
            row["ari_parse_error"] = failed_ann.get(rec, "")

            # ensure all columns exist
            w.writerow({c: row.get(c, "") for c in cols})

    print(f"Wrote: {out_csv}")
    if missing_ari:
        print(f"Missing .ari for {len(missing_ari)} records (first 10): {missing_ari[:10]}")
    if failed_ann:
        print(f"Failed parsing .ari for {len(failed_ann)} records (first 5): {list(failed_ann.items())[:5]}")


if __name__ == "__main__":
    main()



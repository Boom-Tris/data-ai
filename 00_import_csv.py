# -*- coding: utf-8 -*-
"""
00_import_csv.py — แปลง CSV ที่คนเขียน/ติดฉลากเอง ให้เป็น JSONL รูปแบบเดียวกับที่โน้ตบุ๊กใช้

ใช้กับไฟล์จากชุด data_for_boom (01_all_labeled.csv) หรือ CSV อะไรก็ได้ที่มีคอลัมน์:
    message, category, intent, severity, is_joke, summary   (summary ไม่บังคับ)

ตัวอย่าง
    # เอาเฉพาะแถว split=train ไปต่อท้ายไฟล์เทรน
    python 00_import_csv.py 01_all_labeled.csv --split train --append

    # เอาแถวที่คนติดฉลากแล้วไปเป็น gold eval set (เขียนทับของเดิม)
    python 00_import_csv.py gold_labeled.csv --out eval --overwrite
"""
import argparse
import csv
import json
import sys

import hr_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", choices=["train", "eval"], default="train")
    ap.add_argument("--split", default=None,
                    help="ถ้า CSV มีคอลัมน์ split ให้กรองเอาเฉพาะค่านี้ (เช่น train / gold)")
    ap.add_argument("--append", action="store_true", help="ต่อท้ายไฟล์เดิมแทนการเขียนทับ")
    ap.add_argument("--overwrite", action="store_true", help="เขียนทับไฟล์เดิม")
    args = ap.parse_args()

    target = C.DATA_PATH if args.out == "train" else C.EVAL_DATA_PATH
    if target.exists() and not (args.append or args.overwrite):
        print(f"❌ {target} มีอยู่แล้ว — ใส่ --append หรือ --overwrite ให้ชัดเจนก่อน")
        sys.exit(1)

    mode = "a" if args.append else "w"
    written = skipped = 0
    problems = []

    with open(args.csv_path, encoding="utf-8-sig", newline="") as fin, \
         open(target, mode, encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        cols = {c.strip() for c in (reader.fieldnames or [])}
        for need in ("message", "category", "intent"):
            if need not in cols:
                print(f"❌ CSV ไม่มีคอลัมน์ '{need}' — เจอ: {sorted(cols)}")
                sys.exit(1)

        for i, row in enumerate(reader, start=2):
            if args.split and (row.get("split") or "").strip() != args.split:
                continue

            msg = (row.get("message") or "").strip()
            cat = C.normalize_category(row.get("category"))
            intent = C.normalize_intent(row.get("intent"))
            if not msg or not cat or not intent:
                skipped += 1
                problems.append(f"  บรรทัด {i}: ข้อมูลไม่ครบ (message/category/intent)")
                continue
            if cat not in C.CATEGORY_LABELS:
                problems.append(f"  บรรทัด {i}: category '{cat}' ไม่อยู่ใน taxonomy")
            if intent not in C.INTENT_LABELS:
                problems.append(f"  บรรทัด {i}: intent '{intent}' ไม่อยู่ใน taxonomy")

            assistant = {
                "category": cat,
                "intent": intent,
                "severity": C.normalize_severity(row.get("severity")) or "none",
                "is_joke": bool(C.normalize_is_joke(row.get("is_joke"))),
                "summary": (row.get("summary") or "").strip(),
            }
            record = {"messages": [
                {"role": "system", "content": C.SYSTEM_PROMPT},
                {"role": "user", "content": C.normalize_thai(msg)},
                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
            ]}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ เขียน {written} แถว → {target}  (โหมด: {'ต่อท้าย' if args.append else 'เขียนทับ'})")
    if skipped:
        print(f"⚠️  ข้าม {skipped} แถวเพราะข้อมูลไม่ครบ")
    if problems:
        print(f"\n⚠️  พบ {len(problems)} จุดที่ควรเช็ค:")
        for p in problems[:20]:
            print(p)
        if len(problems) > 20:
            print(f"  … อีก {len(problems) - 20} จุด")


if __name__ == "__main__":
    main()

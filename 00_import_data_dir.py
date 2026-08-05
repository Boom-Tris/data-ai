# -*- coding: utf-8 -*-
"""
00_import_data_dir.py — รวมข้อมูลจากโฟลเดอร์ data (*.jsonl) ทั้งหมดเข้าด้วยกันอย่างอัตโนมัติ

คุณสมบัติ:
  1. ดึงไฟล์ .jsonl ทุกไฟล์ในโฟลเดอร์ data (เช่น hr_ai_dataset.jsonl, hr_dataset.jsonl ฯลฯ)
  2. ทำความสะอาดภาษาไทย (Normalization), กรองข้อความซ้ำ (Deduplication)
  3. ตรวจสอบ Taxonomy (17 หมวด, 5 intent, severity, is_joke)
  4. แบ่งชุดทดลอง Eval Set กระจายครบ 17 หมวด (153 แถว) และชุด Train Set (10,151 แถว)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import hr_common as C


def main():
    data_dir = Path("../data").resolve()
    if not data_dir.exists():
        data_dir = Path("/home/boomtris/Downloads/f/data").resolve()

    C.require(data_dir.exists(), f"ไม่พบโฟลเดอร์ข้อมูลที่ {data_dir}")

    all_sources = sorted(data_dir.glob("*.jsonl"))
    C.require(all_sources, f"ไม่พบไฟล์ .jsonl ใน {data_dir}")

    print("=" * 60)
    print(f"📂 รวมชุดข้อมูลจากโฟลเดอร์: {data_dir.name} ({len(all_sources)} ไฟล์)")
    print("=" * 60)

    seen_users = set()
    records = []

    for src in all_sources:
        print(f"  📄 อ่านไฟล์ {src.name}...", end=" ", flush=True)
        read_n = 0
        with src.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                read_n += 1
                try:
                    rec = json.loads(line)
                    msgs = rec.get("messages", [])
                    assert len(msgs) == 3
                    u = C.normalize_thai(msgs[1]["content"])
                    if not u or u in seen_users:
                        continue
                    ast = json.loads(msgs[2]["content"])
                    cat = C.normalize_category(ast.get("category"))
                    intent = C.normalize_intent(ast.get("intent"))
                    sev = C.normalize_severity(ast.get("severity")) or "none"
                    joke = bool(C.normalize_is_joke(ast.get("is_joke")))
                    summary = str(ast.get("summary", "")).strip()

                    if cat in C.CATEGORY_LABELS and intent in C.INTENT_LABELS:
                        seen_users.add(u)
                        records.append({
                            "category": cat,
                            "intent": intent,
                            "data": {
                                "messages": [
                                    {"role": "system", "content": C.SYSTEM_PROMPT},
                                    {"role": "user", "content": u},
                                    {"role": "assistant", "content": json.dumps({
                                        "category": cat, "intent": intent, "severity": sev,
                                        "is_joke": joke, "summary": summary
                                    }, ensure_ascii=False)}
                                ]
                            }
                        })
                except Exception:
                    pass
        print(f"อ่านแล้ว {read_n} แถว")

    print(f"\n📊 พบข้อมูลภาษาไทยที่ถูกต้องและไม่ซ้ำกันรวม: {len(records)} แถว")

    # สุ่มแบ่ง Eval Set 153 แถวแบบกระจายทุกหมวด (Stratified Evaluation Set)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    eval_records = []
    train_records = []
    rnd = random.Random(42)

    for cat, item_list in by_cat.items():
        rnd.shuffle(item_list)
        eval_count = min(9, len(item_list))
        eval_records.extend(item_list[:eval_count])
        train_records.extend(item_list[eval_count:])

    # เขียนบันทึกเข้า workspace
    with C.EVAL_DATA_PATH.open("w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r["data"], ensure_ascii=False) + "\n")

    with C.DATA_PATH.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r["data"], ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"✅ บันทึกชุด Eval Set  : {len(eval_records)} แถว -> {C.EVAL_DATA_PATH.name}")
    print(f"✅ บันทึกชุด Train Set : {len(train_records)} แถว -> {C.DATA_PATH.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

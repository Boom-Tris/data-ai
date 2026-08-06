# -*- coding: utf-8 -*-
"""
03_evaluate.py — ประเมินผล (แทน cell 33 เดิม)

แก้จากโน้ตบุ๊กเดิม 5 จุด:

  🔴 1. บั๊กที่ทำให้ตัวเลขพองที่สุด — ของเดิมเขียนว่า

           if true_cat and pred_cat:
               all_true_categories.append(true_cat)
               all_predicted_categories.append(pred_cat)

       ถ้าโมเดลตอบ JSON ไม่ได้ pred_cat จะเป็น None แล้ว **ทั้งแถวถูกโยนทิ้ง**
       แปลว่าตัวหารของ accuracy ไม่ใช่ "จำนวนข้อสอบ" แต่เป็น "จำนวนข้อที่โมเดลตอบได้"
       ตอบไม่ได้ = ไม่ถูกนับว่าผิด = ฟรี
       ของใหม่นับเป็นคลาส __PARSE_FAIL__ ซึ่งผิดเสมอ และรายงานจำนวนแยกให้เห็น

  2. วัดแค่ category กับ intent — severity / is_joke ไม่เคยถูกวัดเลยสักครั้ง
     ทั้งที่ severity คือฟิลด์ที่ HR ใช้ตัดสินว่าต้องรีบแก้ไหม
     ของใหม่วัดครบ 4 ฟิลด์ + Exact Match (ทุกฟิลด์ต้องถูกพร้อมกัน)

  3. confusion matrix ถูกวาดเป็นรูปอย่างเดียว อ่านตัวเลขย้อนหลังไม่ได้
     ของใหม่ dump เป็น JSON + CSV ด้วย

  4. รายงานแค่ accuracy — ของใหม่เพิ่ม macro-F1, balanced accuracy,
     Cohen's Kappa, MCC ตามที่จำเป็นเมื่อคลาสไม่สมดุล

  5. ฟอนต์ไทยใช้ !wget ซึ่งเป็นคำสั่ง shell ของ Colab
     ของใหม่หาฟอนต์ในเครื่องก่อน ถ้าไม่มีก็ข้ามการวาดกราฟไปเลย ไม่ล้มทั้งสคริปต์

ตัวอย่าง
    python 03_evaluate.py
    python 03_evaluate.py --limit 20          # ลองสั้นๆ ก่อน
    python 03_evaluate.py --base-only         # วัดโมเดลฐานที่ยังไม่ fine-tune (baseline)
"""
import argparse
import csv
import json
from pathlib import Path

import hr_common as C

FIELDS = ["category", "intent", "severity", "is_joke"]


def find_thai_font():
    """หาฟอนต์ที่วาดภาษาไทยได้ในเครื่อง — ไม่ดาวน์โหลดอะไรทั้งนั้น"""
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        return None
    wanted = ["Noto Sans Thai", "Noto Serif Thai", "TH Sarabun New", "TH SarabunPSK",
              "Sarabun", "Leelawadee UI", "Tahoma", "Ayuthaya", "Thonburi", "Loma", "Garuda"]
    available = {f.name for f in fm.fontManager.ttflist}
    for w in wanted:
        if w in available:
            return w
    return None


def load_eval(path, limit=None):
    C.require(path.exists(), f"ไม่พบไฟล์ประเมิน {path}\n   สร้างด้วย 00_import_csv.py --out eval")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line)["messages"]
                user = next(m["content"] for m in msgs if m["role"] == "user")
                truth = json.loads(next(m["content"] for m in msgs if m["role"] == "assistant"))
                rows.append((user, truth))
            except Exception:
                continue
            if limit and len(rows) >= limit:
                break
    C.require(rows, "ไฟล์ประเมินอ่านไม่ได้เลยสักแถว")
    return rows


def truth_values(truth):
    return {
        "category": C.normalize_category(truth.get("category")),
        "intent": C.normalize_intent(truth.get("intent")),
        "severity": C.normalize_severity(truth.get("severity")),
        "is_joke": C.normalize_is_joke(truth.get("is_joke")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--base-only", action="store_true",
                    help="ไม่โหลด LoRA — ใช้วัด baseline ว่าโมเดลฐานทำได้เท่าไหร่ก่อนเทรน")
    ap.add_argument("--tag", default=None, help="ชื่อกำกับไฟล์ผลลัพธ์")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 cohen_kappa_score, confusion_matrix, f1_score,
                                 matthews_corrcoef)
    from transformers import AutoModelForCausalLM

    tag = args.tag or ("base" if args.base_only else "finetuned")

    C.banner("1) โหลดโมเดล")
    rt = C.detect_runtime()
    C.hf_login_if_possible()
    if args.base_only:
        tokenizer = C.load_tokenizer()
    else:
        C.require(C.ADAPTER_DIR.exists(), f"ไม่พบ adapter ที่ {C.ADAPTER_DIR} — รัน 02_train.py ก่อน")
        tokenizer = C.load_tokenizer(C.ADAPTER_DIR)

    model = AutoModelForCausalLM.from_pretrained(**C.get_load_kwargs(rt))
    if rt["device"] != "cuda":
        model = model.to(rt["device"])
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(C.ADAPTER_DIR))
    model.eval()
    print(f"✅ พร้อมประเมิน ({'base model' if args.base_only else 'fine-tuned'})")

    C.banner("2) รันทำนาย")
    rows = load_eval(C.EVAL_DATA_PATH, args.limit)
    print(f"ชุดประเมิน {len(rows)} ข้อความ")

    y_true = {f: [] for f in FIELDS}
    y_pred = {f: [] for f in FIELDS}
    records, n_parse_fail = [], 0

    for i, (user, truth) in enumerate(rows, 1):
        messages = [{"role": "system", "content": C.SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        enc = C.encode_chat(tokenizer, messages, rt["device"], add_generation_prompt=True)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                       # greedy — ไม่ส่ง temperature มาด้วย
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        del out, enc
        if rt["device"] == "cuda":
            torch.cuda.empty_cache()

        parsed = C.parse_json_object(text)
        t = truth_values(truth)
        if parsed is None:
            n_parse_fail += 1
            p = {f: C.PARSE_FAIL for f in FIELDS}
        else:
            p = {
                "category": C.normalize_category(parsed.get("category")) or C.PARSE_FAIL,
                "intent": C.normalize_intent(parsed.get("intent")) or C.PARSE_FAIL,
                "severity": C.normalize_severity(parsed.get("severity")) or C.PARSE_FAIL,
                "is_joke": C.normalize_is_joke(parsed.get("is_joke")),
            }
            if p["is_joke"] is None:
                p["is_joke"] = C.PARSE_FAIL

        for f in FIELDS:
            tv = t[f] if t[f] is not None else "__NO_TRUTH__"
            y_true[f].append(str(tv))
            y_pred[f].append(str(p[f]))

        records.append({
            "message": user,
            **{f"true_{f}": str(t[f]) for f in FIELDS},
            **{f"pred_{f}": str(p[f]) for f in FIELDS},
            "exact_match": all(str(t[f]) == str(p[f]) for f in FIELDS),
            "raw_output": text[:600],
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    C.banner("3) ผลลัพธ์")
    if n_parse_fail:
        print(f"🔴 โมเดลตอบ JSON ไม่ได้ {n_parse_fail}/{len(rows)} แถว "
              f"({100*n_parse_fail/len(rows):.1f}%) — นับเป็นตอบผิดทั้งหมด")
        print("   (โน้ตบุ๊กเดิมโยนแถวพวกนี้ทิ้ง ทำให้ accuracy สูงกว่าความจริง)")
    else:
        print("✅ โมเดลตอบ JSON ได้ครบทุกแถว")

    summary = {"tag": tag, "n_eval": len(rows), "n_parse_fail": n_parse_fail, "fields": {}}
    font = find_thai_font()

    for f in FIELDS:
        yt, yp = y_true[f], y_pred[f]
        labels = sorted(set(yt) | set(yp))
        acc = accuracy_score(yt, yp)
        macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        weighted_f1 = f1_score(yt, yp, average="weighted", zero_division=0)
        try:
            bal = balanced_accuracy_score(yt, yp)
        except Exception:
            bal = float("nan")
        kappa = cohen_kappa_score(yt, yp)
        try:
            mcc = matthews_corrcoef(yt, yp)
        except Exception:
            mcc = float("nan")

        print(f"\n=== {f} ===")
        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Balanced Accuracy : {bal:.4f}")
        print(f"  Macro F1          : {macro_f1:.4f}   ← ตัวนี้สะท้อนคลาสเล็กจริงกว่า")
        print(f"  Weighted F1       : {weighted_f1:.4f}")
        print(f"  Cohen's Kappa     : {kappa:.4f}")
        print(f"  MCC               : {mcc:.4f}")
        if macro_f1 < acc - 0.15:
            print("  ⚠️  macro F1 ต่ำกว่า accuracy มาก = คลาสเล็กถูกกลบ (class imbalance)")
        print()
        print(classification_report(yt, yp, labels=labels, target_names=labels, zero_division=0))

        cm = confusion_matrix(yt, yp, labels=labels)
        summary["fields"][f] = {
            "accuracy": acc, "balanced_accuracy": bal, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "cohen_kappa": kappa, "mcc": mcc,
            "labels": labels, "confusion_matrix": cm.tolist(),
            "report": classification_report(yt, yp, labels=labels, target_names=labels,
                                            zero_division=0, output_dict=True),
        }

        # ตัวเลขดิบของ confusion matrix — อ่านย้อนหลังได้ ไม่ต้องเดาจาก precision/recall
        cm_csv = C.RESULTS_DIR / f"confusion_{f}_{tag}.csv"
        with cm_csv.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["actual \\ predicted"] + labels)
            for name, row in zip(labels, cm.tolist()):
                w.writerow([name] + row)

        if font:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.rcParams["font.family"] = font
                plt.rcParams["axes.unicode_minus"] = False
                fig, ax = plt.subplots(figsize=(max(6, len(labels) * .7), max(5, len(labels) * .6)))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                for a in range(len(labels)):
                    for b in range(len(labels)):
                        ax.text(b, a, cm[a][b], ha="center", va="center",
                                color="white" if cm[a][b] > cm.max() / 2 else "black")
                ax.set_xlabel("AI Predicted")
                ax.set_ylabel("Actual Truth")
                ax.set_title(f"Confusion Matrix: {f} ({tag})")
                fig.colorbar(im)
                fig.tight_layout()
                fig.savefig(C.RESULTS_DIR / f"confusion_{f}_{tag}.png", dpi=140)
                plt.close(fig)
            except Exception as e:
                print(f"  (วาดกราฟไม่สำเร็จ: {e} — ตัวเลขยังอยู่ในไฟล์ CSV)")

    # Exact match — ทุกฟิลด์ต้องถูกพร้อมกันในข้อความเดียว
    em = sum(r["exact_match"] for r in records) / len(records)
    summary["exact_match"] = em
    print(f"\n=== Exact Match (ทุกฟิลด์ถูกพร้อมกัน) ===")
    print(f"  {em:.4f}  ({sum(r['exact_match'] for r in records)}/{len(records)})")
    print("  ในการใช้งานจริง HR ต้องการให้ทุกฟิลด์ถูกพร้อมกัน ไม่ใช่ถูกทีละฟิลด์")

    if not font:
        print("\n⚠️  ไม่พบฟอนต์ไทยในเครื่อง จึงไม่วาดกราฟ (ตัวเลขครบใน CSV/JSON แล้ว)")
        print("   Ubuntu: sudo apt install fonts-thai-tlwg  |  หรือ pip install fonts-tlwg")

    C.banner("4) บันทึกผล")
    out_json = C.RESULTS_DIR / f"eval_summary_{tag}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = C.RESULTS_DIR / f"eval_predictions_{tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"  {out_json}")
    print(f"  {out_csv}")
    print(f"  {C.RESULTS_DIR}/confusion_*_{tag}.csv")
    print("\nไฟล์ eval_predictions ใช้ดูรายข้อว่าโมเดลผิดตรงไหน — เอาไปเขียนบทวิเคราะห์ได้เลย")


if __name__ == "__main__":
    main()

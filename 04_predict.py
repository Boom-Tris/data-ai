# -*- coding: utf-8 -*-
"""
04_predict.py — ลองยิงข้อความเดี่ยว หรือทั้งไฟล์ (แทน cell 26/27 เดิม)

    python 04_predict.py "แอร์เสียมาสามวันแล้วครับ ร้อนมาก"
    python 04_predict.py --interactive
    python 04_predict.py --file messages.txt --out predictions.jsonl
"""
import argparse
import json
import sys

import hr_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="ข้อความที่จะให้วิเคราะห์")
    ap.add_argument("--interactive", "-i", action="store_true", help="พิมพ์โต้ตอบทีละบรรทัด")
    ap.add_argument("--file", help="ไฟล์ .txt ข้อความละบรรทัด")
    ap.add_argument("--out", help="เขียนผลเป็น JSONL")
    ap.add_argument("--base-only", action="store_true", help="ไม่โหลด LoRA")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    rt = C.detect_runtime(verbose=False)
    C.hf_login_if_possible()
    tokenizer = C.load_tokenizer(None if args.base_only else C.ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(**C.get_load_kwargs(rt))
    if rt["device"] != "cuda":
        model = model.to(rt["device"])
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(C.ADAPTER_DIR))
    model.eval()
    print(f"✅ พร้อมแล้ว ({rt['device']})\n")

    def analyze(text):
        messages = [{"role": "system", "content": C.SYSTEM_PROMPT},
                    {"role": "user", "content": text}]
        enc = C.encode_chat(tokenizer, messages, rt["device"], add_generation_prompt=True)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        return C.parse_json_object(raw), raw

    def show(text):
        parsed, raw = analyze(text)
        print(f"💬 {text}")
        if parsed:
            print("🤖 " + json.dumps(parsed, ensure_ascii=False, indent=2))
        else:
            print(f"⚠️  แปลง JSON ไม่ได้ — ดิบ:\n{raw[:500]}")
        print()
        return {"message": text, "prediction": parsed, "raw": raw}

    results = []
    if args.file:
        lines = [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
        for l in lines:
            results.append(show(l))
    elif args.interactive:
        print("พิมพ์ข้อความแล้วกด Enter — ออกด้วย Ctrl+C หรือพิมพ์ exit\n")
        try:
            while True:
                t = input("> ").strip()
                if t.lower() in ("exit", "quit", ""):
                    break
                results.append(show(t))
        except (KeyboardInterrupt, EOFError):
            print()
    elif args.text:
        results.append(show(" ".join(args.text)))
    else:
        for demo in ["งานหนักมากเลยครับช่วงนี้ ทำงานไม่ทันแล้ว",
                     "หัวหน้าวันนี้แต่งตัวหล่อจังครับ สงสัยมีนัด",
                     "เบิกเบี้ยเลี้ยงรอบที่แล้วยังไม่ได้เลยครับ ถามครั้งที่ 4"]:
            results.append(show(demo))

    if args.out and results:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ เขียนผล {len(results)} แถว → {args.out}")


if __name__ == "__main__":
    main()

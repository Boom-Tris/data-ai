# -*- coding: utf-8 -*-
"""
02_train.py — เตรียม dataset + เทรน LoRA (แทน cell 18/20/22/24 เดิม)

แก้จากโน้ตบุ๊กเดิม 6 จุด:

  1. เลิกใช้ trl.SFTTrainer → ใช้ transformers.Trainer ตรงๆ
     TRL เปลี่ยน API บ่อยมาก (SFTConfig, processing_class ฯลฯ) พอ pip install -U
     บนเครื่องใหม่แล้วโค้ดเดิมพังทันที ในเมื่อเรา tokenize เองอยู่แล้ว SFTTrainer
     ไม่ได้ช่วยอะไรเพิ่ม ตัดออกลด dependency ที่พังง่ายไปหนึ่งตัว

  2. labels เดิมสร้างจาก "คนละสตริง" กับ input_ids
        prompt     = template(messages[1:])   ← user+assistant
        full       = template(messages, add_generation_prompt=True)
        labels     = tokenize(prompt)
     สองอันนี้ token ไม่ตรงตำแหน่งกันเลย (บังเอิญไม่พังเพราะ padding เท่ากันที่ 256
     และ DataCollatorForLanguageModeling เขียนทับ labels ให้ทีหลัง = โค้ดตายที่ทำให้เข้าใจผิด)
     ของใหม่: labels = input_ids แล้ว mask ส่วน prompt เป็น -100
     → โมเดลถูกสอนให้ "ตอบ JSON" ไม่ใช่ "ท่องคำถามซ้ำ"

  3. add_generation_prompt=True บนบทสนทนาที่มี assistant อยู่แล้ว
     ทำให้ท้าย sequence มีหัว assistant ซ้อนอีกอัน = template เพี้ยน

  4. pad_token = eos_token แล้ว mask ทุกตำแหน่งที่ == pad
     → EOS ตัวจริงถูก mask ไปด้วย โมเดลไม่เคยถูกสอนให้หยุด
     ของใหม่แยก pad token ออกมา (ดู hr_common.load_tokenizer)

  5. padding="max_length" ที่ 256 ทุกแถว — เปลืองและตัดข้อความยาวทิ้งเงียบๆ
     ของใหม่ใช้ dynamic padding + รายงานว่าจะโดนตัดกี่แถวก่อนเริ่มเทรน

  6. bf16=True ตายตัว — T4/RTX20xx ไม่รองรับ bf16 ของใหม่ตรวจให้อัตโนมัติ
     และเพิ่ม validation split + save checkpoint (ของเดิม save_strategy="no"
     ทำให้เลือก checkpoint ที่ดีที่สุดไม่ได้เลย)

ตัวอย่าง
    python 02_train.py
    python 02_train.py --epochs 3 --max-seq-len 384 --val-ratio 0.1
    python 02_train.py --legacy-no-mask     # ปิด prompt masking เพื่อเทียบกับผลเดิม
"""
import argparse
import json
import math

import hr_common as C


def load_records(path):
    C.require(path.exists(), f"ไม่พบไฟล์ข้อมูล {path}\n   รัน 01_generate_data.py หรือ 00_import_csv.py ก่อน")
    rows = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                msgs = rec["messages"]
                assert len(msgs) == 3
                rows.append(msgs)
            except Exception:
                bad += 1
    if bad:
        print(f"⚠️  ข้าม {bad} บรรทัดที่อ่านไม่ได้")
    C.require(rows, "ไม่มีข้อมูลที่ใช้ได้เลยในไฟล์")
    return rows


def encode_example(tokenizer, msgs, max_len, mask_prompt=True):
    """
    คืน input_ids / attention_mask / labels ที่ตรงตำแหน่งกันจริง
    labels ของส่วน prompt = -100 (ไม่คิด loss) เหลือเฉพาะคำตอบ JSON ที่ต้องเรียน
    """
    prompt_text = C.render_chat(tokenizer, msgs[:2], add_generation_prompt=True)
    full_text = C.render_chat(tokenizer, msgs, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # เผื่อกรณี template ไม่ทำให้ prompt เป็น prefix ของ full เป๊ะ
    if full_ids[:len(prompt_ids)] != prompt_ids:
        common = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            common += 1
        prompt_len = common
    else:
        prompt_len = len(prompt_ids)

    # ให้แน่ใจว่ามี EOS ปิดท้าย โมเดลจะได้เรียนรู้ว่าจบตรงไหน
    eos_ids = {tokenizer.eos_token_id}
    gen_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(gen_eos, (list, tuple)):
        eos_ids |= set(gen_eos)
    if not full_ids or full_ids[-1] not in eos_ids:
        full_ids = full_ids + [tokenizer.eos_token_id]

    truncated = len(full_ids) > max_len
    full_ids = full_ids[:max_len]
    labels = list(full_ids)
    if mask_prompt:
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }, truncated, prompt_len, len(full_ids)


class PadCollator:
    """dynamic padding — pad แค่เท่าที่ batch นั้นต้องการ ไม่ยัด 256 ทุกแถว"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        n = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = n - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-seq-len", type=int, default=256,
                    help="ความยาวสูงสุดของ sequence (256 เหมาะกับ 6 GB VRAM เช่น RTX 3050)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="สะสม gradient เพื่อจำลอง batch size ใหญ่โดยไม่เปลือง VRAM")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="กันข้อมูลไว้ดู validation loss (0 = ไม่กัน)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy-no-mask", action="store_true",
                    help="ปิด prompt masking ให้เหมือนพฤติกรรมเดิม ใช้ตอนอยากเทียบผล")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)

    C.banner("1) ตรวจเครื่อง")
    rt = C.detect_runtime()
    C.hf_login_if_possible()

    C.banner("2) เตรียมข้อมูล")
    tokenizer = C.load_tokenizer()
    records = load_records(C.DATA_PATH)
    print(f"อ่านมา {len(records)} แถวจาก {C.DATA_PATH.name}")

    examples, lengths, n_trunc = [], [], 0
    for msgs in records:
        ex, trunc, _plen, flen = encode_example(
            tokenizer, msgs, args.max_seq_len, mask_prompt=not args.legacy_no_mask
        )
        # ถ้า prompt ยาวจน labels เหลือแต่ -100 ทั้งแถว แถวนั้นสอนอะไรไม่ได้
        if all(x == -100 for x in ex["labels"]):
            continue
        examples.append(ex)
        lengths.append(flen)
        n_trunc += int(trunc)

    lengths_sorted = sorted(lengths)
    def pct(p):
        return lengths_sorted[min(len(lengths_sorted) - 1, int(len(lengths_sorted) * p))]
    print(f"ความยาว token: min={lengths_sorted[0]}  p50={pct(.5)}  p90={pct(.9)}  max={lengths_sorted[-1]}")
    if n_trunc:
        share = 100 * n_trunc / max(len(examples), 1)
        print(f"⚠️  มี {n_trunc} แถว ({share:.1f}%) ยาวเกิน max_seq_len={args.max_seq_len} และถูกตัด")
        print("    แถวที่ถูกตัดคือแถวที่ JSON คำตอบขาดครึ่ง → สอนให้โมเดลตอบ JSON ไม่ครบ")
        print(f"    แนะนำ: --max-seq-len {int(math.ceil(lengths_sorted[-1] / 64) * 64)}")
    else:
        print("✅ ไม่มีแถวไหนถูกตัด")

    val = []
    if args.val_ratio > 0 and len(examples) > 20:
        import random
        rnd = random.Random(args.seed)
        idx = list(range(len(examples)))
        rnd.shuffle(idx)
        k = max(1, int(len(examples) * args.val_ratio))
        val = [examples[i] for i in idx[:k]]
        examples = [examples[i] for i in idx[k:]]
        print(f"แบ่ง validation {len(val)} แถว / train {len(examples)} แถว")
    print("หมายเหตุ: validation ชุดนี้ใช้ดู loss ระหว่างเทรนเท่านั้น "
          "ไม่ใช่ gold set ที่ใช้วัดผลสุดท้าย")

    C.banner("3) โหลดโมเดล + ติด LoRA")
    model = AutoModelForCausalLM.from_pretrained(**C.get_load_kwargs(rt))
    if rt["device"] != "cuda":
        model = model.to(rt["device"])
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if rt["use_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    C.banner("4) เทรน")
    steps_per_epoch = max(1, len(examples) // (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    kwargs = dict(
        output_dir=str(C.CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.1 * total_steps)),
        logging_steps=max(1, total_steps // 30),
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
        bf16=rt["bf16"],
        fp16=rt["fp16"],
        optim="paged_adamw_8bit" if rt["use_4bit"] else "adamw_torch",
        save_strategy="epoch",
        save_total_limit=2,
    )
    if val:
        kwargs.update(eval_strategy="epoch", per_device_eval_batch_size=args.batch_size)

    try:
        training_args = TrainingArguments(**kwargs)
    except TypeError:
        # transformers รุ่นเก่าใช้ evaluation_strategy แทน eval_strategy
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=examples,
        eval_dataset=val or None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    print(f"ประมาณ {total_steps} steps  ({steps_per_epoch} steps/epoch)")
    trainer.train()
    print("✅ เทรนเสร็จ")

    if val:
        metrics = trainer.evaluate()
        print("validation:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
        (C.RESULTS_DIR / "train_val_metrics.json").write_text(
            json.dumps({"log_history": trainer.state.log_history, "final_eval": metrics},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"บันทึก loss curve → {C.RESULTS_DIR / 'train_val_metrics.json'}")

    C.banner("5) บันทึก adapter")
    C.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(C.ADAPTER_DIR))
    tokenizer.save_pretrained(str(C.ADAPTER_DIR))
    (C.ADAPTER_DIR / "train_config.json").write_text(
        json.dumps(vars(args) | {"model_id": C.MODEL_ID, "n_train": len(examples),
                                 "n_val": len(val), "n_truncated": n_trunc},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ เซฟที่ {C.ADAPTER_DIR}")


if __name__ == "__main__":
    main()

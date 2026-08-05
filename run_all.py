# -*- coding: utf-8 -*-
"""
run_all.py — สคริปต์รันอัตโนมัติไฟล์เดียวจบทุกขั้นตอน (One-Click Automated Pipeline)

รันแค่คำสั่งเดียว:
    python run_all.py

ลำดับการทำงานอัตโนมัติ:
  1. ตรวจสอบเครื่องและสภาพแวดล้อม (check_env.py)
  2. ตรวจสอบ/นำข้อมูลเข้าจาก CSV หรือ JSONL (00_import_csv.py ถ้ามี CSV)
  3. ประเมิน Baseline ด้วยโมเดลฐานก่อนเทรน (03_evaluate.py --base-only --tag base)
  4. เทรน LoRA Model พร้อมเซฟ Checkpoints (02_train.py)
  5. ประเมินผลโมเดลที่ fine-tune แล้ว (03_evaluate.py --tag finetuned)
  6. ทดสอบยิงคำทำนายตัวอย่าง (04_predict.py)

ระบบจะบันทึก Log การทำงานทั้งหมดไว้ที่:
  - workspace/results/run_all_latest.log
  - workspace/results/run_all_<TIMESTAMP>.log
"""

import datetime
import os
import sys
import time
import traceback
from pathlib import Path

import hr_common as C


class TeeLogger:
    """Class ช่วยบันทึก Log ลงไฟล์พร้อมกับแสดงผลออกหน้าจอ Terminal แบบ Real-time"""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def run_step(step_name, module_name, func_name="main", args=None):
    """รันแต่ละสคริปต์พร้อมจับเวลาและการคืนหน่วยความจำ"""
    print("\n" + "═" * 70)
    print(f"📌 [STEP] {step_name}")
    print("═" * 70, flush=True)

    start_time = time.time()
    orig_argv = sys.argv
    if args is not None:
        sys.argv = [module_name + ".py"] + args

    try:
        import importlib
        mod = importlib.import_module(module_name)
        if hasattr(mod, func_name):
            getattr(mod, func_name)()
        elapsed = time.time() - start_time
        print(f"\n✅ [{step_name}] สำเร็จ (ใช้เวลา {elapsed:.1f} วินาที)", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ [{step_name}] เกิดข้อผิดพลาดหลังจากรันไป {elapsed:.1f} วินาที:", flush=True)
        traceback.print_exc()
        return False
    finally:
        sys.argv = orig_argv
        # คืน memory ทันทีหลังจบแต่ละ Step
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = C.RESULTS_DIR / f"run_all_{timestamp}.log"
    latest_log_path = C.RESULTS_DIR / "run_all_latest.log"

    # ตั้งค่าให้ Stream ทั้ง Terminal และ Log File
    tee = TeeLogger(log_file_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 70)
    print("🚀 HR Typhoon — Master Automated Pipeline Run")
    print(f"⏰ เริ่มรันเมื่อ: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 บันทึก Log ไว้ที่: {log_file_path}")
    print("=" * 70)

    total_start = time.time()
    success_steps = []
    failed_steps = []

    # 1. ตรวจสอบเครื่อง (check_env)
    if run_step("1/5 ตรวจสอบสภาพแวดล้อมเครื่อง", "check_env"):
        success_steps.append("check_env")
    else:
        failed_steps.append("check_env")

    # 2. ตรวจสอบข้อมูล/นำเข้าชุดข้อมูลจากโฟลเดอร์ data หรือ CSV
    data_dir = Path("../data").resolve()
    if not data_dir.exists():
        data_dir = Path("/home/boomtris/Downloads/f/data").resolve()

    if data_dir.exists() and list(data_dir.glob("*.jsonl")):
        run_step("นำเข้าและจัดระเบียบชุดข้อมูลจากโฟลเดอร์ data", "00_import_data_dir")
    else:
        csv_candidates = list(Path(".").glob("*.csv")) + list(C.PROJECT_DIR.glob("*.csv"))
        if csv_candidates and (not C.DATA_PATH.exists() or not C.EVAL_DATA_PATH.exists()):
            target_csv = csv_candidates[0]
            print(f"\n💡 พบไฟล์ CSV: {target_csv.name} — ทำการแปลงข้อมูลอัตโนมัติ")
            run_step("นำเข้าชุดข้อมูล Train จาก CSV", "00_import_csv",
                     args=[str(target_csv), "--split", "train", "--out", "train", "--overwrite"])
            run_step("นำเข้าชุดข้อมูล Eval จาก CSV", "00_import_csv",
                     args=[str(target_csv), "--split", "gold", "--out", "eval", "--overwrite"])
        elif C.DATA_PATH.exists() and C.EVAL_DATA_PATH.exists():
            print(f"\n✅ พบไฟล์ข้อมูลพร้อมใช้งานแล้ว: {C.DATA_PATH.name} และ {C.EVAL_DATA_PATH.name}")

    # 3. ประเมิน Baseline ด้วยโมเดลฐานก่อนเทรน
    if run_step("2/5 ประเมิน Baseline (Base Model)", "03_evaluate", args=["--base-only", "--tag", "base"]):
        success_steps.append("evaluate_base")
    else:
        failed_steps.append("evaluate_base")

    # 4. เทรน LoRA Model
    if run_step("3/5 เทรน LoRA Model", "02_train"):
        success_steps.append("train_lora")
    else:
        failed_steps.append("train_lora")

    # 5. ประเมินผล Fine-tuned Model
    if run_step("4/5 ประเมินผลโมเดลที่ Fine-tuned แล้ว", "03_evaluate", args=["--tag", "finetuned"]):
        success_steps.append("evaluate_finetuned")
    else:
        failed_steps.append("evaluate_finetuned")

    # 6. ทดสอบยิงคำทำนายตัวอย่าง
    if run_step("5/5 ทดสอบยิงทำนายตัวอย่าง", "04_predict"):
        success_steps.append("predict_demo")
    else:
        failed_steps.append("predict_demo")

    total_elapsed = time.time() - total_start

    # สรุปผลการรัน
    print("\n" + "═" * 70)
    print("📊 สรุปผลการรัน Master Pipeline ทั้งหมด")
    print("═" * 70)
    print(f"⏱️  เวลารวมทั้งหมด : {total_elapsed / 60:.2f} นาที")
    print(f"✅ ขั้นตอนที่สำเร็จ  : {len(success_steps)}/{len(success_steps) + len(failed_steps)} ({', '.join(success_steps)})")
    if failed_steps:
        print(f"❌ ขั้นตอนที่ไม่สำเร็จ : {len(failed_steps)} ({', '.join(failed_steps)})")
    print(f"📁 ผลลัพธ์ทั้งหมดเก็บที่ : {C.RESULTS_DIR}")
    print(f"📝 ไฟล์ Log ทั้งหมด   : {log_file_path}")
    print("═" * 70 + "\n")

    # คัดลอกไปไว้ที่ run_all_latest.log ด้วย
    try:
        latest_log_path.write_text(log_file_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()

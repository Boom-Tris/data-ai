# -*- coding: utf-8 -*-
"""
check_env.py — รันตัวนี้ก่อนเสมอ

ตรวจว่าเครื่องพร้อมไหม ก่อนจะเสียเวลาโหลดโมเดล 16 GB แล้วค่อยมาพังตอนท้าย
    python check_env.py
"""
import importlib
import platform
import shutil
import sys
from pathlib import Path

import hr_common as C

C.banner("1) Python และระบบปฏิบัติการ")
print(f"  python   : {sys.version.split()[0]}  ({platform.python_implementation()})")
print(f"  platform : {platform.system()} {platform.release()} / {platform.machine()}")
if sys.version_info < (3, 9):
    print("  ❌ ต้องใช้ Python 3.9 ขึ้นไป")
elif sys.version_info >= (3, 13):
    print("  ⚠️  Python 3.13+ อาจมีปัญหากับ wheel ของ torch/bitsandbytes — 3.10-3.12 ปลอดภัยกว่า")
else:
    print("  ✅ เวอร์ชัน Python ใช้ได้")

C.banner("2) ไลบรารีที่ต้องมี")
REQUIRED = ["torch", "transformers", "peft", "accelerate", "datasets", "sklearn", "numpy"]
OPTIONAL = ["bitsandbytes", "matplotlib", "seaborn", "pythainlp", "dotenv", "pandas"]

missing = []
for name in REQUIRED:
    try:
        m = importlib.import_module(name)
        print(f"  ✅ {name:<16} {getattr(m, '__version__', '?')}")
    except Exception as e:
        missing.append(name)
        print(f"  ❌ {name:<16} ไม่พบ ({type(e).__name__})")

for name in OPTIONAL:
    try:
        m = importlib.import_module(name)
        print(f"  ✅ {name:<16} {getattr(m, '__version__', '?')}  (ไม่บังคับ)")
    except Exception:
        print(f"  ⚠️  {name:<16} ไม่พบ (ไม่บังคับ แต่บางฟีเจอร์จะถูกข้าม)")

if missing:
    print(f"\n  ติดตั้งเพิ่ม: pip install {' '.join(missing)}")
    print("  (torch ต้องลงแยกตาม CUDA ของเครื่อง — ดู README)")
    sys.exit(1)

C.banner("3) ฮาร์ดแวร์")
rt = C.detect_runtime(verbose=True)

C.banner("4) พื้นที่ดิสก์และโฟลเดอร์งาน")
print(f"  HR_PROJECT_DIR = {C.PROJECT_DIR}")
usage = shutil.disk_usage(C.PROJECT_DIR)
free_gb = usage.free / 1024 ** 3
print(f"  พื้นที่ว่าง     : {free_gb:.1f} GB")
if free_gb < 25:
    print("  ⚠️  โมเดล 8B กินพื้นที่ ~16 GB + adapter + cache — ควรมีว่างอย่างน้อย 25 GB")
else:
    print("  ✅ พื้นที่พอ")

hf_home = Path.home() / ".cache" / "huggingface"
print(f"  cache โมเดล    : {hf_home}")
print("  (ย้ายที่เก็บได้ด้วย  export HF_HOME=/path/ที่ต้องการ)")

C.banner("5) Hugging Face token")
if C.HF_TOKEN:
    print(f"  ✅ พบ HF_TOKEN (ขึ้นต้นด้วย {C.HF_TOKEN[:4]}…)")
else:
    print("  ⚠️  ไม่พบ HF_TOKEN — ก็อป .env.example เป็น .env แล้วใส่ token")

C.banner("6) ไฟล์ข้อมูล")
for label, p in [("train", C.DATA_PATH), ("eval", C.EVAL_DATA_PATH)]:
    if p.exists():
        n = sum(1 for _ in p.open(encoding="utf-8"))
        print(f"  ✅ {label:<6} {p}  ({n} แถว)")
    else:
        print(f"  ⚠️  {label:<6} ยังไม่มี → {p}")

C.banner("สรุป")
if rt["device"] == "cuda" and rt["use_4bit"]:
    print("  พร้อมเทรนเต็มรูปแบบ → รัน 01_generate_data.py ต่อได้เลย")
elif rt["device"] == "cuda":
    print("  มี GPU แต่ไม่มี bitsandbytes → จะโหลดโมเดลแบบไม่ quantize ต้อง VRAM สูง")
    print("  ลอง: pip install bitsandbytes")
else:
    print("  ไม่มี GPU → รันได้แต่ช้ามาก แนะนำตั้ง HR_MODEL_ID เป็นโมเดลเล็กเพื่อทดสอบโค้ดก่อน")
    print("  เช่น: export HR_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct")

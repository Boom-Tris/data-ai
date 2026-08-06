# -*- coding: utf-8 -*-
"""
hr_common.py — ของกลางที่ทุกสคริปต์เรียกใช้
รวม: config, ตรวจเครื่อง, taxonomy, normalize, parse JSON, encode chat แบบทนหลายเวอร์ชัน

ไม่มีอะไรที่ผูกกับ Google Colab ในไฟล์นี้
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- config
# โหลด .env ถ้ามี (ไม่บังคับ)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(os.getenv("HR_PROJECT_DIR", "./workspace")).expanduser().resolve()
MODEL_ID = os.getenv("HR_MODEL_ID", "scb10x/typhoon-v1.5-8b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or None

DATA_PATH = PROJECT_DIR / "train_data_intent.jsonl"
EVAL_DATA_PATH = PROJECT_DIR / "eval_data_intent.jsonl"
ADAPTER_DIR = PROJECT_DIR / "hr_intent_lora_model"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

for _d in (PROJECT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ลด fragmentation ของ CUDA allocator — ต้องตั้งก่อน import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# กัน tokenizers เตือนเรื่อง fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_PROMPT = (
    "คุณคือระบบ AI ผู้ช่วย HR รับฟังและวิเคราะห์เจตนาจากข้อความพนักงาน "
    "ประเมินความรุนแรง แยกแยะการบ่นออกจากการพูดเล่น และตอบเป็น JSON"
)

# ---------------------------------------------------------------- taxonomy
CATEGORY_LABELS = [
    "แอร์เสีย",
    "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "หัวหน้า",
    "เพื่อนร่วมงาน",
    "งานเยอะ/ภาระงาน",
    "ชั่วโมงทำงาน/OT",
    "เงินเดือน/ค่าตอบแทน",
    "สวัสดิการ",
    "ห้องทำงานร้อน",
    "ความสะอาด/สภาพแวดล้อมออฟฟิศ",
    "ที่จอดรถ/การเดินทาง",
    "ความปลอดภัยในที่ทำงาน",
    "อาหาร/โรงอาหาร",
    "ระบบงาน/ขั้นตอนการทำงาน",
    "การสื่อสารภายในองค์กร",
    "ลูกค้า",
    "อื่นๆ",
]

INTENT_LABELS = ["บ่น", "ประชด", "พูดเล่น", "ชมเชย", "สอบถาม"]
SEVERITY_LABELS = ["none", "low", "medium", "high"]

INTENT_GUIDE = {
    "บ่น": "พนักงานระบายความไม่พอใจ/ความเหนื่อยล้าตรงๆ อย่างจริงจัง โดยไม่มีน้ำเสียงแดกดันหรือคำถาม เช่น 'งานเยอะจนแทบไม่ได้พักเลยวันนี้'",
    "ประชด": "พนักงานพูดในเชิงแดกดัน โดยใช้คำชมหรือคำบวกแต่ความหมายจริงตรงข้าม มักมีคำเช่น 'ดีจัง' 'เก่งมาก' 'สุดยอด' ทั้งที่สถานการณ์แย่",
    "พูดเล่น": "โทนเบาสบาย ไม่ได้ตั้งใจร้องเรียนจริงจัง เป็นมุกตลกในบริบทงาน ไม่มีเจตนาให้ HR ต้องรีบแก้ไข",
    "ชมเชย": "ชื่นชม/ขอบคุณอย่างจริงใจ ไม่มีความหมายแฝงเชิงลบ",
    "สอบถาม": "ต้องการทราบข้อมูล/ขั้นตอน/สถานะ มีลักษณะเป็นคำถามปลายเปิดเพื่อขอความชัดเจน ไม่ได้ระบายความไม่พอใจ",
}

INTENT_MERGE_MAP = {
    "ประชด": "ประชด", "ประชดประชัน": "ประชด", "ประชดหัวหน้า": "ประชด", "ประชดประชันหัวหน้า": "ประชด",
    "แดกดัน": "ประชด",
    "บ่น": "บ่น", "บ่นเรื่องงาน": "บ่น", "บ่นเรื่องแอร์เสีย": "บ่น", "บ่นเรื่องหัวหน้า": "บ่น",
    "แจ้งปัญหา": "บ่น", "ร้องเรียน": "บ่น", "ระบาย": "บ่น",
    "พูดเล่น": "พูดเล่น", "แซว": "พูดเล่น", "แซวหัวหน้า": "พูดเล่น", "ล้อเล่น": "พูดเล่น", "หยอกล้อ": "พูดเล่น",
    "ชมเชย": "ชมเชย", "ชื่นชม": "ชมเชย", "ขอบคุณ": "ชมเชย",
    "สอบถาม": "สอบถาม", "ถามข้อมูล": "สอบถาม", "คำถาม": "สอบถาม",
}

CATEGORY_MERGE_MAP = {
    "แอร์เสีย": "แอร์เสีย", "แอรเสีย": "แอร์เสีย", "แอร์ไม่เย็น": "แอร์เสีย", "แอร์ไม่ทำงาน": "แอร์เสีย", "แอร์": "แอร์เสีย",
    "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "คอมพิวเตอร์เสีย": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง", "คอมพิวเตอร์เสมอ": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "คอมเสีย": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง", "คอมพิวเตอร์": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "ไอทีขัดข้อง": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง", "อินเทอร์เน็ตล่ม": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "เครื่องปรินต์": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง", "ปริ้นเตอร์": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "อุปกรณ์": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง", "ระบบไอที": "อุปกรณ์ไอที/คอมพิวเตอร์ขัดข้อง",
    "หัวหน้า": "หัวหน้า", "เจ้านาย": "หัวหน้า", "ผู้บังคับบัญชา": "หัวหน้า",
    "เพื่อนร่วมงาน": "เพื่อนร่วมงาน", "ทีม": "เพื่อนร่วมงาน",
    "งาน": "งานเยอะ/ภาระงาน", "งานเยอะ": "งานเยอะ/ภาระงาน", "เรื่องงาน": "งานเยอะ/ภาระงาน",
    "ภาระงาน": "งานเยอะ/ภาระงาน", "งานเยอะ/ภาระงาน": "งานเยอะ/ภาระงาน",
    "ชั่วโมงทำงาน": "ชั่วโมงทำงาน/OT", "OT": "ชั่วโมงทำงาน/OT", "โอที": "ชั่วโมงทำงาน/OT",
    "ล่วงเวลา": "ชั่วโมงทำงาน/OT", "ชั่วโมงทำงาน/OT": "ชั่วโมงทำงาน/OT",
    "เงินเดือน": "เงินเดือน/ค่าตอบแทน", "ค่าตอบแทน": "เงินเดือน/ค่าตอบแทน", "โบนัส": "เงินเดือน/ค่าตอบแทน",
    "เบี้ยเลี้ยง": "เงินเดือน/ค่าตอบแทน", "เงินเดือน/ค่าตอบแทน": "เงินเดือน/ค่าตอบแทน",
    "สวัสดิการ": "สวัสดิการ", "ประกันสุขภาพ": "สวัสดิการ", "วันลา": "สวัสดิการ",
    "ห้องทำงาน": "ห้องทำงานร้อน", "ห้องทำงานร้อน": "ห้องทำงานร้อน", "อากาศร้อน": "ห้องทำงานร้อน",
    "ความสะอาด": "ความสะอาด/สภาพแวดล้อมออฟฟิศ", "สภาพแวดล้อมออฟฟิศ": "ความสะอาด/สภาพแวดล้อมออฟฟิศ",
    "สภาพแวดล้อม": "ความสะอาด/สภาพแวดล้อมออฟฟิศ", "ห้องน้ำ": "ความสะอาด/สภาพแวดล้อมออฟฟิศ",
    "ความสะอาด/สภาพแวดล้อมออฟฟิศ": "ความสะอาด/สภาพแวดล้อมออฟฟิศ",
    "ที่จอดรถ": "ที่จอดรถ/การเดินทาง", "การเดินทาง": "ที่จอดรถ/การเดินทาง", "ที่จอดรถ/การเดินทาง": "ที่จอดรถ/การเดินทาง",
    "ความปลอดภัย": "ความปลอดภัยในที่ทำงาน", "ความปลอดภัยในที่ทำงาน": "ความปลอดภัยในที่ทำงาน",
    "อาหาร": "อาหาร/โรงอาหาร", "โรงอาหาร": "อาหาร/โรงอาหาร", "อาหาร/โรงอาหาร": "อาหาร/โรงอาหาร",
    "ระบบงาน": "ระบบงาน/ขั้นตอนการทำงาน", "ขั้นตอนการทำงาน": "ระบบงาน/ขั้นตอนการทำงาน",
    "กระบวนการ": "ระบบงาน/ขั้นตอนการทำงาน", "ระบบงาน/ขั้นตอนการทำงาน": "ระบบงาน/ขั้นตอนการทำงาน",
    "การสื่อสาร": "การสื่อสารภายในองค์กร", "การสื่อสารภายในองค์กร": "การสื่อสารภายในองค์กร",
    "ลูกค้า": "ลูกค้า",
    "พักผ่อน": "อื่นๆ", "อื่นๆ": "อื่นๆ", "อื่น ๆ": "อื่นๆ",
}

# ค่าที่ใช้แทน "โมเดลตอบไม่ได้/JSON พัง" — นับเป็นคำตอบผิด ไม่ใช่โยนแถวทิ้ง
PARSE_FAIL = "__PARSE_FAIL__"


def _norm(label, table):
    if label is None:
        return None
    label = str(label).strip()
    if not label:
        return None
    return table.get(label, label)


def normalize_intent(label):
    return _norm(label, INTENT_MERGE_MAP)


def normalize_category(label):
    return _norm(label, CATEGORY_MERGE_MAP)


def normalize_severity(label):
    if label is None:
        return None
    s = str(label).strip().lower()
    alias = {"ไม่มี": "none", "ต่ำ": "low", "ปานกลาง": "medium", "กลาง": "medium", "สูง": "high",
             "n/a": "none", "": None}
    s = alias.get(s, s)
    return s if s in SEVERITY_LABELS else s


def normalize_is_joke(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "ใช่"):
        return True
    if s in ("false", "0", "no", "ไม่ใช่"):
        return False
    return None


# ---------------------------------------------------------------- JSON parsing
def parse_json_array(text):
    """ดึง JSON array ตัวแรกที่ balance วงเล็บถูกออกมา ทนกับข้อความห้อยท้าย"""
    start = text.find("[")
    if start == -1:
        return []
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return []
    return []


def parse_json_object(text):
    """เหมือนข้างบนแต่เป็น object เดี่ยว — ใช้ตอนอ่านคำตอบของโมเดลตอนประเมิน"""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def normalize_thai(text):
    """เรียก pythainlp ถ้ามี ถ้าไม่มีก็แค่บีบช่องว่าง — ไม่ให้ทั้งสคริปต์ล้มเพราะ optional dep"""
    try:
        from pythainlp.util import normalize as _pt_normalize
        text = _pt_normalize(text)
    except Exception:
        pass
    return re.sub(r"\s{2,}", " ", str(text)).strip()


# ---------------------------------------------------------------- hardware
def detect_runtime(verbose=True):
    """
    ตรวจเครื่องแล้วคืน dict บอกว่าจะใช้ device อะไร dtype อะไร quantize ได้ไหม
    แทนที่จะ hardcode cuda + bfloat16 แบบในโน้ตบุ๊กเดิม
    """
    import torch

    info = {"device": "cpu", "dtype": torch.float32, "use_4bit": False,
            "bf16": False, "fp16": False, "vram_gb": None, "gpu_name": None}

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        # bf16 ต้อง Ampere ขึ้นไป (RTX 30xx/40xx, A100...) — T4/RTX 20xx ใช้ไม่ได้
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        info["bf16"] = bool(bf16_ok)
        info["fp16"] = not bf16_ok
        info["dtype"] = torch.bfloat16 if bf16_ok else torch.float16
        try:
            import bitsandbytes  # noqa: F401
            info["use_4bit"] = True
        except Exception:
            info["use_4bit"] = False
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Mac Apple Silicon — bitsandbytes ใช้ไม่ได้ ต้องโหลดเต็ม precision
        info["device"] = "mps"
        info["dtype"] = torch.float16

    if verbose:
        print("=" * 60)
        print(f"  device      : {info['device']}")
        if info["gpu_name"]:
            print(f"  GPU         : {info['gpu_name']} ({info['vram_gb']} GB)")
        print(f"  dtype       : {info['dtype']}")
        print(f"  4-bit quant : {info['use_4bit']}")
        print(f"  bf16 / fp16 : {info['bf16']} / {info['fp16']}")
        print("=" * 60)
        if info["device"] == "cuda" and info["vram_gb"] and info["vram_gb"] <= 7.0:
            print(f"💡 ตรวจพบ GPU VRAM {info['vram_gb']} GB (เช่น RTX 3050 6GB) — ปรับแต่งโหมดประหยัด VRAM อัตโนมัติ")
            print("   (แนะนำใช้: 4-bit quant + max_seq_len=256 + grad_accum=8 + paged_adamw_8bit)")
        elif info["device"] == "cuda" and info["vram_gb"] and info["vram_gb"] < 10 and info["use_4bit"]:
            print("⚠️  VRAM ต่ำกว่า 10 GB — เทรน 8B แบบ 4-bit อาจ OOM หากไม่ตั้ง max_seq_len สั้นลง")
        if not info["use_4bit"] and info["device"] == "cuda":
            print("⚠️  ไม่พบ bitsandbytes → จะโหลดโมเดลแบบไม่ quantize (ต้องการ VRAM ~16 GB+)")
        if info["device"] == "cpu":
            print("⚠️  ไม่พบ GPU — โมเดล 8B บน CPU จะช้ามาก (หลายชั่วโมงต่อการเทรน 1 รอบ)")
            print("    แนะนำให้ตั้ง HR_MODEL_ID เป็นโมเดลเล็กกว่านี้เพื่อทดสอบว่าโค้ดเดินได้ก่อน")
    return info


def configure_safety_limits():
    """จำกัด CPU threads และตั้งค่าความปลอดภัย ป้องกันเครื่องร้อนจนดับ"""
    import torch
    if hasattr(torch, "set_num_threads"):
        try:
            # จำกัด CPU Threads ไม่เกิน 4 เพื่อไม่ให้ CPU (เช่น i7-14700F) วิ่ง 100% ร้อนจนเครื่องดับ
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass


def get_max_memory(rt):
    """
    สร้าง max_memory map เมื่อ VRAM มีจำกัด และไม่ได้ใช้ 4-bit quant
    หมายเหตุ: bitsandbytes 4-bit quant ไม่รองรับการ offload ไป CPU
    การจำกัด max_memory จะทำให้ transformers พยายามส่ง layer ไป CPU และเกิด ValueError
    """
    if not rt["use_4bit"] and rt["device"] == "cuda" and rt["vram_gb"] and rt["vram_gb"] <= 7.0:
        return {0: f"{int(rt['vram_gb'] - 1.0)}GB", "cpu": "12GB"}
    return None


def build_quant_config(rt):
    """สร้าง BitsAndBytesConfig เฉพาะเมื่อทำได้จริง ไม่งั้นคืน None"""
    if not rt["use_4bit"]:
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=rt["dtype"],
    )


def get_load_kwargs(rt):
    """คืน dict สำหรับ AutoModelForCausalLM.from_pretrained พร้อมระบบป้องกัน VRAM Overflown & CPU Offloading"""
    configure_safety_limits()
    quant = build_quant_config(rt)
    max_mem = get_max_memory(rt)
    offload_dir = PROJECT_DIR / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    # บน Windows/GPU 6GB การใช้ device_map="auto" จะทำให้ accelerate คำนวณ VRAM ที่เหลือ
    # (ซึ่งโดน Windows OS/Display ดึงไปบางส่วน) แล้วพยายามแอบส่ง 1-2 layers ไป CPU
    # ทำให้เกิด ValueError จาก bitsandbytes 4-bit validator
    # การบังคับ device_map={"": 0} จะเจาะจงให้โมเดลทั้งตัวลง GPU 0 โดยไม่แบ่งไป CPU
    if rt["device"] == "cuda":
        dev_map = {"": 0} if quant is not None else "auto"
    else:
        dev_map = None

    kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "quantization_config": quant,
        "dtype": rt["dtype"],
        "device_map": dev_map,
        "low_cpu_mem_usage": True,
        "token": HF_TOKEN,
    }
    if max_mem and quant is None:
        kwargs["max_memory"] = max_mem
        kwargs["offload_folder"] = str(offload_dir)
        kwargs["offload_state_dict"] = True
    return kwargs





def hf_login_if_possible():
    if not HF_TOKEN:
        print("ℹ️  ไม่พบ HF_TOKEN (ตั้งใน .env หรือ export ก็ได้)")
        print("    ถ้าโมเดลเป็น public ก็ไม่จำเป็น แต่ถ้า gated จะโหลดไม่ผ่าน")
        return
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("✅ login Hugging Face สำเร็จ")
    except Exception as e:
        print(f"⚠️  login ไม่สำเร็จ: {e}")


# ---------------------------------------------------------------- chat encoding
def encode_chat(tokenizer, messages, device, add_generation_prompt=True):
    """
    apply_chat_template คืนค่าไม่เหมือนกันในแต่ละเวอร์ชันของ transformers
    (บางเวอร์ชันคืน tensor ล้วน บางเวอร์ชันคืน dict) — โน้ตบุ๊กเดิมสมมติแบบเดียว
    ทำให้ย้ายเครื่องแล้วพังตรงนี้บ่อยที่สุด ฟังก์ชันนี้รองรับทั้งสองแบบ
    """
    import torch
    enc = None
    try:
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=add_generation_prompt, return_dict=True,
        )
    except TypeError:
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=add_generation_prompt,
        )
    if not hasattr(enc, "keys"):
        ids = enc
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    out = {}
    for k in ("input_ids", "attention_mask"):
        if k in enc:
            out[k] = enc[k].to(device)
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(out["input_ids"])
    return out


def render_chat(tokenizer, messages, add_generation_prompt=False):
    """คืน string ของ chat template (ไม่ tokenize) ใช้ตอนเตรียม dataset"""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def load_tokenizer(path=None):
    """
    โหลด tokenizer แล้ว **แยก pad ออกจาก eos**

    บั๊กเดิม: โน้ตบุ๊กตั้ง tokenizer.pad_token = tokenizer.eos_token
    ตอนเทรน collator จะ mask ทุกตำแหน่งที่ == pad_token_id ให้เป็น -100
    ซึ่งเผลอ mask EOS ตัวจริงท้ายประโยคไปด้วย → โมเดล "ไม่เคยถูกสอนให้หยุด"
    ผลคือตอน generate มันจะพ่นต่อไปเรื่อยๆ หลังจบ JSON แล้ว
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path or MODEL_ID), token=HF_TOKEN)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        added = tok.add_special_tokens({"pad_token": "<|hr_pad|>"})
        if added == 0 and tok.pad_token is None:
            tok.pad_token = tok.eos_token
            print("⚠️  เพิ่ม pad token แยกไม่สำเร็จ ใช้ eos เป็น pad (EOS จะถูก mask)")
        else:
            print(f"✅ เพิ่ม pad token แยกจาก eos แล้ว (pad_id={tok.pad_token_id}, eos_id={tok.eos_token_id})")
    return tok


def banner(title):
    print()
    print("━" * 64)
    print(f"  {title}")
    print("━" * 64)


def require(condition, message):
    if not condition:
        print(f"\n❌ {message}")
        sys.exit(1)

# HR Typhoon — ฉบับรันบนเครื่อง Local (RTX 3050 6GB / Windows & Linux)

ระบบวิเคราะห์เจตนาพนักงาน (HR Intent Analysis) แปลงจาก Jupyter Notebook บน Colab ให้เป็น Python Pipeline สำหรับรันบนเครื่องตัวเองได้อย่างมีประสิทธิภาพ พร้อมระบบป้องกัน VRAM ล้นและถอดบั๊กเดิมออกทั้งหมด

---

## ⚡ 1. การใช้งานด่วน (Quick Start)

### 🌟 วิธีที่ 1: รันคำสั่งเดียวจบทุกขั้นตอน (แนะนำ)
```bash
python run_all.py
```
*(คำสั่งนี้จะรันตรวจสอบเครื่อง -> นำเข้าข้อมูล -> วัด Baseline -> เทรน LoRA -> ประเมินผล Fine-tuned -> ทดสอบทำนาย พร้อมบันทึก Log ลง `workspace/results/run_all_latest.log` อัตโนมัติ)*

---

### 🛠️ วิธีที่ 2: รันแยกทีละขั้นตอน

```bash
# 1. ตรวจสอบสภาพแวดล้อมเครื่อง
python check_env.py

# 2. นำเข้าข้อมูลจาก CSV (ถ้ามีไฟล์ 01_all_labeled.csv)
python 00_import_csv.py 01_all_labeled.csv --split train --out train --overwrite
python 00_import_csv.py 01_all_labeled.csv --split gold  --out eval  --overwrite

# 3. เทรน LoRA Model (ตั้งค่าเซฟ VRAM เหมาะกับ RTX 3050 6GB)
python 02_train.py --epochs 3 --max-seq-len 256 --grad-accum 8

# 4. ประเมินผล (เทียบโมเดลฐานก่อนเทรน VS โมเดล Fine-tuned)
python 03_evaluate.py --base-only --tag base
python 03_evaluate.py --tag finetuned

# 5. ทดสอบยิงข้อความทำนาย
python 04_predict.py --interactive
```

ผลลัพธ์ กราฟ Confusion Matrix และตัวเลขดิบทั้งหมดจะถูกบันทึกไว้ใน `workspace/results/`

---

## 🛡️ 2. การปรับแต่งสำหรับ GPU 6 GB VRAM (RTX 3050 6GB)

ระบบถูกออกแบบด้วยมาตรการป้องกันความปลอดภัยของฮาร์ดแวร์เพื่อไม่ให้เกิด CUDA OOM หรือเครื่องดับ:

1. **4-Bit NF4 Quantization (bitsandbytes)**: บีบอัดโมเดล 8B เหลือ ~4.5 GB VRAM
2. **Hard-Capped VRAM Ceiling (4.0 GB)**: ล็อกการจอง GPU VRAM ไว้ไม่เกิน 4.0 GB เสมอ เหลือ 2.0 GB ให้ระบบปฏิบัติการ Windows ป้องกันหน้าจอดับ
3. **Automatic CPU Offloading**: ส่วนเกินของโมเดลจะถูกสตรีมลง System RAM (16GB) ผ่าน `offload_folder`
4. **CPU Thermal Guard**: จำกัด PyTorch ไว้ไม่เกิน 4 Threads ป้องกัน CPU (เช่น i7-14700F) ร้อนเกิน 100%
5. **Paged 8-Bit AdamW**: ย้าย Memory ของ Optimizer ไปไว้ที่ System RAM อัตโนมัติเมื่อเกิดสไปค์

---

## 🐞 3. สรุปบั๊ก 13 จุดเดิมที่ได้รับการแก้ไขแล้ว

| # | รายการบั๊กเดิม | สถานะและการแก้ไขในชุดนี้ |
|---|---|---|
| 🔴 1 | **Accuracy พองตัว**: แถวที่โมเดลตอบ JSON พังถูกโยนทิ้ง ไม่ถูกนับว่าผิด | **แก้ไขแล้ว**: นับแถวพังเป็น `__PARSE_FAIL__` ซึ่งเป็นความผิดเสมอ |
| 🔴 2 | **Circular Eval**: ข้อมูล Eval ถูกเจนด้วยโมเดลที่ Fine-tune แล้ว | **แก้ไขแล้ว**: แยกชุดประเมิน Gold Set เป็นไฟล์อิสระ ประเมินเทียบ Base Model แท้ |
| 🔴 3 | **Labels ไม่ตรงกับ Input**: Labels สร้างจากคนละสตริง ไม่มี Prompt Masking | **แก้ไขแล้ว**: ใช้ Masking `-100` ในส่วน Prompt ให้ Loss ตกลงเฉพาะคำตอบ JSON |
| 🔴 4 | **`pad_token = eos_token`**: ทำให้ EOS ท้ายประโยคถูก Mask โมเดลไม่รู้จักจุดหยุด | **แก้ไขแล้ว**: เพิ่ม Special Pad Token แยก (`<|hr_pad|>`) และ Resize Embedding |
| 🟠 5 | **ตัดที่ 256 Token โดยไม่บอก**: ตัดคำตอบ JSON ขาดครึ่ง | **แก้ไขแล้ว**: คำนวณความยาว Token พร้อมพิมพ์ p50/p90/max และเตือนหากมีการ Truncate |
| 🟠 6 | **`bf16=True` ตายตัว**: เครื่องที่ไม่รองรับ bf16 รันไม่ผ่าน | **แก้ไขแล้ว**: ตรวจสอบ `torch.cuda.is_bf16_supported()` และสลับไป fp16 อัตโนมัติ |
| 🟠 7 | **`apply_chat_template` พัง**: Transformers ต่างเวอร์ชันคืนค่าไม่เหมือนกัน | **แก้ไขแล้ว**: สร้าง `C.encode_chat()` รองรับทั้ง Dict และ Tensor |
| 🟠 8 | **`trl.SFTTrainer` API เปลี่ยนบ่อย**: อัปเดตแล้วโค้ดล้มทันที | **แก้ไขแล้ว**: ใช้ `transformers.Trainer` ดั้งเดิม เสถียรสูง |
| 🟠 9 | **ไม่มี Checkpoint / Eval**: `save_strategy="no"` เลือก Checkpoint ไม่ได้ | **แก้ไขแล้ว**: Save ทุก Epoch มี Validation Split และบันทึก Loss Curve |
| 🟠 10 | **Warning `temperature`**: ส่ง temperature ทั้งที่ `do_sample=False` | **แก้ไขแล้ว**: ตัด Parameter ที่ซ้ำซ้อนออก |
| 🟠 11 | **วัดเฉพาะ Category/Intent**: ไม่เคยวัด Severity, Is_Joke, Exact Match | **แก้ไขแล้ว**: วัดครบทั้ง 4 ฟิลด์ + Exact Match (ทุกฟิลด์ต้องถูกพร้อมกัน) |
| 🟠 12 | **เมตริกน้อยเกินไป**: มีแค่ Accuracy | **แก้ไขแล้ว**: เพิ่ม Balanced Accuracy, Macro F1, Kappa, MCC และ Dump CSV ดิบ |
| ⚪ 13 | **พึ่งพา Colab**: `drive.mount`, `!wget` ล้มเมื่อรันบนเครื่อง Local | **แก้ไขแล้ว**: ถอดคำสั่ง Colab ออกทั้งหมด หาฟอนต์ไทยใน OS อัตโนมัติ |

---

## 📁 4. โครงสร้างไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่การทำงาน |
|---|---|
| [run_all.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/run_all.py) | **(หลัก)** สคริปต์รันอัตโนมัติ 1-Click ครบทุกขั้นตอน พร้อมบันทึก Full Log |
| [hr_common.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/hr_common.py) | Config, Taxonomy, Normalization, ฮาร์ดแวร์ Guard, CPU/GPU Offloading |
| [check_env.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/check_env.py) | ตรวจสอบความพร้อมของฮาร์ดแวร์ GPU/RAM และแพ็กเกจในเครื่อง |
| [00_import_csv.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/00_import_csv.py) | แปลงไฟล์ CSV (เช่น `01_all_labeled.csv`) เข้าสู่รูปแบบ JSONL |
| [02_train.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/02_train.py) | โหลดข้อมูล, Mask Prompt, เทรน LoRA Adapter และบันทึก Checkpoint |
| [03_evaluate.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/03_evaluate.py) | ประเมินผล 4 ฟิลด์ + Exact Match + เมตริกครบชุด + เซฟ Confusion Matrix |
| [04_predict.py](file:///home/boomtris/Downloads/f/hr_typhoon_local/04_predict.py) | ทดสอบป้อนข้อความเดี่ยว/ไฟล์ เพื่อทำนายเจตนาแบบ Interactive |

---

## 🤖 5. การเจนข้อมูลสังเคราะห์ (Data Generation)

การสร้างข้อมูลสังเคราะห์เพิ่มเติมได้ถูกแยกออกไปรันบน **Google Colab** เพื่อไม่ให้ดึงทรัพยากรเครื่อง:
- สคริปต์ notebook: `colab_generate_data.ipynb` (อยู่ในโฟลเดอร์ด้านนอก)
- ใช้ Gemini API หรือ OpenAI API ในการเจนข้อมูล 17 หมวดอย่างรวดเร็ว
- เมื่อเจนเสร็จ นำไฟล์ `train_data_generated.jsonl` มาวางแทนที่ `workspace/train_data_intent.jsonl` แล้วรัน `python run_all.py` ได้ทันที

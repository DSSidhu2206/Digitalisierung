#!/usr/bin/env python3
"""
Build the Kaggle Donut fine-tuning notebook (`train_donut.ipynb`).

Kept as a builder so the notebook JSON is always valid and regenerable:
    python training/build_notebook.py
"""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Fine-tune Donut on synthetic German documents

Trains an **OCR-free** image→JSON extraction model (Donut) on the synthetic
dataset, then you evaluate it on the **real** held-out `test_documents/`.

**Setup on Kaggle**
1. *Add Input* → attach the uploaded `german-docs-synthetic` dataset.
2. *Settings* → Accelerator → **GPU T4**.
3. Run All. Checkpoints land in `/kaggle/working/donut-german`.

If your dataset's input path differs, fix `DATA` in the config cell.
"""),

    ("code", """!pip -q install -U "transformers>=4.40" datasets sentencepiece"""),

    ("code", """import json, random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor, VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

# --- config ---------------------------------------------------------------
DATA = Path("/kaggle/input/german-docs-synthetic")   # images/ + manifest.*.jsonl
OUT = "/kaggle/working/donut-german"
BASE_MODEL = "naver-clova-ix/donut-base"
TASK = "<s_docextract>"
MAX_LEN = 384
EPOCHS = 3
MAX_TRAIN = 6000     # start with a subset to fit a Kaggle session; raise later
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| dataset present:", DATA.exists())"""),

    ("code", """# Load the manifests + the pretrained Donut, and register field tokens.
train_recs = [json.loads(l) for l in open(DATA / "manifest.train.jsonl")]
val_recs   = [json.loads(l) for l in open(DATA / "manifest.val.jsonl")]
random.Random(0).shuffle(train_recs)
print(f"train={len(train_recs)}  val={len(val_recs)}")

processor = DonutProcessor.from_pretrained(BASE_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

field_keys = sorted({k for r in train_recs for k in r["fields"]})
new_tokens = [TASK] + [f"<s_{k}>" for k in field_keys] + [f"</s_{k}>" for k in field_keys]
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK)
model.config.use_cache = False       # required with gradient checkpointing
print("added", len(new_tokens), "special tokens for fields:", field_keys)"""),

    ("code", """# image -> pixel_values ; fields dict -> Donut token sequence -> labels
def json2token(obj):
    if isinstance(obj, dict):
        return "".join(f"<s_{k}>{json2token(v)}</s_{k}>" for k, v in sorted(obj.items()))
    return str(obj)

class DocDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(DATA / r["image"]).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.squeeze(0)
        target = TASK + json2token(r["fields"]) + processor.tokenizer.eos_token
        labels = processor.tokenizer(
            target, add_special_tokens=False, max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

train_ds = DocDataset(train_recs[:MAX_TRAIN])
val_ds = DocDataset(val_recs[:400])
print("training on", len(train_ds), "samples")"""),

    ("code", """args = Seq2SeqTrainingArguments(
    output_dir=OUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
processor.save_pretrained(OUT)
model.save_pretrained(OUT)
print("saved checkpoint to", OUT)"""),

    ("code", """# Sanity decode on a held-out val image (synthetic) — does it parse to fields?
model.eval().to(device)
r = val_recs[0]
pv = processor(Image.open(DATA / r["image"]).convert("RGB"), return_tensors="pt").pixel_values.to(device)
out = model.generate(
    pv, max_length=MAX_LEN,
    decoder_start_token_id=model.config.decoder_start_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)
print("PRED:", processor.token2json(processor.batch_decode(out)[0]))
print("TRUE:", r["fields"])"""),

    ("markdown", """## Next: evaluate on the REAL hold-out

The number that matters comes from the *real* documents, not synthetic val:

1. Download `/kaggle/working/donut-german` (the **Output** tab).
2. Locally, wrap it behind the project's extractor interface and run
   `python scripts/eval_german_documents.py` on `test_documents/` for the
   head-to-head vs **Surya 90% / Apple Vision 85%**.

A model trained only on synthetic data that scores well on the real, degraded
documents is the result worth showing — it proves the synthetic pipeline
transfers. If accuracy is weak on a field, add targeted layout/degradation
variety to the generator and retrain (raise `MAX_TRAIN`).
"""),
]


def _cell(kind: str, src: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    nb = {
        "cells": [_cell(k, s) for k, s in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / "train_donut.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()

## Kazakhstan History QA Bot (11th Grade)

This project fine-tunes an open LLM (Qwen 2.5 3B Instruct via Unsloth + QLoRA) to answer factual questions about the **History of Kazakhstan (11th grade, textbook pages 3–23)**.

The final result is:
- **Clean OCR text** of the textbook fragment.
- **Supervised QA dataset** (300+ high‑quality fact‑based Q/A pairs, mostly in Kazakh, ENТ‑style).
- **Fine‑tuned instruction model** that answers short exam‑like questions on Kazakh history.

## Repository contents

- `history_finetuning.ipynb` – main training notebook with:
  - OCR text loading,
  - SFT dataset preparation,
  - Unsloth + QLoRA fine‑tuning for `Qwen/Qwen2.5-3B-Instruct`,
  - training logs and loss plot,
  - comparison of base vs fine‑tuned model answers,
  - optional upload to Hugging Face (dataset + model).
- `history_text.txt` – extracted textbook text (History of Kazakhstan, 11th grade, pages 3–23, Russian).
- `kz-tarih-qa-dataset.jsonl` – SFT dataset, one JSON object per line:
  ```json
  {"question": "...", "answer": "..."}
  ```

## Training setup

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Framework:** Unsloth + `transformers`, `trl`, `peft`, `bitsandbytes`
- **Method:** QLoRA (4‑bit), instruction‑style SFT on concatenated prompts.
- **Key hyperparameters:**
  - max sequence length: 1024
  - epochs: 4
  - batch size: 2, grad accumulation: 4
  - learning rate: 2e‑4
  - LoRA: r = 16, alpha = 32, dropout = 0.05
- **Result:** training loss decreases from ~1.1 to **≈0.27** (see loss plot in the notebook).

## Hugging Face resources

Replace placeholders with your actual URLs:

- **Dataset:** `TSaltanat/kz-tarih-11grade-qa-dataset`
- **Model (QLoRA adapters):** `TSaltanat/kz-tarih-qwen2.5-3b-qlora-kazakh-history`

Example loading code:

```python
from unsloth import FastLanguageModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "TSaltanat/kz-tarih-qwen2.5-3b-qlora-kazakh-history"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    adapter_name = ADAPTER_REPO,
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

def generate(question: str, max_new_tokens: int = 128) -> str:
    prompt = f"""Ниже приведён вопрос по истории Казахстана и ожидается краткий, точный и ясный ответ.

Вопрос: {question}
Ответ:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
```

## How to run locally

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install "unsloth[torch]" "transformers>=4.39.0" "accelerate>=0.28.0" \
       "peft>=0.11.0" "bitsandbytes>=0.43.0" "datasets>=2.18.0" "trl>=0.9.0"
   ```
3. Open `history_finetuning.ipynb` in Jupyter/Colab and run all cells to:
   - load the dataset,
   - fine‑tune the model,
   - test QA generation.

## License and usage

- The training data is derived from the **History of Kazakhstan, 11th grade** textbook (pages 3–23).\n- The fine‑tuned model is intended for **educational and research use only**.


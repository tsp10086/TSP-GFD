"""Prepare class/relation prototype tensors using locally available models."""
import argparse
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm", required=True, help="Local model directory or Hugging Face model name")
    p.add_argument("--encoder", required=True, help="Local sentence encoder or Hugging Face model name")
    p.add_argument("--output", type=Path, required=True, help="Locally generated prototype cache")
    args = p.parse_args()
    schema = json.loads((ROOT / "prompts" / "task_schema.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=dtype).to(device).eval()
    keys = ("benign", "risk", "relation")
    generated = []
    with torch.inference_mode():
        for key in keys:
            prompt = schema["instruction"] + schema[key]
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            output = llm.generate(**tokens, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(output[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            generated.append(text or schema[key])
    del llm
    encoder = SentenceTransformer(args.encoder, device="cpu")
    def encode(texts):
        return encoder.encode(texts, batch_size=32, convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False).detach().cpu().float()
    result = {"llm": encode(generated), "schema": encode([schema[k] for k in keys])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)

if __name__ == "__main__":
    main()
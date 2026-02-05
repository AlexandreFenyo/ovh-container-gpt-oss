
import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download, save_torch_model


BASE_MODEL_ID = "gpt-oss-20b"
LORA_ADAPTER_ID = "gpt-oss-20b-lora"


def maybe_copy_file_from_hub(repo_id: str, filename: str, out_dir: str) -> None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(out_dir, filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./gpt-oss-20b-merged")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_shard_size", type=str, default="2GB")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # Tokenizer (souvent le repo LoRA contient les bons fichiers)
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_ID, use_fast=True)

    # Charger le modèle de base
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,                 # (torch_dtype est deprecated)
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )

    # Charger l'adaptateur LoRA
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)

    # Merger LoRA -> poids standards
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Sauver config + tokenizer (OK via transformers)
    model.config.save_pretrained(args.out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # 2) Sauver les POIDS sans passer par transformers.save_pretrained()
    #    -> évite revert_weight_conversion() et donc le NotImplementedError.
    #
    #    Important: pour les modèles HF/transformers, on conseille de passer
    #    model._tied_weights_keys pour gérer proprement les tenseurs partagés.
    tied = getattr(model, "_tied_weights_keys", None)
    save_torch_model(
        model,
        args.out_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
        shared_tensors_to_discard=tied,
    )

    # Fichiers annexes utiles si présents
    maybe_copy_file_from_hub(LORA_ADAPTER_ID, "chat_template.jinja", args.out_dir)

    print(f"OK: modèle mergé (poids standalone) sauvegardé dans: {args.out_dir}")


if __name__ == "__main__":
    main()


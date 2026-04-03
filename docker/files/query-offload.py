import argparse
import glob
import os
import re
import sys
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful chatbot assistant for the Mon Espace Santé website. "
    "You answer only based on the information present in the FAQ. "
    "If the information is not available, you must respond with the predefined refusal message and nothing else."
)

CHANNEL_PATTERN = re.compile(r"<\|channel\|>(analysis|final)")


def _load_params(path):
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            params[key.strip()] = value.strip().strip('"')
    return params


def _resolve_param(params, key, default):
    value = params.get(key)
    return value if value else default


def _extract_channel(text, channel_name):
    marker = f"<|channel|>{channel_name}"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_match = CHANNEL_PATTERN.search(text, start)
    end = next_match.start() if next_match else len(text)
    return text[start:end].strip()


def _strip_special_tokens(text):
    return (
        text.replace("<|start|>", "")
        .replace("<|message|>", "")
        .replace("<|end|>", "")
        .replace("<|assistant|>", "")
        .strip()
    )


def _sanitize_component(value):
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"\s+", "_", text)
    text = text.replace("\n", "_").replace("\r", "_")
    return text or "empty"


def _natural_checkpoint_key(path):
    name = Path(path).name
    match = re.search(r"checkpoint-(\d+)", name)
    return (0, int(match.group(1))) if match else (1, name)


def _extract_row_value(row, key):
    if key in row:
        return row[key]
    for container_key in ("metadata", "info", "sample", "example"):
        container = row.get(container_key)
        if isinstance(container, dict) and key in container:
            return container[key]
    raise KeyError(f"Missing required field in dataset row: {key}")


def _extract_user_prompt(messages):
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content") or ""
    raise KeyError("Missing user message in dataset row")


def _load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(base_model_name, device_map, dtype):
    quantization_config = Mxfp4Config(dequantize=True)
    return AutoModelForCausalLM.from_pretrained(
        base_model_name,
        attn_implementation="eager",
        dtype=dtype,
        quantization_config=quantization_config,
        use_cache=False,
        device_map=device_map,
    )


def _load_model(base_model_name, adapter_path, adapter_name, device_map, dtype):
    base_model = _load_base_model(base_model_name, device_map, dtype)
    if adapter_path is None:
        return base_model
    model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    return model


def _build_messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate_response(model, tokenizer, system_prompt, user_prompt, max_new_tokens):
    messages = _build_messages(system_prompt, user_prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    analysis = _extract_channel(raw_response, "analysis")
    final = _extract_channel(raw_response, "final")

    thinking = _strip_special_tokens(analysis) if analysis else ""
    assistant = _strip_special_tokens(final) if final else _strip_special_tokens(raw_response)
    return thinking, assistant


def _write_text_file(path, text):
    if path.exists():
        print(f"ERROR: fichier deja existant: {path}")
        raise SystemExit(1)
    path.write_text(text, encoding="utf-8")


def _release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Génère les réponses et le thinking pour le base model, les checkpoints et le modèle final."
    )
    parser.add_argument("--answers-dir", type=str, default="answers")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--params-path", type=str, default="params.cfg")
    parser.add_argument("--base-model-name", type=str, default=None)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    params_path = args.params_path
    if not os.path.isabs(params_path):
        candidate = os.path.join(os.path.dirname(__file__), params_path)
        params_path = candidate if os.path.exists(candidate) else params_path
    params = _load_params(params_path) if os.path.exists(params_path) else {}

    dataset_name = args.dataset_name or _resolve_param(params, "var_dataset_name", "fenyo/FAQ-MES-WEB")
    base_model_name = args.base_model_name or "gpt-oss-20b"
    adapter_dir = args.adapter_dir or _resolve_param(params, "ft_output_dir", "gpt-oss-20b-lora")
    tokenizer_name = args.tokenizer_name or _resolve_param(params, "tokenizer", base_model_name)
    system_prompt = args.system_prompt or _resolve_param(params, "system_prompt", DEFAULT_SYSTEM_PROMPT)

    answers_dir = Path(args.answers_dir)
    if answers_dir.exists():
        print(f"ERROR: le repertoire existe deja: {answers_dir}")
        raise SystemExit(1)
    answers_dir.mkdir(parents=True)

    checkpoint_dirs = sorted(glob.glob(os.path.join(adapter_dir, "checkpoint*")), key=_natural_checkpoint_key)

    tokenizer = _load_tokenizer(tokenizer_name)
    dtype = getattr(torch, args.dtype)
    dataset = load_dataset(dataset_name, split=args.dataset_split)
    print(f"Dataset: {dataset_name} / {args.dataset_split} ({len(dataset)} rows)")
    print(f"Answers: {answers_dir}")

    model_specs = [("base", None)]
    model_specs.extend((Path(path).name, path) for path in checkpoint_dirs)
    model_specs.append(("final", adapter_dir))

    for model_name, adapter_path in model_specs:
        print(f"Running model: {model_name}")
        if adapter_path is None:
            model = _load_model(base_model_name, None, model_name, args.device_map, dtype)
        else:
            print(f"Loading checkpoint adapter: {adapter_path}")
            model = _load_model(base_model_name, adapter_path, model_name, args.device_map, dtype)
        model.eval()
        for index, row in enumerate(dataset):
            messages = row["messages"]
            user_prompt = _extract_user_prompt(messages)
            origin = _sanitize_component(_extract_row_value(row, "origin"))
            row_id = _sanitize_component(_extract_row_value(row, "id"))
            row_type = _sanitize_component(_extract_row_value(row, "type"))
            variant_q = _sanitize_component(_extract_row_value(row, "variant_q"))
            variant_a = _sanitize_component(_extract_row_value(row, "variant_a"))

            assistant_path = answers_dir / (
                f"{model_name}-assistant-{origin}-{row_id}-{row_type}-{variant_q}-{variant_a}.txt"
            )
            thinking_path = answers_dir / (
                f"{model_name}-thinking-{origin}-{row_id}-{row_type}-{variant_q}-{variant_a}.txt"
            )

            if model_name == "base":
                with model.disable_adapter():
                    thinking, assistant = _generate_response(
                        model,
                        tokenizer,
                        system_prompt,
                        user_prompt,
                        args.max_new_tokens,
                    )
            else:
                model.set_adapter(model_name)
                thinking, assistant = _generate_response(
                    model,
                    tokenizer,
                    system_prompt,
                    user_prompt,
                    args.max_new_tokens,
                )

            _write_text_file(assistant_path, assistant)
            _write_text_file(thinking_path, thinking)

            if index % 25 == 0:
                print(f"  {index + 1}/{len(dataset)}")

        _release_model(model)


if __name__ == "__main__":
    main()

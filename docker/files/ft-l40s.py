import os
import shlex

import torch
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils.quantization_config import Mxfp4Config
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

wandb.login(key=os.environ["wandbkey"])
login(token=os.environ["hfkey"])

quantization_config = Mxfp4Config(dequantize=True)

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    use_cache=False,
    # device_map="auto",
)


def _coerce_value(raw):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _load_params(path):
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = shlex.split(stripped)
            if not tokens or "=" not in tokens[0]:
                continue
            key, first_value = tokens[0].split("=", 1)
            values = [first_value] + tokens[1:]
            if len(values) == 1:
                params[key] = _coerce_value(values[0])
            else:
                params[key] = [_coerce_value(v) for v in values]
    return params


def _get_param(params, key, alias=None, default=None, required=True):
    if key in params:
        return params[key]
    if alias and alias in params:
        return params[alias]
    if required:
        raise ValueError(f"Missing required parameter: {key}")
    return default


def _describe_messages(messages, limit=800):
    text = repr(messages)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _describe_message_structure(messages):
    if not isinstance(messages, list):
        return f"type={type(messages).__name__} value={_describe_messages(messages)}"
    parts = []
    for index, message in enumerate(messages):
        if isinstance(message, dict):
            content = message.get("content")
            parts.append(
                f"[{index}] keys={sorted(message.keys())} "
                f"role={type(message.get('role')).__name__} "
                f"content_type={type(content).__name__} "
                f"content_none={content is None} "
                f"thinking_type={type(message.get('thinking')).__name__} "
                f"thinking_none={message.get('thinking') is None}"
            )
        else:
            parts.append(f"[{index}] type={type(message).__name__} value={_describe_messages(message)}")
    return " | ".join(parts)


def _strip_thinking(messages, drop_thinking_always=False, drop_thinking_none=False):
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for message in messages:
        if isinstance(message, dict):
            if drop_thinking_always:
                cleaned.append({k: v for k, v in message.items() if k != "thinking"})
            elif drop_thinking_none and message.get("thinking") is None:
                cleaned.append({k: v for k, v in message.items() if k != "thinking"})
            else:
                cleaned.append(message)
        else:
            cleaned.append(message)
    return cleaned


def _prepare_chat_messages(messages, drop_thinking_always=False, drop_thinking_none=False):
    return _strip_thinking(messages, drop_thinking_always, drop_thinking_none)


def _render_chat_text(tokenizer, messages, drop_thinking_always=False, drop_thinking_none=False):
    prepared_messages = _prepare_chat_messages(
        messages,
        drop_thinking_always=drop_thinking_always,
        drop_thinking_none=drop_thinking_none,
    )
    return tokenizer.apply_chat_template(prepared_messages, tokenize=False, add_generation_prompt=False)


def _materialize_text_dataset(
    dataset,
    tokenizer,
    dataset_name,
    drop_thinking_always=False,
    drop_thinking_none=False,
):
    rows = []
    for index, row in enumerate(dataset):
        messages = row["messages"]
        try:
            text = _render_chat_text(
                tokenizer,
                messages,
                drop_thinking_always=drop_thinking_always,
                drop_thinking_none=drop_thinking_none,
            )
        except Exception as exc:
            prepared_messages = _prepare_chat_messages(
                messages,
                drop_thinking_always=drop_thinking_always,
                drop_thinking_none=drop_thinking_none,
            )
            print(f"{dataset_name}[{index}] failed while rendering chat text")
            print(f"{dataset_name}[{index}] messages={_describe_messages(messages)}")
            print(f"{dataset_name}[{index}] structure={_describe_message_structure(messages)}")
            print(f"{dataset_name}[{index}] prepared_messages={_describe_messages(prepared_messages)}")
            print(f"{dataset_name}[{index}] prepared_structure={_describe_message_structure(prepared_messages)}")
            raise RuntimeError(f"Failed to render {dataset_name} row {index}") from exc
        rows.append({"text": text})
    return Dataset.from_list(rows)


def _trace_raw_chat_dataset(
    dataset,
    tokenizer,
    dataset_name,
    drop_thinking_always=False,
    drop_thinking_none=False,
    sample_limit=3,
):
    print(f"Tracing dataset {dataset_name}: {len(dataset)} rows")
    for index in range(min(sample_limit, len(dataset))):
        row = dataset[index]
        messages = row.get("messages")
        prepared_messages = _prepare_chat_messages(
            messages,
            drop_thinking_always=drop_thinking_always,
            drop_thinking_none=drop_thinking_none,
        )
        print(f"{dataset_name}[{index}] keys={list(row.keys())}")
        print(f"{dataset_name}[{index}] messages={_describe_messages(messages)}")
        print(f"{dataset_name}[{index}] structure={_describe_message_structure(messages)}")
        if prepared_messages != messages:
            print(f"{dataset_name}[{index}] prepared_messages={_describe_messages(prepared_messages)}")
            print(f"{dataset_name}[{index}] prepared_structure={_describe_message_structure(prepared_messages)}")
        text = tokenizer.apply_chat_template(prepared_messages, tokenize=False, add_generation_prompt=False)
        print(f"{dataset_name}[{index}] text={_describe_messages(text)}")


def _trace_text_dataset(dataset, dataset_name, sample_limit=3):
    print(f"Tracing text dataset {dataset_name}: {len(dataset)} rows")
    for index in range(min(sample_limit, len(dataset))):
        row = dataset[index]
        text = row.get("text")
        print(f"{dataset_name}[{index}] keys={list(row.keys())}")
        print(f"{dataset_name}[{index}] text={_describe_messages(text)}")


params_path = os.environ.get("PARAMS_CFG", "params.cfg")
if not os.path.isabs(params_path):
    params_path = os.path.join(os.path.dirname(__file__), params_path)
params = _load_params(params_path)

known_keys = {
    "var_dataset_name",
    "var_wandb_project",
    "var_wandb_run",
    "wandb_notebook_name",
    "tokenizer",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
    "ft_learning_rate",
    "ft_gradient_checkpointing",
    "ft_num_train_epochs",
    "ft_logging_steps",
    "ft_per_device_train_batch_size",
    "ft_gradient_accumulation_steps",
    "ft_max_length",
    "ft_warmup_ratio",
    "ft_lr_scheduler_type",
    "ft_output_dir",
    "ft_push_to_hub",
    "ft_report_to",
    "ft_eval_strategy",
    "ft_eval_steps",
    "ft_drop_thinking_always",
    "ft_drop_thinking_none",
}
unknown_keys = sorted(k for k in params.keys() if k not in known_keys)
if unknown_keys:
    print(f"Unknown keys in {params_path}: {', '.join(unknown_keys)}")

target_modules = _get_param(params, "lora_target_modules")
if isinstance(target_modules, str):
    target_modules = [target_modules]

resolved_params = {
    "var_dataset_name": _get_param(params, "var_dataset_name"),
    "var_wandb_project": _get_param(params, "var_wandb_project"),
    "var_wandb_run": _get_param(params, "var_wandb_run"),
    "wandb_notebook_name": _get_param(params, "wandb_notebook_name"),
    "tokenizer": _get_param(params, "tokenizer"),
    "lora_r": _get_param(params, "lora_r"),
    "lora_alpha": _get_param(params, "lora_alpha"),
    "lora_dropout": _get_param(params, "lora_dropout"),
    "lora_bias": _get_param(params, "lora_bias"),
    "lora_target_modules": target_modules,
    "ft_learning_rate": _get_param(params, "ft_learning_rate"),
    "ft_gradient_checkpointing": _get_param(params, "ft_gradient_checkpointing"),
    "ft_num_train_epochs": _get_param(params, "ft_num_train_epochs"),
    "ft_logging_steps": _get_param(params, "ft_logging_steps"),
    "ft_per_device_train_batch_size": _get_param(params, "ft_per_device_train_batch_size"),
    "ft_gradient_accumulation_steps": _get_param(params, "ft_gradient_accumulation_steps"),
    "ft_max_length": _get_param(params, "ft_max_length"),
    "ft_warmup_ratio": _get_param(params, "ft_warmup_ratio"),
    "ft_lr_scheduler_type": _get_param(params, "ft_lr_scheduler_type"),
    "ft_output_dir": _get_param(params, "ft_output_dir"),
    "ft_push_to_hub": _get_param(params, "ft_push_to_hub"),
    "ft_report_to": _get_param(params, "ft_report_to"),
    "ft_eval_strategy": _get_param(params, "ft_eval_strategy"),
    "ft_eval_steps": _get_param(params, "ft_eval_steps"),
    "ft_drop_thinking_always": _get_param(params, "ft_drop_thinking_always", required=False, default=False),
    "ft_drop_thinking_none": _get_param(params, "ft_drop_thinking_none", required=False, default=False),
}

print(f"Config values from {params_path}:")
for key in sorted(resolved_params.keys()):
    print(f"  {key}={resolved_params[key]}")

var_dataset_name = resolved_params["var_dataset_name"]
var_wandb_project = resolved_params["var_wandb_project"]
var_wandb_run = resolved_params["var_wandb_run"]
wandb_notebook_name = resolved_params["wandb_notebook_name"]
tokenizer_name = resolved_params["tokenizer"]
model_name = tokenizer_name

os.environ["WANDB_NOTEBOOK_NAME"] = wandb_notebook_name
wandb.init(project=var_wandb_project, entity="alexandre-fenyo-fenyonet", name=var_wandb_run)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

train_dataset = load_dataset(var_dataset_name, split="train")
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "messages"])

eval_dataset = load_dataset(var_dataset_name, split="validation")
eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "messages"])

drop_thinking_always = resolved_params["ft_drop_thinking_always"]
drop_thinking_none = resolved_params["ft_drop_thinking_none"]

if drop_thinking_always:
    print("Thinking mode: drop thinking from all messages")
elif drop_thinking_none:
    print("Thinking mode: drop thinking only when it is None")
else:
    print("Thinking mode: keep thinking unchanged")

train_dataset = _materialize_text_dataset(
    train_dataset,
    drop_thinking_always=drop_thinking_always,
    drop_thinking_none=drop_thinking_none,
    tokenizer=tokenizer,
    dataset_name="train",
)

eval_dataset = _materialize_text_dataset(
    eval_dataset,
    drop_thinking_always=drop_thinking_always,
    drop_thinking_none=drop_thinking_none,
    tokenizer=tokenizer,
    dataset_name="validation",
)

_trace_raw_chat_dataset(
    train_dataset,
    tokenizer,
    "train",
    drop_thinking_always=drop_thinking_always,
    drop_thinking_none=drop_thinking_none,
)
_trace_raw_chat_dataset(
    eval_dataset,
    tokenizer,
    "validation",
    drop_thinking_always=drop_thinking_always,
    drop_thinking_none=drop_thinking_none,
)

_trace_text_dataset(train_dataset, "train")
_trace_text_dataset(eval_dataset, "validation")

peft_config = LoraConfig(
    r=resolved_params["lora_r"],
    lora_alpha=resolved_params["lora_alpha"],
    lora_dropout=resolved_params["lora_dropout"],
    bias=resolved_params["lora_bias"],
    target_modules=target_modules,
)

training_args = SFTConfig(
    learning_rate=resolved_params["ft_learning_rate"],
    gradient_checkpointing=resolved_params["ft_gradient_checkpointing"],
    num_train_epochs=resolved_params["ft_num_train_epochs"],
    logging_steps=resolved_params["ft_logging_steps"],
    per_device_train_batch_size=resolved_params["ft_per_device_train_batch_size"],
    gradient_accumulation_steps=resolved_params["ft_gradient_accumulation_steps"],
    max_length=resolved_params["ft_max_length"],
    warmup_ratio=resolved_params["ft_warmup_ratio"],
    lr_scheduler_type=resolved_params["ft_lr_scheduler_type"],
    output_dir=resolved_params["ft_output_dir"],
    push_to_hub=resolved_params["ft_push_to_hub"],
    report_to=resolved_params["ft_report_to"],
    eval_strategy=resolved_params["ft_eval_strategy"],
    eval_steps=resolved_params["ft_eval_steps"],
    save_strategy="epoch",
    save_total_limit=100,
    bf16=False,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    dataset_text_field="text",
)

trainer.train()
trainer.save_model(training_args.output_dir)

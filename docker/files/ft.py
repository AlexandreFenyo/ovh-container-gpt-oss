
import os
import shlex
import wandb
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer

wandb.login(key=os.environ['wandbkey'])
login(token=os.environ['hfkey'])

quantization_config = Mxfp4Config(dequantize=True)

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
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

os.environ['WANDB_NOTEBOOK_NAME'] = wandb_notebook_name
wandb.init(project=var_wandb_project, entity="alexandre-fenyo-fenyonet", name=var_wandb_run)

# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Chargement du modèle
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

# Chargement des jeux de données
train_dataset = load_dataset(var_dataset_name, split="train")
# Conserve uniquement la colonne "messages" (ou adapte selon ton schéma)
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "messages"])

eval_dataset = load_dataset(var_dataset_name, split="validation")
eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "messages"])

peft_config = LoraConfig(
    r=resolved_params["lora_r"],
    lora_alpha=resolved_params["lora_alpha"],
    lora_dropout=resolved_params["lora_dropout"],
    bias=resolved_params["lora_bias"],
    target_modules=target_modules,
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_kwargs = dict(
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
)

training_args = SFTConfig(**training_kwargs)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)
# trainer.push_to_hub(dataset_name=var_dataset_name)

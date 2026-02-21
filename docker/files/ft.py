
import os
import wandb
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer

var_dataset_name = os.environ['dataset_name']
var_wandb_project = os.environ['wandb_project']
var_wandb_run = os.environ['wandb_run']

os.environ['WANDB_NOTEBOOK_NAME'] = 'gpt-oss-20b-FAQ-MES'
wandb.login(key=os.environ['wandbkey'])
wandb.init(project=var_wandb_project, entity="alexandre-fenyo-fenyonet", name=var_wandb_run)
login(token=os.environ['hfkey'])

# Chargement des jeux de données
train_dataset = load_dataset(var_dataset_name, split="train")
# Conserve uniquement la colonne "messages" (ou adapte selon ton schéma)
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "messages"])

eval_dataset = load_dataset(var_dataset_name, split="validation")
eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "messages"])

tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")
quantization_config = Mxfp4Config(dequantize=True)

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    # device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained("gpt-oss-20b", **model_kwargs)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="all",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(
    learning_rate=1e-4,
    gradient_checkpointing=True,
    num_train_epochs=20,
    logging_steps=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    max_length=2048,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    output_dir="gpt-oss-20b-lora",
    push_to_hub=True,
    report_to="wandb",
    # Activation de l’évaluation
    eval_strategy="epoch",   # 'steps' ou 'epoch'
#    eval_steps=1000,           # fréquence d’évaluation (toutes les 1000 steps)
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,      # NEW: dataset d’évaluation
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)
# trainer.push_to_hub(dataset_name=var_dataset_name)


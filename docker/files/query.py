
import argparse
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import Mxfp4Config

def main():
    # Définir l'argument en ligne de commande
    parser = argparse.ArgumentParser(description="Charge un modèle avec option de remplacement du nom via CLI et possibilité de spécifier le prompt utilisateur.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nom du modèle à charger via from_pretrained. Si omis, utilise le nom par défaut 'gpt-oss-20b-lora'."
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="Contenu du message utilisateur à envoyer comme dernier message du contexte. Si omis, utilise le prompt par défaut."
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")

    quantization_config = Mxfp4Config(dequantize=True)

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,            # remplace torch_dtype deprecated
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    def load_model(model_override_name=None, **kwargs):
        model_name = model_override_name if model_override_name else "gpt-oss-20b-lora"
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Utilisation du nom du modèle passé en CLI (ou défaut)
    model = load_model(args.model_name, **model_kwargs)

    default_user_prompt = "Quelles sont les étapes pour le compte Mon espace santé de mon enfant qui vient d’atteindre la majorité ?"
    user_prompt = args.user_prompt if args.user_prompt is not None else default_user_prompt

    messages = [
        {"role": "system", "content": "You are a helpful chatbot assistant for the Mon Espace Santé website."},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,   # obtenir un dict avec input_ids, attention_mask, etc.
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=2048)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(response)

if __name__ == "__main__":
    main()



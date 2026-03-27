import argparse
import traceback

import tokenizers
import transformers
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "Ministral-8B-Instruct-2410"
DEFAULT_TEXT = "Mon espace sante, qu'est-ce que c'est ?"


def try_load(tokenizer_name, use_fast, fix_mistral_regex):
    print(
        f"TEST load use_fast={use_fast} "
        f"fix_mistral_regex={fix_mistral_regex}"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=use_fast,
            fix_mistral_regex=fix_mistral_regex,
        )
        encoded = tokenizer(DEFAULT_TEXT, return_attention_mask=False)
        token_count = len(encoded["input_ids"])
        print("  RESULT: ok")
        print(f"  TOKENIZER_CLASS: {tokenizer.__class__.__name__}")
        print(f"  TOKEN_COUNT: {token_count}")
        print(f"  FIRST_IDS: {encoded['input_ids'][:16]}")
        return True
    except Exception as exc:
        print("  RESULT: error")
        print(f"  ERROR_TYPE: {type(exc).__name__}")
        print(f"  ERROR: {exc}")
        print("  TRACEBACK:")
        print(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostique le chargement du tokenizer Mistral/Ministral "
            "avec ou sans fix_mistral_regex."
        )
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer a tester. Par defaut: {DEFAULT_TOKENIZER}",
    )
    args = parser.parse_args()

    print(f"transformers={transformers.__version__}")
    print(f"tokenizers={tokenizers.__version__}")
    print(f"tokenizer_name={args.tokenizer_name}")
    print()

    results = []
    for use_fast in (True, False):
        for fix_mistral_regex in (True, False):
            ok = try_load(
                args.tokenizer_name,
                use_fast=use_fast,
                fix_mistral_regex=fix_mistral_regex,
            )
            results.append((use_fast, fix_mistral_regex, ok))
            print()

    print("SUMMARY")
    for use_fast, fix_mistral_regex, ok in results:
        print(
            f"  use_fast={use_fast} "
            f"fix_mistral_regex={fix_mistral_regex} "
            f"ok={ok}"
        )


if __name__ == "__main__":
    main()

"""
Point config/llm_providers.json at OpenAI model names.

The committed configuration uses model IDs served through an OpenAI-compatible
gateway (set OPENAI_BASE_URL to use it). If you are calling the OpenAI API
directly with your own key, run this script once to switch every use case to
standard OpenAI model names.

Usage (from the repository root):
    uv run python scripts/use_openai_models.py
    uv run python scripts/use_openai_models.py --chat-model gpt-4o-mini
"""
import argparse
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/llm_providers.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat-model", default="gpt-4o", help="OpenAI chat model (default: gpt-4o)")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    openai = config["providers"]["openai"]
    openai["default_model"] = args.chat_model
    openai["default_embedding_model"] = args.embedding_model

    for name, use_case in config.get("use_cases", {}).items():
        if use_case.get("provider") != "openai":
            continue
        model = args.embedding_model if name == "embeddings" else args.chat_model
        use_case["model"] = model
        if "fallback_model" in use_case:
            use_case["fallback_model"] = model

    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {CONFIG_PATH.name}: chat model = {args.chat_model}, embedding model = {args.embedding_model}")


if __name__ == "__main__":
    main()

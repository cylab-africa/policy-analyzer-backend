# Contributing to the Policy Analyzer backend

Thank you for your interest in improving the Policy Analyzer. Everyone taking part in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** or **suggest a feature** by opening an issue.
- **Correct the data.** If a country's classification, a source link or a statement is wrong, outdated or missing, open a *Data or content correction* issue and link to the primary source (for example the official gazette).
- **Improve the documentation**, including the README.
- **Fix a bug or build a feature** with a pull request.

To report a security problem, do not open an issue. Follow [SECURITY.md](SECURITY.md).

## Before you start

For anything larger than a small fix, please open an issue first to discuss the approach. This avoids wasted effort if the change doesn't fit the project's direction.

## Setting up

Follow [Local development](README.md#local-development) in the README. You will need an OpenAI-compatible API key to run questions end to end.

## Making a change

1. Fork the repository and create a branch from `main`, such as `fix/short-description` or `feature/short-description`.
2. Keep the change focused. Separate unrelated changes into separate pull requests.
3. Test it and describe in the pull request what you ran and what you saw. Run `uv run pytest` before you push. The tests cover the country reference data, and CI runs them on every pull request. The suite is small, so adding tests with your change is very welcome.
4. Update the documentation when behavior or settings change: the README, `.env.example` and `config/`.
5. Open a pull request against `main` and fill in the template.

## Guidelines

- **No secrets.** Never commit `.env` files, API keys, passwords or credentials. `.env.example` holds placeholders only.
- **No private or copyrighted documents.** Don't commit policy documents you aren't permitted to redistribute, or large data files. Link to the source instead.
- **Prompt changes affect every answer.** If you change `config/prompts.json` or a prompt written in `app/services/query_handler.py`, explain why and include example questions with answers before and after.
- **Dependencies.** Edit `pyproject.toml`, then regenerate `uv.lock` (`uv lock`) and `requirements.txt` (`uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt`). LangChain packages are capped below 1.0 because the code uses APIs that 1.0 removed.
- **Style.** Match the surrounding code.

## Licensing of contributions

By submitting a contribution you agree that it is released under the project's [MIT License](LICENSE).

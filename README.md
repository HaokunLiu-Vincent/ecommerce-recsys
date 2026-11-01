# H&M Recommendation System

A reproducible recommendation system using historical purchases, article metadata,
and images. Project scope and the evaluation contract live in `docs/`.

## Local development

Use Python 3.12 and uv. This is a non-package project. DuckDB is the only current
runtime dependency; large ML frameworks are intentionally deferred.

Install uv using its official installer if needed, then run:

```sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

`uv.lock` pins runtime and development dependencies. Ruff handles Python and
notebooks; the commands above only check existing content. Review diagnostics
before making changes. Pyright uses standard mode and `.venv`, checking Python
source rather than exploratory notebooks. There are currently no maintained
Python source files or tests; Pyright may report nothing to analyze.

pytest exit code 5 means no tests were collected, not that tests passed. To treat
that state as a successful empty-suite check while preserving actual failures:

```sh
uv run pytest; result=$?
if [ "$result" -ne 0 ] && [ "$result" -ne 5 ]; then exit "$result"; fi
```

Add real tests under `tests/` when implementation starts. No dummy tests are needed.

pre-commit is configured but hooks are not installed because Git has not been
initialized. After separately choosing to initialize Git, run:

```sh
uv run pre-commit install
uv run pre-commit run --all-files
```

Ruff hooks check without automatic fixes. Whitespace/newline hooks exclude
notebooks. Keep raw data, image features, model artifacts and credentials local.
No datasets or models are downloaded by this setup.

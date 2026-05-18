# Examples

| File | What it does |
|------|--------------|
| `openai_quickstart.py` | **Start here if you have an API key.** Runs 50 factual QA questions against `gpt-4o-mini`, produces bootstrap CIs, a RigorChecker audit, cost summary, and error analysis. Cost under $0.01. |
| `full_workflow.py` | Complete pipeline: power analysis, evaluation, model comparison, FDR correction, judge validation, HTML tearsheet. No API keys required. |
| `benchmark_audit.py` | Five real-world failure modes: underpowered n=50 eval, class-imbalanced dataset, multiple testing false positives, unreliable LLM judge, and a well-designed passing experiment. |
| `walkthrough.ipynb` | Interactive notebook version of the full workflow. Open in Binder with no install: [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bonnie-mcconnell/evalkit/main?urlpath=lab/tree/examples/walkthrough.ipynb) |

## Quick start

```bash
# With an OpenAI API key (recommended first step)
pip install "evalkit-research[openai]"
export OPENAI_API_KEY=sk-...
python examples/openai_quickstart.py

# Without an API key (mock model, no cost)
pip install -e ".[dev]"       # from repo root
python examples/full_workflow.py
python examples/benchmark_audit.py
```

See `data/` for the demo datasets and their format documentation.

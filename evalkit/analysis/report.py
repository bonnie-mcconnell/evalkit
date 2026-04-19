"""
HTML report generator for evalkit experiment results.

Produces a self-contained HTML file with no external dependencies at render time.
All CSS, JavaScript, and data are inlined. The report can be emailed, opened
offline, or committed to a repo without breaking.

Design principle: the report is a *tearsheet* - a single-page summary that a
hiring manager or senior engineer can read in 2 minutes and understand exactly
what was evaluated and what the results mean statistically.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalkit.analysis.rigour import AuditReport
    from evalkit.core.experiment import ExperimentResult

from evalkit.analysis.rigour import _severity_sort_key

logger = logging.getLogger(__name__)


def _e(text: str) -> str:
    """Escape text for safe insertion into HTML."""
    return html_lib.escape(str(text))


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>evalkit - {{ experiment_name }}</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #00d4aa;
    --accent2: #7c5cfc;
    --warn: #f59e0b;
    --error: #ef4444;
    --ok: #22c55e;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
         font-size: 14px; line-height: 1.6; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
  header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
  header h1 { font-size: 1.6rem; font-weight: 700; color: var(--accent); letter-spacing: -0.02em; }
  header .meta { color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }  # noqa: E501
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }  # noqa: E501
  .card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }  # noqa: E501
  .card .value { font-size: 1.8rem; font-weight: 700; font-family: var(--mono); margin-top: 0.25rem; }  # noqa: E501
  .card .ci { font-size: 0.8rem; color: var(--muted); font-family: var(--mono); }
  .card .value.accent { color: var(--accent); }
  .section { margin-bottom: 2rem; }
  .section h2 { font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;  # noqa: E501
                color: var(--muted); margin-bottom: 1rem; border-bottom: 1px solid var(--border);
                padding-bottom: 0.5rem; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 0.6rem 1rem; font-size: 0.75rem; text-transform: uppercase;
       letter-spacing: 0.06em; color: var(--muted); border-bottom: 1px solid var(--border); }
  td { padding: 0.75rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.04);
       font-family: var(--mono); font-size: 0.85rem; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .finding { padding: 0.85rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; font-size: 0.875rem; }  # noqa: E501
  .finding.error { background: rgba(239,68,68,0.1); border-left: 3px solid var(--error); }
  .finding.warning { background: rgba(245,158,11,0.1); border-left: 3px solid var(--warn); }
  .finding.info { background: rgba(0,212,170,0.08); border-left: 3px solid var(--accent); }
  .finding .code { font-family: var(--mono); font-weight: 600; font-size: 0.8rem; }
  .finding .action { color: var(--muted); margin-top: 0.25rem; font-size: 0.82rem; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem;
           font-weight: 600; font-family: var(--mono); }
  .badge.pass { background: rgba(34,197,94,0.15); color: var(--ok); }
  .badge.fail { background: rgba(239,68,68,0.15); color: var(--error); }
  .ci-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
  .ci-bar { height: 6px; background: var(--border); border-radius: 3px; flex: 1; position: relative; min-width: 80px; }  # noqa: E501
  .ci-bar .fill { position: absolute; height: 100%; background: var(--accent2); border-radius: 3px; }  # noqa: E501
  .ci-bar .point { position: absolute; width: 2px; height: 10px; top: -2px; background: var(--accent); border-radius: 1px; }  # noqa: E501
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
           color: var(--muted); font-size: 0.8rem; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>evalkit - {{ experiment_name }}</h1>
    <div class="meta">Generated {{ generated_at }} · Model: {{ model }} · Dataset: {{ dataset_name }} (n={{ n }})</div>  # noqa: E501
  </header>

  <div class="grid">
    {{ metric_cards }}
    <div class="card">
      <div class="label">Cost</div>
      <div class="value accent">${{ total_cost }}</div>
      <div class="ci">{{ total_tokens }} tokens</div>
    </div>
    <div class="card">
      <div class="label">RigorChecker</div>
      <div class="value {{ rigor_status_class }}">{{ rigor_status }}</div>
      <div class="ci">{{ n_errors }} errors · {{ n_warnings }} warnings</div>
    </div>
  </div>

  <div class="section">
    <h2>Metrics with 95% Confidence Intervals</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>95% CI</th><th>n</th><th>Interval</th></tr></thead>  # noqa: E501
      <tbody>{{ metric_rows }}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>RigorChecker Audit <span class="badge {{ rigor_badge_class }}">{{ rigor_badge_text }}</span></h2>  # noqa: E501
    {{ findings_html }}
  </div>

  <div class="section">
    <h2>Run Details</h2>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>{{ run_details }}</tbody>
    </table>
  </div>
</div>
<footer>evalkit-research · Rigorous LLM evaluation · <a href="https://github.com/bonnie-mcconnell/evalkit" style="color:var(--accent)">github</a></footer>  # noqa: E501
</body>
</html>"""


class ReportGenerator:
    """
    Generate a self-contained HTML tearsheet from an ExperimentResult.

    The report is designed to be the artifact you share in a PR, email,
    or portfolio - it contains everything a reader needs to assess whether
    the evaluation was statistically sound.
    """

    def generate(
        self,
        result: ExperimentResult,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate the HTML report.

        Parameters
        ----------
        result:
            The ExperimentResult from Experiment.run().
        output_path:
            If provided, write the HTML to this path.

        Returns
        -------
        HTML string of the complete report.
        """
        html = self._render(result)

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("Report written to %s", output_path)

        return html

    def _render(self, result: ExperimentResult) -> str:
        run = result.run_result
        audit = result.posthoc_audit
        is_pass = audit.passed

        # Build substitution map and apply ALL replacements in a single regex pass.
        # Sequential str.replace() is vulnerable: if a replaced value contains
        # a template token (e.g. experiment named "{{ model }}"), a later call
        # would expand that token incorrectly. re.sub with a callback applies
        # all substitutions simultaneously, so no replaced value is re-scanned.
        substitutions = {
            "experiment_name": _e(result.experiment_name),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": _e(run.model),
            "dataset_name": _e(run.dataset_name),
            "n": str(run.n),
            "total_cost": f"{run.total_cost_usd:.4f}",
            "total_tokens": f"{run.total_tokens:,}",
            "rigor_status": "PASS" if is_pass else "FAIL",
            "rigor_status_class": "accent" if is_pass else "",
            "n_errors": str(len(audit.errors)),
            "n_warnings": str(len(audit.warnings)),
            "rigor_badge_class": "pass" if is_pass else "fail",
            "rigor_badge_text": "PASS" if is_pass else "FAIL",
            "metric_cards": self._metric_cards(result.metrics),
            "metric_rows": self._metric_rows(result.metrics),
            "findings_html": self._findings_html(audit),
            "run_details": self._run_details(result),
        }

        pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        return pattern.sub(
            lambda m: substitutions.get(m.group(1), m.group(0)),
            _REPORT_TEMPLATE,
        )

    def _metric_cards(self, metrics: dict[str, Any]) -> str:
        cards = []
        for name, m in list(metrics.items())[:3]:
            cards.append(
                f'<div class="card">'
                f'<div class="label">{_e(name)}</div>'
                f'<div class="value accent">{m.value:.4f}</div>'
                f'<div class="ci">95% CI: [{m.ci_lower:.4f}, {m.ci_upper:.4f}]</div>'
                f"</div>"
            )
        return "\n".join(cards)

    def _metric_rows(self, metrics: dict[str, Any]) -> str:
        rows = []
        for name, m in metrics.items():
            width_pct = m.ci_width * 100
            point_pct = m.value * 100
            bar = (
                f'<div class="ci-bar-wrap">'
                f'<div class="ci-bar">'
                f'<div class="fill" style="left:{m.ci_lower * 100:.1f}%;width:{width_pct:.1f}%"></div>'  # noqa: E501
                f'<div class="point" style="left:{point_pct:.1f}%"></div>'
                f"</div>"
                f'<span style="font-size:0.75rem;color:var(--muted)">±{m.margin_of_error:.4f}</span>'  # noqa: E501
                f"</div>"
            )
            rows.append(
                f"<tr>"
                f"<td>{_e(name)}</td>"
                f"<td>{m.value:.4f}</td>"
                f"<td>[{m.ci_lower:.4f}, {m.ci_upper:.4f}]</td>"
                f"<td>{m.n}</td>"
                f"<td>{bar}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _findings_html(self, audit: AuditReport) -> str:
        if not audit.findings:
            return '<p style="color:var(--ok)">✅ No issues found. Experiment appears statistically sound.</p>'  # noqa: E501

        parts = []
        for f in sorted(audit.findings, key=_severity_sort_key):
            parts.append(
                f'<div class="finding {f.severity.value}">'
                f'<div class="code">[{_e(f.code)}] {_e(f.message)}</div>'
                f'<div class="action">→ {_e(f.action)}</div>'
                f"</div>"
            )
        return "\n".join(parts)

    def _run_details(self, result: ExperimentResult) -> str:
        run = result.run_result
        rows_data = [
            ("Experiment", result.experiment_name),
            ("Model", run.model),
            ("Dataset", run.dataset_name),
            ("Examples evaluated", str(run.n)),
            ("Wall time", f"{run.wall_time_seconds:.2f}s"),
            ("Total tokens", f"{run.total_tokens:,}"),
            ("Estimated cost", f"${run.total_cost_usd:.6f}"),
            ("Raw accuracy", f"{sum(run.correct) / run.n:.4f}" if run.n else "N/A"),
        ]
        rows = [f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in rows_data]
        return "\n".join(rows)

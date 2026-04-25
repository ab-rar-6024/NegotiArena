"""
training/plot_curves.py — Generate reward curve HTML from training results
==========================================================================
Reads checkpoints/cpu_run/reward_curve.csv and produces a self-contained
HTML file with interactive Plotly charts — no matplotlib, no GPU.

Run:
    python -m training.plot_curves --input checkpoints/cpu_run/reward_curve.csv
    # Opens reward_curves.html in browser
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import webbrowser


def load_csv(path: str) -> dict[str, list]:
    data: dict[str, list] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def generate_html(data: dict[str, list], output_path: str) -> str:
    steps = data.get("step", [])
    f1 = data.get("overseer_f1", [])
    det = data.get("detection_rate", [])
    fp  = data.get("false_positive_rate", [])

    # Compute improvement annotations
    f1_before = f1[0] if f1 else 0
    f1_after  = f1[-1] if f1 else 0
    gap = f1_after - f1_before

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>NegotiArena Training Curves</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    body {{ background:#0f0f1a; color:#e0e0e0; font-family:sans-serif;
           max-width:1100px; margin:40px auto; padding:20px; }}
    h1 {{ color:#4a9eff; }}
    h3 {{ color:#aaa; font-weight:400; }}
    .metric-row {{ display:flex; gap:20px; margin:24px 0; }}
    .card {{ background:#1a1a2e; border:1px solid #333; border-radius:8px;
             padding:20px; flex:1; text-align:center; }}
    .card .val {{ font-size:2.2em; font-weight:bold; margin:8px 0; }}
    .card .label {{ color:#aaa; font-size:0.9em; }}
    .green {{ color:#00cc66; }}
    .red   {{ color:#ff4a4a; }}
    .blue  {{ color:#4a9eff; }}
    footer {{ color:#555; font-size:0.8em; margin-top:40px; text-align:center; }}
  </style>
</head>
<body>
  <h1>🏛️ NegotiArena — Training Results</h1>
  <h3>REINFORCE on CPU → same reward signal used for GRPO on GPU</h3>

  <div class="metric-row">
    <div class="card">
      <div class="label">Overseer F1 (Before)</div>
      <div class="val red">{f1_before:.3f}</div>
      <div class="label">Random baseline</div>
    </div>
    <div class="card">
      <div class="label">Overseer F1 (After)</div>
      <div class="val green">{f1_after:.3f}</div>
      <div class="label">After {int(steps[-1]) if steps else 0} training steps</div>
    </div>
    <div class="card">
      <div class="label">Improvement</div>
      <div class="val blue">+{gap:.3f}</div>
      <div class="label">{gap/f1_before*100:.0f}% relative gain</div>
    </div>
    <div class="card">
      <div class="label">FP Rate (After)</div>
      <div class="val green">{fp[-1]:.1%}</div>
      <div class="label">Was {fp[0]:.1%}</div>
    </div>
  </div>

  <div id="chart_f1"></div>
  <div id="chart_det"></div>

  <script>
  const steps = {json.dumps(steps)};
  const f1    = {json.dumps(f1)};
  const det   = {json.dumps(det)};
  const fp    = {json.dumps(fp)};

  const layout = {{
    paper_bgcolor: '#0f0f1a',
    plot_bgcolor:  '#1a1a2e',
    font: {{ color: '#cccccc' }},
    xaxis: {{ title: 'Training Step', gridcolor: '#333' }},
    legend: {{ bgcolor: '#1a1a2e' }},
    height: 360,
    margin: {{ t: 50, b: 50 }},
  }};

  // Chart 1: F1 curve with before/after annotation
  Plotly.newPlot('chart_f1', [
    {{
      x: steps, y: f1, mode: 'lines+markers',
      name: 'Overseer F1', line: {{ color: '#00cc66', width: 3 }},
      marker: {{ size: 6 }}
    }},
    {{
      x: [steps[0]], y: [f1[0]], mode: 'markers',
      name: 'Before', marker: {{ color: '#ff4a4a', size: 14, symbol: 'circle' }}
    }},
    {{
      x: [steps[steps.length-1]], y: [f1[f1.length-1]], mode: 'markers',
      name: 'After', marker: {{ color: '#00cc66', size: 14, symbol: 'star' }}
    }}
  ], {{
    ...layout,
    title: {{ text: 'Overseer F1 — Coalition Detection Improvement', font: {{ color: '#4a9eff', size: 16 }} }},
    yaxis: {{ title: 'F1 Score', range: [0, 1], gridcolor: '#333' }},
    annotations: [{{
      x: steps[steps.length-1], y: f1[f1.length-1],
      text: ` After: ${{f1[f1.length-1].toFixed(3)}}`,
      showarrow: true, arrowhead: 2,
      font: {{ color: '#00cc66', size: 13 }},
      bgcolor: '#1a2e1a', bordercolor: '#00cc66',
    }}]
  }});

  // Chart 2: Detection rate vs FP rate
  Plotly.newPlot('chart_det', [
    {{
      x: steps, y: det, mode: 'lines', name: 'Detection Rate',
      line: {{ color: '#4a9eff', width: 2 }}
    }},
    {{
      x: steps, y: fp, mode: 'lines', name: 'False Positive Rate',
      line: {{ color: '#ff9a4a', width: 2, dash: 'dot' }}
    }}
  ], {{
    ...layout,
    title: {{ text: 'Detection Rate vs False Positive Rate', font: {{ color: '#4a9eff', size: 16 }} }},
    yaxis: {{ title: 'Rate', range: [0, 1], gridcolor: '#333' }},
  }});
  </script>

  <footer>
    NegotiArena | Meta × Scaler OpenEnv Hackathon | Theme 1: Multi-Agent Interactions<br>
    CPU training uses REINFORCE with the same reward signal as GPU GRPO training.
  </footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/cpu_run/reward_curve.csv")
    parser.add_argument("--output", default="reward_curves.html")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open in browser after generating")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        print("   Run training first: python -m training.train_cpu --steps 200")
        return

    data = load_csv(args.input)
    out = generate_html(data, args.output)
    print(f"✅ Chart saved to {out}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
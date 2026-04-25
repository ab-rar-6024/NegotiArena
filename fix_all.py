"""
fix_all.py — Run this ONCE to fix all 8 blockers
=================================================
Run from D:\\negotiarena:
    python fix_all.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

def log(msg): print(f"  {msg}")

print("\n🔧 NegotiArena — Fixing all blockers\n")

# ── FIX 1: Delete root-level index.html (duplicate of server/index.html) ──
root_html = os.path.join(ROOT, "index.html")
if os.path.exists(root_html):
    os.remove(root_html)
    log("✅ Deleted root index.html (server/index.html is the real one)")
else:
    log("✅ Root index.html already gone")

# ── FIX 2: Create missing __init__.py files ──
for pkg in ["demo", "tests", "server", "training", "evaluation"]:
    init = os.path.join(ROOT, pkg, "__init__.py")
    os.makedirs(os.path.join(ROOT, pkg), exist_ok=True)
    if not os.path.exists(init):
        open(init, "w").close()
        log(f"✅ Created {pkg}/__init__.py")
    else:
        log(f"✅ {pkg}/__init__.py already exists")

# ── FIX 3: Delete duplicate sft_episodes.json (keep only .jsonl) ──
json_dup = os.path.join(ROOT, "data", "sft_episodes.json")
if os.path.exists(json_dup):
    os.remove(json_dup)
    log("✅ Deleted data/sft_episodes.json (keeping .jsonl)")
else:
    log("✅ No duplicate .json file")

# ── FIX 4: Delete old colab_training_notebook.py (replaced by .ipynb) ──
old_nb = os.path.join(ROOT, "training", "colab_training_notebook.py")
if os.path.exists(old_nb):
    os.remove(old_nb)
    log("✅ Deleted training/colab_training_notebook.py (replaced by .ipynb)")
else:
    log("✅ Old notebook already removed")

# ── FIX 5: Update .gitignore ──
gitignore_path = os.path.join(ROOT, ".gitignore")
gitignore_content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Environment
.env
.venv/
venv/

# Training outputs (too large for git/HF Spaces)
checkpoints/
wandb/

# Keep reward evidence
!checkpoints/cpu_run/
!checkpoints/cpu_run/reward_curve.csv
!checkpoints/cpu_run/training_results.json

# Data (regenerate with generate_sft_data.py)
data/

# Misc
.DS_Store
.idea/
.vscode/
*.pt
*.safetensors
sft_episodes.json
"""
with open(gitignore_path, "w") as f:
    f.write(gitignore_content)
log("✅ Updated .gitignore (keeps reward_curve.csv, excludes wandb/data/checkpoints)")

# ── FIX 6: Verify NegotiArena_Training.ipynb exists ──
nb_path = os.path.join(ROOT, "NegotiArena_Training.ipynb")
if os.path.exists(nb_path):
    log("✅ NegotiArena_Training.ipynb exists")
else:
    log("❌ NegotiArena_Training.ipynb MISSING — download from Claude outputs folder")

# ── FIX 7: Verify server/index.html exists ──
idx_path = os.path.join(ROOT, "server", "index.html")
if os.path.exists(idx_path):
    log("✅ server/index.html exists")
else:
    log("❌ server/index.html MISSING — download from Claude outputs folder")

# ── FIX 8: Print final checklist ──
print("\n" + "="*55)
print("  REMAINING MANUAL STEPS (do these next)")
print("="*55)

manual = [
    ("1", "Create HF account → huggingface.co/join"),
    ("2", "Run: pip install huggingface_hub"),
    ("3", "Run: huggingface-cli login  (paste your token)"),
    ("4", "Create Space: huggingface.co/new-space → Docker → name: negotiarena"),
    ("5", "Run: git remote add space https://huggingface.co/spaces/USERNAME/negotiarena"),
    ("6", "Run: git add -A && git commit -m 'Complete submission' && git push space main"),
    ("7", "Write mini-blog: huggingface.co/new-model → name: negotiarena-blog"),
    ("8", "Update README: replace YOUR_HF_USERNAME and YOUR_GITHUB_USERNAME"),
    ("9", "Upload NegotiArena_Training.ipynb to Kaggle and run it"),
]
for num, step in manual:
    print(f"  [{num}] {step}")

print("\n" + "="*55)
print("  FINAL FILE TREE (what it should look like)")
print("="*55)
print("""
negotiarena/
├── server/
│   ├── __init__.py  ✅
│   ├── app.py       ✅
│   └── index.html   ✅ (frontend)
├── training/
│   ├── __init__.py  ✅
│   ├── generate_sft_data.py  ✅
│   ├── train_cpu.py          ✅
│   ├── train_grpo.py         ✅
│   ├── plot_curves.py        ✅
│   └── prompts.py            ✅
├── evaluation/
│   ├── __init__.py  ✅
│   └── evaluator.py ✅
├── demo/
│   ├── __init__.py  ✅
│   └── app.py       ✅
├── tests/
│   ├── __init__.py  ✅
│   └── test_env.py  ✅
├── checkpoints/cpu_run/
│   ├── reward_curve.csv      ✅ (training evidence)
│   └── training_results.json ✅
├── negotiarena_env.py  ✅
├── inference.py        ✅
├── NegotiArena_Training.ipynb  ← DOWNLOAD FROM CLAUDE
├── openenv.yaml        ✅
├── Dockerfile          ✅
├── README.md           ✅ (updated with all links)
├── pyproject.toml      ✅
├── requirements.txt    ✅
├── reward_curves.html  ✅
├── uv.lock             ✅
└── .gitignore          ✅ (updated)

NOT in git (gitignored):
├── data/               (regenerated locally)
├── wandb/              (use W&B URL instead)
└── venv/
""")

print("Run this script again after fixes to verify.")
# ML Research Papers
A monorepo collecting my weekly **research-paper digests, from-scratch re-implementations, and reproducible notebooks/files**.

Each paper lives in its own folder and is independent of other folders:
```
paper-slug/
├─ notebook.ipynb     # Hands-on demo or replication
├─ lib/               # Subroutines & utilities
├─ data/              # Imported datasets (if needed)
├─ checkpoints/       # Saved model files and metadata (if needed)
├─ thumbnail.png      # Hero image used in blog & social posts
├─ requirements.txt   # External libraries used
└─ README.md          # 1-min summary + hands-on results
```

> **Goal:** enhance my skills & experience in the research side of AI and make cutting-edge ML papers less opaque.

## Papers in This Repo
| Date Published     | Paper |
|----------|--------------------------------------------------|
| 2017-06  | [Attention Is All You Need](./attention-is-all-you-need) |
<!-- | 2018-10  | [BERT: Pre-training of Deep Bidirectional Transformers](./bert) |
| 2022-05  | [Chain-of-Thought Prompting Elicits Reasoning in LLMs](./chain-of-thought) |
| 2022-11  | [QLoRA: Efficient Finetuning of Quantized LLMs](./qlora) |
| 2023-01  | [Hidden Technical Debt in Machine Learning Systems](./hidden-technical-debt) | -->


## Quick Start
### Copy Whole Repo
```bash
# 1. Clone the repo (shallow — saves bandwidth)
git clone --depth 1 https://github.com/Mahmh/ml-research-papers.git
cd ml-research-papers

# 2. Create & activate an environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the paper-specific dependencies
cd paper-slug
pip install -r requirements.txt

# 4. Launch your favorite notebook runner or editor
jupyter lab
```
### Copy Only One Folder
```bash
git clone --filter=blob:none --sparse https://github.com/Mahmh/ml-research-papers.git
cd ml-research-papers
git sparse-checkout set paper-slug   # replace `paper-slug` with the only folder you want to fetch
```
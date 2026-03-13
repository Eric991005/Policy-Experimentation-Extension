# Policy Experimentation Extension

Research code for the extension study on policy diffusion:

> **Central policy texts → Local government response texts → City-level outcomes**

This repository contains reproducible pipeline scripts used to build panels, run regressions, and generate visualizations.

## Repository scope

Included:
- Pipeline and analysis scripts (`scripts/*.py`)
- Dependency list (`requirements.txt`)

Not included:
- Raw data and processed large datasets
- Intermediate artifacts and logs
- Credentials or secrets

## Structure

```text
scripts/
  extension_pipeline.py         # end-to-end MVP pipeline (Tasks 1-8)
  enhance_macro_and_rerun.py    # macro merge + enhanced Model B
  add_event_models.py           # event-window + lag dynamic models
  build_city_level_maps.py      # city-level map generation
requirements.txt
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run scripts from your project workspace (default expected root on server is `/root/autodl-tmp/`).

## Notes

- This repo is code-only by design.
- If you need full reproducibility with data manifests/hashes, add a separate replication package.

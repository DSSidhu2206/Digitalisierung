# Digitalisierung

German bureaucratic-document extraction: upload a scanned form, get validated
structured fields. A production OCR pipeline (Surya / Apple Vision → layout
mapping → symbolic validation) reaching **90% exact-match** on real, degraded
documents — plus a rigorous R&D experiment testing whether a synthetic-trained,
OCR-free model could beat it.

- **Methodology** — [DATASET.md](DATASET.md): train-on-synthetic / eval-on-real.
- **Experiment & honest results** — [training/RESULTS.md](training/RESULTS.md):
  Donut **40%** vs OCR **90%**, with the analysis of why.
- **Audit & remediation** — [ANALYSIS.md](ANALYSIS.md),
  [REMEDIATION.md](REMEDIATION.md).

## Quick Start

```bash
cd ~/Documents/Digitalisierung/digitalisierung_complete
bash setup_mac.sh      # Install deps & download models
bash start.sh          # Build frontend if needed, then start API server
# Open http://localhost:8000/dashboard
```

## Files

```
Digitalisierung/
├── backend/            # FastAPI backend
│   ├── app/            # All Python modules
│   ├── config.py       # Settings
│   ├── requirements.txt
│   └── tests/          # Test suite
├── frontend/           # Next.js App Router dashboard
│   ├── src/            # Source code and styles
│   ├── out/            # Static export served by FastAPI
│   ├── package.json
│   └── next.config.ts
├── setup_mac.sh
├── start_server.sh
├── start_dev.sh
└── test.sh
```

## Frontend

```bash
cd frontend
npm install
npm run build
```

The dashboard is served by FastAPI at `/dashboard`. Do not open
`frontend/dashboard.html` directly; it is only a compatibility redirect.

## API

- `POST /api/v1/extract` — Upload image, get structured extraction
- `POST /api/v1/corrections` — Submit correction
- `GET /api/v1/extractions/{id}` — Get extraction
- `GET /api/v1/stats` — Pipeline stats
- `GET /api/v1/health` — System health

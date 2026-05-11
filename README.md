# Digitalisierung

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

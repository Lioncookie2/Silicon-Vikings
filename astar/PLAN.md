# Astar Island

## Oppsett

1. Logg inn på [app.ainm.no](https://app.ainm.no)
2. Hent JWT fra nettleser (cookie `access_token` eller Authorization header)
3. `export ACCESS_TOKEN='...'`

## Kommandoer (fra repo-roten)

```bash
export ACCESS_TOKEN=...
export PYTHONPATH=.

# Uniform baseline (alle seeds)
python -m astar.predict

# Kun se form uten å sende inn
python -m astar.predict --dry-run

# Utforsk med viewport-grid (bruker query-budsjett)
python -m astar.explore --max-queries 50
```

## Terrengklasser (6)

Tom, bosetning, havn, ruin, skog, fjell — sannsynligheter per celle må summere til 1; unngå eksakt 0.0.

# Neste 5–10 simulate-queries (strategi)

Målet er **maksimal læring om globale (skjulte) parametre** med få vinduer: samme parametre gjelder alle seeds, så du vil ha **variasjon på tvers av seeds** og **dekning av ulike geografier** (kyst vs innland, tett bosetning vs periferi).

## Anbefalt rekkefølge (7 kall etter første)

Anta at første kall var **seed 0**, viewport **(0,0) 15×15** — typisk kyst/nordvest.

| # | seed | viewport (ca.) | Hvorfor |
|---|------|----------------|---------|
| 2 | 1 | (0,0) 15×15 | Samme «hjørne» på annet kart → isolerer seed-spesifikt vs globalt mønster. |
| 3 | 2 | (0,0) 15×15 | Tredje seed — bekrefter om mønstre gjentas. |
| 4 | 0 | (20,12) 15×15 | Sentralt/innland — annen dynamikk enn kyst-hjørnet. |
| 5 | 3 | (12,20) 15×15 | Sørøst-kvadrant på ny seed — utforsk ny region. |
| 6 | 4 | (25,0) 15×15 | Nordøst kyststripe — port/krig/handel ofte kystnært. |
| 7 | 0 | (0,20) 15×15 | Sørvest — ofte annen topografi enn (0,0). |

Justér koordinater innenfor **0…25** for 40×40 med **15×15**-vindu (maks start (25,25)).

## Prinsipp

1. **Minst én query per seed** du ennå ikke har sett (her: 1–4 hvis du bare har seed 0).
2. **To ulike regioner på samme seed** (f.eks. hjørne + sentrum) for å kalibrere baseline mot lokal stokastikk.
3. Unngå **kun** overlappende vinduer samme sted — **stride** (f.eks. 12) eller hopp til andre hjørner.

## Bruk av observasjoner

- Tell **ruin / settlement / port**-andel i viewport (`astar.analyze_explore`) — hvis ruin er høy overalt, kan du **heve P(ruin)** i baseline nær marginal land i neste iterasjon (empirisk justering).
- Sammenlign **samme viewport-koordinater** på seed 0 vs 1: stor forskjell → kartet; lik «atferd» → hint om globale parametre (aggressivitet, handel, osv.).

Dette erstatter ikke en full simulator; det **styrer** neste queries og senere manuelle vekter i prioren.

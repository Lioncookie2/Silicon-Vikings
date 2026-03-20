# Git-samarbeid (Silicon Vikings)

## Daglig arbeidsflyt

1. **Hent siste fra GitHub før du starter**
   ```bash
   git pull origin main
   ```

2. **Jobb i korte commits** – én logisk endring om gangen (f.eks. «tripletex: fix health check»).

3. **Push ofte** så Sebastian ser endringer raskt:
   ```bash
   git add -p   # eller git add <filer>
   git commit -m "kort beskrivelse"
   git push origin main
   ```

4. Hvis `git push` feiler fordi `origin/main` har nye commits:
   ```bash
   git pull --rebase origin main
   git push origin main
   ```
   (`rebase` gir renere historikk enn merge-commit for små lag.)

## Unngå krasj / merge-konflikter

| Tips | Hvorfor |
|------|---------|
| **Pull før du endrer** | Du bygger på nyeste `main`. |
| **Del ansvar** | F.eks. du: `tripletex/`, Sebastian: `norgesgruppen/` når mulig. |
| **Ikke commit store genererte filer** | `data/yolo/`, `runs/`, `best.pt` er ignorert – kjør `prepare_dataset` lokalt. |
| **Kommuniser** i chat: «jeg jobber i `tripletex/agent.py` nå». |

## Grener (valgfritt)

For større eksperimenter uten å blokkere `main`:

```bash
git checkout -b feature/tripletex-handlers
# ... jobb ...
git push -u origin feature/tripletex-handlers
# Pull request på GitHub → merge til main
```

## Ved konflikt

```bash
git pull origin main
# Git markerer filer – åpne dem, fjern <<<<<< / ====== / >>>>>>, velg riktig kode
git add <filer>
git commit -m "Merge: løs konflikt i ..."
git push
```

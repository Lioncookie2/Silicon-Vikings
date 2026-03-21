# Baseline ep100

Denne mappen inneholder baseline-modellen etter 100 epoker trening.

## Innhold
- `model.onnx`: ONNX-eksport for robust inferens i submission-miljo.
- `results.csv`: Trenings- og valideringsmetrikker per epoke.
- `args.yaml`: Treningsargumenter brukt i kjøringen.

Merk: `best.pt` er ikke inkludert her fordi filen overstiger GitHub sin filgrense.

## Kilde
Filene er kopiert fra:
- `submission/model.onnx`
- `runs/ngd_yolo/baseline/results.csv`
- `runs/ngd_yolo/baseline/args.yaml`

Lokal `best.pt` finnes fortsatt i:
- `runs/ngd_yolo/baseline/weights/best.pt`

## Bruk
For submission brukes `model.onnx` sammen med `submission/run.py`.

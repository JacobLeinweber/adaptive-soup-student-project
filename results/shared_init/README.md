# Stufe 2 und 3 mit gemeinsamem Startpunkt (T11)

Alle Zweige starten vom **mitgelieferten Seed-52-Source-only-Checkpoint** des jeweiligen Transfers
(`models/<Familie>/<Transfer>/model_last.pt`), für alle Seeds und für Methode und Kontrolle gleich
(Vorgabe des Betreuers aus Meeting 3). Der Seed steuert nur noch die Zufälligkeit des Adaptionstrainings:
Reihenfolge der Source- und Zielbatches, Dropout-Masken und die Initialisierung des DANN-Diskriminators.
Alles andere wie Stufe 2: AdaTime-Gewichte, 30 Epochen, Adam (LR 3e-4, Weight Decay 1e-4), Batchgröße 32,
Modell nach der letzten Epoche.

## Inhalt

| Ordner | Inhalt |
|---|---|
| `<Variante>/seed_<Seed>/<Transfer>/` | Läufe der Seeds 100, 445, 545, 881 (4 × 12 Transfers × 6 Varianten = 288), dieselben Dateien wie in Stufe 2 |
| `soup/<soup_methods\|soup_controls>/<bn_avg\|bn_target>/seed_<Seed>/<Transfer>/` | uniforme Soups, alle fünf Seeds |
| `ensemble/<methods\|controls>/seed_<Seed>/<Transfer>/` | Output-Ensembles, alle fünf Seeds |
| `source_val_predictions/<Zweig>/seed_<Seed>/<Transfer>/` | Source-Val-Vorhersagen der Zweige für SRC, alle fünf Seeds |
| `tables/` | `T3_*.csv` (Stufe 2) und `T8_*.csv` (Stufe 3) für den gemeinsamen Startpunkt, gleiche Namen und Spalten wie in `results/tables/` |

Ein Lauf enthält `model_last.pt` (nicht in Git), bei DANN `discriminator_last.pt` (nicht in Git), `train_log.csv`,
`predictions/<Split>.npz` (Source-Test, Target-Train, Target-Val, Target-Test) und `config.json` mit
`"init": "shared"`, Init-Checkpoint samt SHA-256, Commit, Umgebung und Laufzeit.

| Methode | Variante | Kontrolle (λ = 0, Zielbatches laufen mit, Source-Gewicht der Methode) |
|---|---|---|
| Deep CORAL | `deep_coral` | `lambda0_target_pass` |
| DANN | `dann` | `dann_lambda0_target_pass` |
| DSAN | `dsan` | `dsan_lambda0_target_pass` |

**Seed 52 liegt bei den Läufen nicht hier:** Mit gemeinsamem Startpunkt sind die Seed-52-Läufe genau die
Stufe-2-Läufe in `results/<Variante>/seed_52/` und werden von dort gelesen. Soups, Ensembles und
Source-Val-Vorhersagen wurden für alle fünf Seeds neu berechnet, auch für Seed 52.

Vergleich mit dem bisherigen Startpunkt (jeder Seed vom eigenen Source-only-Modell): `results/tables/T11_*.csv`,
Abbildungen 13–15 und Abschnitt „Shared starting point“ in `project_docs/results.md`.

## Herkunft

- Läufe: `run_shared_init.py` (ruft `run_deep_coral.py`, `run_dann.py`, `run_dsan.py` mit `--init shared` auf),
  Commit `63d799f`, Nacht 22./23.11.2026, 03:04–10:03 Uhr, kein Lauf gescheitert.
- Stufe 3: `run_soup.py --all --init shared`, Commit `0130ba3`; Tabellen: `make_dsan_tables.py --init shared`,
  `make_soup_tables.py --init shared`; Vergleich: `make_shared_init_tables.py`, `make_shared_init_figures.py` (`f7d822f`).
- Gerät: Apple M1, MPS, macOS 27.2.1. Umgebung: Python 3.13.15, PyTorch 2.14.1, numpy 2.4.6. Die Läufe der Nacht
  liefen in der temporären Umgebung des T10-Tests, Stufe 3 und die Tabellen in `~/.venvs/adaptive-soup`
  (dieselben Paketversionen wie `.venv`).

## Prüfungen

- DANN, MHEALTH → PAMAP2, Seed 52 mit `--init shared` neu gerechnet: **bitgleich** mit
  `results/dann/seed_52/MHEALTH_to_PAMAP2/` (Gewichte, Diskriminator, Vorhersagen auf vier Splits, Epochen-Log).
- DANN, MHEALTH → PAMAP2, Seed 100 in `~/.venvs/adaptive-soup` neu gerechnet: bitgleich mit dem Lauf der Nacht.
- Alle 288 Läufe: `init = shared`, Init-Checkpoint mit SHA-256 des Seed-52-Checkpoints, Commit `63d799f`.
- Stufe 3: Alle Zutaten einer Soup und alle Seeds eines Transfers starten vom selben Checkpoint; neu berechnete
  Vorhersagen der Zweige = gespeicherte (Übereinstimmung 1,0). Die Seed-52-Soups und -Ensembles sind bitgleich mit
  Stufe 3 (48 von 48 Gewichtssätzen, alle Vorhersagen und Logits).
- Auf MPS ist Training nicht garantiert bitgenau reproduzierbar; die Prüfungen zeigen, dass es hier so war.

Ziel-Labels werden beim Training und bei der Soup-Bildung nicht gelesen, nur in den Tabellenskripten zur Auswertung.

# Source-only-Baseline: PAMAP2 → RealWorld

Das mitgelieferte Modell wurde auf der CPU neu ausgewertet; kein erneutes Training.

| Split | Fenster | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| source_train | 3646 | 99.97% | 99.97% | 99.97% |
| source_val | 849 | 65.96% | 60.79% | 64.09% |
| source_test | 1426 | 96.63% | 96.40% | 96.23% |
| target_train | 9685 | 36.70% | 34.87% | 36.03% |
| target_val | 2084 | 45.11% | 39.79% | 42.87% |
| target_test | 2894 | 44.75% | 38.05% | 42.71% |

Die Ziel-Labels wurden ausschließlich zur Bewertung verwendet. Es gab keine Modelloptimierung auf Ziel-Fenstern; die mitgelieferte domänenspezifische Normalisierung wurde beibehalten.

Die JSON-Datei enthält Einzelmetriken pro Aktivität und Person sowie den Abgleich mit den gespeicherten Vorhersagen und Metriken des Betreuers.

Checkpoint SHA-256: `4d79188fca981a9fe665584f98d12a7672c89da9c8faf08a4db00622cf59c76c`
Repository-Commit: `64f61bd02500f5400991e859c56dc1020a5d1160`

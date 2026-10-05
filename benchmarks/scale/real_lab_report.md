639 tools, 19 servers, 156 tool requests (98 PubMed/Zotero, 58 other servers)

| strategy | recall | PubMed+Zotero | other servers | EN | 中文 | right server present | tools sent | ms (median) |
|---|---|---|---|---|---|---|---|---|
| bm25 top-10 | 61.5% | 51.0% | 79.3% | 75.7% | 22.0% | 77.6% | 10.0 | 0 |
| emb top-10 | 71.2% | 62.2% | 86.2% | 79.1% | 48.8% | 87.2% | 10.0 | 5 |
| hybrid top-5 | 69.2% | 61.2% | 82.8% | 79.1% | 41.5% | 84.6% | 5.0 | 6 |
| hybrid top-10 | 75.0% | 69.4% | 84.5% | 84.3% | 48.8% | 90.4% | 10.0 | 6 |
| hybrid top-20 | 79.5% | 75.5% | 86.2% | 88.7% | 53.7% | 91.7% | 20.0 | 6 |
| laya flat over 19 servers (top-3) -> hybrid top-10 | 64.1% | 55.1% | 79.3% | 72.2% | 41.5% | 75.0% | 10.0 | 1016 |
| emb gate 3 servers -> hybrid top-10 | 60.3% | 51.0% | 75.9% | 66.1% | 43.9% | 68.6% | 10.0 | 7 |
| hybrid top-12 -> laya keep 5 | 73.7% | 68.4% | 82.8% | 84.3% | 43.9% | 88.5% | 5.0 | 1492 |
| hybrid top-12 -> laya keep 5 | hybrid top-5 | 75.6% | 70.4% | 84.5% | 85.2% | 48.8% | 88.5% | 6.5 | 1374 |
| hybrid top-24 -> laya (2x12) keep 8 | 76.9% | 71.4% | 86.2% | 87.8% | 46.3% | 89.1% | 8.0 | 4848 |
| hybrid top-24 -> laya keep 5 | hybrid top-5 | 77.6% | 71.4% | 87.9% | 87.0% | 51.2% | 89.1% | 8.0 | 4762 |

# LLM-Agents

Dieses Projekt implementiert kognitiv inspirierte LLM-Systeme zur Bewertung agentischer Fähigkeiten anhand des EWOK-Benchmarks. Wir haben das gesamte Dataset mit **mistralai/mistral-7b-instruct** getestet und eine Genauigkeit von **68 %** erreicht. Der komplette Durchlauf kostete ca. **0,01 USD** und dauerte etwa **2 Minuten**, dank paralleler API-Aufrufe. Ohne Parallelisierung würde das etwa 20× länger dauern. Für Tests könnt ihr `--max_items` auf z. B. 20 setzen, um schneller Ergebnisse zu erhalten.

Alternative gratis Option: **mistralai/mistral-7b-instruct:free** – funktioniert auch, ist jedoch deutlich langsamer (~2 Stunden für das vollständige Dataset), aber ausreichend für kleine Tests.

---

## 🔧 Installation

1. Repository klonen:
   ```bash
   git clone https://github.com/juverse/LLM-Agents.git
   cd LLM-Agents
   ```

2. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```
3. HuggingFace token einrichten:

   ```bash
   huggingface-cli login
   ```
4. OpenRouter-API-Key und Modellnamen in `main.py` anpassen (Variablen `OPENROUTER_API_KEY` und `MODEL_NAME`).

---

## 🚀 Nutzung

```bash
python main.py \
  --max_items 1-9999 \
  --model mistralai/mistral-7b-instruct \
  --main_prompt STUDY \
  --sub_prompts reasoning,working_memory,theory_of_mind
```

* `--max_items`: Anzahl der Datensätze zum Testen (z. B. 20 für schnelle Durchläufe)
* `--model`: Modellkennung (z. B. `mistralai/mistral-7b-instruct` oder `...:free`)
* `--main_prompt`: Name der Haupt-Prompt-Vorlage (z. B. `study`)
* `--sub_prompts`: Komma-getrennte Submodule (deren .txt-Dateien im Ordner `prompts/sub_prompts/` liegen)

Ergebnisse werden gespeichert unter:

```
results/ISO_DATETIME_ewok_eval.csv
```

Enthalten ist u. a. die Spalte `arguments` mit den verwendeten Parametern.

### 🧠 Analyse

```bash
python analyze_results.py results/*.csv
```

Führt Accuracy-Metriken und Vergleiche über System-Varianten hinweg aus.

---

## 🗂 Prompt-Vorlagen

* `prompts/sub_prompts/*.txt`: Diese Dateien enthalten `{context}`-Platzhalter und werden pro Kontext aufgerufen.
* `prompts/main_prompt/*.txt`: Hier werden Ergebnisse der Submodule referenziert (z. B. `{reasoning_context}`, `{working_memory_context}`, `{target}`).

Beispiele:

* `study.txt`: exakt der Prompt, wie er im Paper verwendet wurde.
* Schaut euch die CSV in `results` an, um zu sehen, wie die finalen Prompts aussehen (Spalten: `context a, context b, answer a, answer b, prompt_1_response, prompt_2_response`, etc.).

---

## ⚙️ Methoden

* Alle Benchmarks (außer `localquant`, `logprob`, `memory-assisted`) wurden via OpenRouter API abgefragt.
* **localquant**: Lokale Auswertung mit vLLM + quantisiertem Modell `neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16`. Leistung liegt \~1–2 % unter dem Originalmodell.
* **logprob**: Log-Wahrscheinlichkeiten vom quantisierten Modell, bewertet die Antwortoptionen.
* **memory-assisted**: Modell erhält bis zu 3 relevante vergangene Kontexte (Speicherung via einfache Relevanzbewertung), dann erneute Anfrage via OpenRouter.

---

## 📊 Ergebnisse

| Setup                | Modell                                               | Overall | Agent‑Properties | Social‑Interactions | Social‑Properties |
| -------------------- | ---------------------------------------------------- | ------: | ---------------: | ------------------: | ----------------: |
| study\_mistral       | mistralai/mistral-7b-instruct                        |   0.724 |            0.675 |               0.826 |             0.927 |
| decision\_making     | mist...-7b-instruct                                  |   0.726 |            0.688 |               0.794 |             0.892 |
| working\_memory      | mist...-7b-instruct                                  |   0.725 |            0.701 |               0.820 |             0.781 |
| deepseek             | deepseek/deepseek-chat-v3-0324                       |   0.850 |            0.803 |               0.994 |             1.000 |
| localquant           | local-vllm                                           |   0.722 |            0.679 |               0.806 |             0.903 |
| cot\_updated         | mist...-7b-instruct                                  |   0.664 |            0.602 |               0.834 |             0.876 |
| combination\_tom\_wm | mist...-7b-instruct                                  |   0.748 |            0.709 |               0.877 |             0.865 |
| bdi                  | mist...-7b-instruct                                  |   0.676 |            0.651 |               0.780 |             0.727 |
| tom                  | mist...-7b-instruct                                  |   0.717 |            0.678 |               0.843 |             0.832 |
| combination\_all     | mist...-7b-instruct                                  |   0.754 |            0.717 |               0.857 |             0.884 |
| combination\_dm\_wm  | mist...-7b-instruct                                  |   0.753 |            0.716 |               0.860 |             0.876 |
| nemo                 | mistralai/mistral-nemo                               |   0.693 |            0.631 |               0.863 |             0.905 |
| memory-assisted      | mist...-7b-instruct                                  |   0.772 |            0.733 |               0.894 |             0.895 |
| combination\_tom\_dm | mist...-7b-instruct                                  |   0.735 |            0.696 |               0.854 |             0.859 |
| logprob              | neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16 |   0.701 |            0.679 |               0.774 |             0.759 |
| logic                | mist...-7b-instruct                                  |   0.648 |            0.613 |               0.737 |             0.773 |
| llama                | meta-llama/llama-3.1-8b-instruct                     |   0.597 |            0.571 |               0.594 |             0.759 |
| cot                  | mist...-7b-instruct                                  |   0.543 |            0.533 |               0.551 |             0.592 |

---

## ✅ Fazit

* Volle Modularität und Kombinationen (bes. memory-assisted) erzielen Top-Performance (bis zu 0.772).
* Reine Chain‑of‑Thought‑Prompts (`cot`) performen deutlich schlechter (\~0.543).
* Lokales quantisiertes Modell ist nah am Cloud-Modell (< 2 %-Differenz).
* Gratis-Modell funktioniert, ist aber langsamer – ideal für Tests.

---

## 📚 Zitation
Bitte zitiert entsprechende Papers (EWOK, CoALA etc.) sowie dieses Projekt, wenn ihr es verwendet.
---

## 🧑‍🤝‍🧑 Autoren & Kontakt

* Moritz Lönker
* Julia Lansche
* Marc Baumholz 
* Bei Fragen oder Anmerkungen: einfach melden!

---

> *Hinweis*: Dieses README ist optimiert für Reproduzierbarkeit und Nachvollziehbarkeit – falls etwas unklar ist, gebt Bescheid 😊.

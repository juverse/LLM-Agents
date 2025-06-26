# LLM-Agents

Habe das ganze Datenset mit Mistral-7b-instruct getestet. 68% accuracy, Kosten für das gesamte Datenset sind 1 cent. Dauer ca. 2 Minuten, habe mehrere parallel Aufrufe, sonst würde das 20x so lange dauern, aber falls ihr Fehler bekommt könnt ihr das abstellen. Ihr könnt auch bei max_items statt 9999, sowas wie 20 eingeben, dann ist es schneller zum testen. 

Es geht auch gratis mit mistralai/mistral-7b-instruct:free, ist aber viel langsamer ca. 2 Std. für das gesamte dataset, aber für kleinere tests sollte das reichen.


# Installation

1. Die libraries aus requirements.txt installieren
2. `huggingface-cli login` - der anleitung folgen, also token von huggingface erstellen und reinposten
3. Openrouter API Key und model name (wenn ihr es ändern wollt) im code angeben
4. `python main.py [--max_items 1-9999] [--model openrouter_model_code] [--main_prompt NAME] [--sub_prompts NAME1,NAME2]`

Die Ergebnisse werden als `results/ISO_DATETIME_ewok_eval.csv` gespeichert. Jede Zeile enthält zusätzlich eine Spalte `arguments` mit den beim Aufruf verwendeten Parametern.

Analyze results:
`python analyze_results.py [csv1] [optional:csv2, csv3, etc.]`


## Prompt templates
Prompt files live in the `prompts` folder.

- `prompts/sub_prompts/NAME.txt` should contain the placeholder `{context}` and is executed once for each context.
- `prompts/main_prompt/NAME.txt` combines the results and must reference `{context1}`, `{context2}`, `{target}` and each subprompt result as `{NAME_context1}`/`{NAME_context2}`.

Run with `python main.py --main_prompt NAME --sub_prompts NAME1,NAME2` to select your templates.

Guckt euch die Beispiele in den Ordnern an und eventuell das csv in results für wie die finalen prompts aussehen.
Relevant sind die columns context a, context b, answer a, answer bm prompt 1, prompt_2.

Das prompt template "study.txt" ist exakt der prompt, der in im paper verwendet wurde.
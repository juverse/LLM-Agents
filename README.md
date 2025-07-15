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


# Prompt templates
Prompt files live in the `prompts` folder.

- `prompts/sub_prompts/NAME.txt` should contain the placeholder `{context}` and is executed once for each context.
- `prompts/main_prompt/NAME.txt` combines the results and must reference `{context1}`, `{context2}`, `{target}` and each subprompt result as `{NAME_context1}`/`{NAME_context2}`.

Run with `python main.py --main_prompt NAME --sub_prompts NAME1,NAME2` to select your templates.

Guckt euch die Beispiele in den Ordnern an und eventuell das csv in results für wie die finalen prompts aussehen.
Relevant sind die columns context a, context b, answer a, answer bm prompt 1, prompt_2.

Das prompt template "study.txt" ist exakt der prompt, der in im paper verwendet wurde.

# Methods
With the excpetion of localquant, logprob, and memory-assisted all benchmarks were performed by querying the [Openrouter API](https://openrouter.ai/). Should be easy to replicate using main.py.
localquant is based on a local [vllm](https://github.com/vllm-project/vllm) cuda install, using the quantized [neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16](https://huggingface.co/RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w8a16) model. The quantized model is expected to perform within approximately 1–2% of the full model on most benchmarks. Logprobs were obtained from the same model, but on the logprob task in comparison to the choice task. The memory-assisted benchmark is not directly comparable, as in this case the model has access to more possible information. Past responses are saved and up to three memories are injected into the prompt based on a simple relevance scoring system, while the inference is done again using Openrouter.

# Results
| name               | model                                                |   overall |   agent-properties |   social-interactions |   social-properties |
|:-------------------|:-----------------------------------------------------|----------:|-------------------:|----------------------:|--------------------:|
| study_mistral      | mistralai/mistral-7b-instruct                        |     0.724 |              0.675 |                 0.826 |               0.927 |
| decision_making    | mistralai/mistral-7b-instruct                        |     0.726 |              0.688 |                 0.794 |               0.892 |
| working_memory     | mistralai/mistral-7b-instruct                        |     0.725 |              0.701 |                 0.82  |               0.781 |
| deepseek           | deepseek/deepseek-chat-v3-0324                       |     0.85  |              0.803 |                 0.994 |               1     |
| localquant         | local-vllm                                           |     0.722 |              0.679 |                 0.806 |               0.903 |
| cot_updated        | mistralai/mistral-7b-instruct                        |     0.664 |              0.602 |                 0.834 |               0.876 |
| combination_tom_wm | mistralai/mistral-7b-instruct                        |     0.748 |              0.709 |                 0.877 |               0.865 |
| bdi                | mistralai/mistral-7b-instruct                        |     0.676 |              0.651 |                 0.78  |               0.727 |
| tom                | mistralai/mistral-7b-instruct                        |     0.717 |              0.678 |                 0.843 |               0.832 |
| combination_all    | mistralai/mistral-7b-instruct                        |     0.754 |              0.717 |                 0.857 |               0.884 |
| combination_dm_wm  | mistralai/mistral-7b-instruct                        |     0.753 |              0.716 |                 0.86  |               0.876 |
| nemo               | mistralai/mistral-nemo                               |     0.693 |              0.631 |                 0.863 |               0.905 |
| memory-assisted    | mistralai/mistral-7b-instruct                        |     0.772 |              0.733 |                 0.894 |               0.895 |
| combination_tom_dm | mistralai/mistral-7b-instruct                        |     0.735 |              0.696 |                 0.854 |               0.859 |
| logprob            | neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16 |     0.701 |              0.679 |                 0.774 |               0.759 |
| logic              | mistralai/mistral-7b-instruct                        |     0.648 |              0.613 |                 0.737 |               0.773 |
| llama              | meta-llama/llama-3.1-8b-instruct                     |     0.597 |              0.571 |                 0.594 |               0.759 |
| cot                | mistralai/mistral-7b-instruct                        |     0.543 |              0.533 |                 0.551 |               0.592 |


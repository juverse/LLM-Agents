# LLM-Agents

Habe das ganze Datenset mit Mistral-7b-instruct getestet. 68% accuracy, Kosten für das gesamte Datenset sind 1 cent. Dauer ca. 2 Minuten, habe mehrere parallel Aufrufe, sonst würde das 20x so lange dauern, aber falls ihr Fehler bekommt könnt ihr das abstellen. Ihr könnt auch bei max_items statt 9999, sowas wie 20 eingeben, dann ist es schneller zum testen. 

Es geht auch gratis mit mistralai/mistral-7b-instruct:free, ist aber viel langsamer ca. 2 Std. für das gesamte dataset, aber für kleinere tests sollte das reichen.


# Installation

1. Die libraries aus requirements.txt installieren
2. `huggingface-cli login` - der anleitung folgen, also token von huggingface erstellen und reinposten
3. Openrouter API Key und model name (wenn ihr es ändern wollt) im code angeben
4. `python test_openrouter.py`



Die vollen Ergebnisse von einem test mit mistral 7b sind in der csv.

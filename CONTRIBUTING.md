# Contributing and support

Issues may report defects in the released code or ask about the documented workflow with the project dataset schema. Include the command, release version and a minimal description of the problem. Remove private paths and credentials. Do not attach research data, trained checkpoints or paper-result tables. Responses have no guaranteed timeline.

Adapting other datasets, channel schemas or scientific methods is outside the documented support scope. Users may modify the code under MIT and are responsible for checking their changes.

For a code contribution, describe the behavior changed, use a branch and pull request, and add a focused regression test where appropriate. Changes to scientific settings or formulas must be explicit. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_docs.py
```

Translation corrections follow docs/TRANSLATIONS.md. Contributions to project code are provided under MIT. Do not commit environments, run outputs, measurements, model files or private audit records.

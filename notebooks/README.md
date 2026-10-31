# Optional notebook

The supported beginner entry point is the Windows command-line workflow in docs/translations/README.vi.md. The quickstart notebook only calls the same CLI and contains no separate scientific implementation. Original research notebooks remain outside this release.

Notebook execution additionally requires a notebook host and ipykernel; these optional tools are not installed by the core setup. If already using VS Code notebooks, install the kernel in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Select that project environment as the kernel, then read every cell before running. The run cell starts real training. Do not execute it merely to preview the file.

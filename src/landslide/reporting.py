"""Export human-readable tables and plots from recorded results, never paper constants."""
import json
from pathlib import Path

from .io import write_csv


def export_tables(evaluation_dir, summary, size_rows):
    """One-row main/boundary tables and three size rows for the current model."""
    out = Path(evaluation_dir)
    main = dict(model=summary['model'], threshold=summary['threshold'],
                iou=summary['overlap_conditional']['iou']['mean'],
                dice=summary['overlap_conditional']['dice']['mean'],
                n_reference_positive=summary['n_reference_positive'],
                n_overlap_eligible=summary['n_overlap_eligible'],
                n_empty_predictions=summary['n_empty_predictions_on_positive'],
                empty_prediction_rate=summary['empty_prediction_rate'])
    main.update({k: summary['pixel_pooled'][k] for k in ['mcc', 'precision', 'recall', 'f1', 'f2', 'fnr', 'fdr', 'accuracy']})
    main.update({k: summary[k] for k in ['object_precision', 'object_recall', 'object_f1']})
    write_csv(out/'Table_MainResults.csv', [main])
    write_csv(out/'Table_SizeRecall.csv', [dict(model=summary['model'], **r) for r in size_rows])
    boundary = dict(model=summary['model'], **{k: v['mean'] for k, v in summary['boundary_positive'].items()},
                    undefined_rate=summary['boundary_undefined_rate'])
    boundary.update({f'{k}_n_defined': v['n_defined'] for k, v in summary['boundary_positive'].items()})
    boundary['n_reference_positive'] = summary['n_reference_positive']
    write_csv(out/'Table_Boundary.csv', [boundary])


def plot_evaluation(out, summary):
    """Use a non-interactive backend so Windows scripts never wait for a plot window."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = summary['confusion']
    matrix = [[c['tn'], c['fp']], [c['fn'], c['tp']]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i][j]:,}', ha='center', va='center',
                    color='white' if matrix[i][j] > max(map(max, matrix))/2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Background', 'Landslide'],
           yticklabels=['Background', 'Landslide'], xlabel='Predicted', ylabel='Reference',
           title='Pooled TEST confusion counts')
    fig.tight_layout(); fig.savefig(Path(out)/'confusion_matrix.png', dpi=160); plt.close(fig)


def plot_training(out, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    for key in ['train_loss', 'val_loss']:
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].set(xlabel='Epoch', ylabel='Loss'); axes[0].legend()
    axes[1].plot(epochs, [r['val_pooled_iou'] for r in history])
    axes[1].set(xlabel='Epoch', ylabel='VAL pooled IoU')
    fig.tight_layout(); fig.savefig(Path(out)/'training_curves.png', dpi=160); plt.close(fig)


def summarize(runs, out):
    """Consolidate completed runs into four tables, retaining missing-stage status."""
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import csv
    groups = {k: [] for k in ['MainResults', 'SizeRecall', 'Boundary', 'Efficiency']}
    for path in sorted(runs.glob('*/evaluation/test_summary.json')):
        for key in ['MainResults', 'SizeRecall', 'Boundary']:
            with (path.parent/f'Table_{key}.csv').open(encoding='utf-8', newline='') as f:
                groups[key].extend(csv.DictReader(f))
        benchmark = path.parent.parent/'benchmark.json'
        if benchmark.exists():
            b = json.loads(benchmark.read_text(encoding='utf-8'))
            groups['Efficiency'].append({k: b[k] for k in ['model', 'params_m', 'gflops', 'latency_mean_ms', 'fps', 'precision']})
    if not groups['MainResults']:
        raise ValueError('No completed runs found. Expected RUNS/model/evaluation/test_summary.json.')
    for key, rows in groups.items():
        if rows:
            write_csv(out/f'Table_{key}.csv', rows)

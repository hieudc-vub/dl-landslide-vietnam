"""Report capabilities without silently changing the scientific configuration."""
import importlib.metadata
import platform
import sys


def check_environment():
    print(f'Python: {platform.python_version()}')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12. The reference environment uses Python 3.12.8.')
    for name in ['torch', 'torchvision', 'numpy', 'scipy', 'segmentation-models-pytorch', 'timm', 'rasterio', 'matplotlib', 'thop']:
        try:
            print(f'{name}: {importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            raise ModuleNotFoundError(f'Missing dependency: {name}') from None
    import torch
    print(f'PyTorch CUDA build: {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB')
        print('CUDA is available. Training uses FP16; calibration/evaluation use FP32.')
    else:
        print('CUDA is unavailable. CPU runs use FP32 and are not equivalent performance measurements.')
    print('Environment check complete. Run the synthetic smoke test before real training.')

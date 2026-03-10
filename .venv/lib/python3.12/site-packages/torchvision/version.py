__version__ = '0.25.0'
git_version = '82df5f599578b383987510836bb05ea97dcc9669'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()

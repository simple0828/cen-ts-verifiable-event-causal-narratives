import sys

import pytest

from cen_ts.paths import gpt2_path

# Importing the upstream comparison must not create bytecode in third_party.
sys.dont_write_bytecode = True


@pytest.fixture
def local_gpt2():
    path = gpt2_path()
    if not path.exists():
        pytest.skip("Local GPT-2 unavailable; set CEN_TS_GPT2_PATH or provide models/gpt2")
    return str(path)


@pytest.fixture
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: GPU integration check was not run")
    return "cuda:0"

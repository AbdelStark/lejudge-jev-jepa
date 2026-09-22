import os

import pytest

os.environ.setdefault("LEJUDGE_MODE", "offline")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="session")
def vocab():
    from lejudge.vocab import load_vocab

    return load_vocab("pusht@1")


@pytest.fixture(scope="session")
def library():
    from lejudge.constraints import load_library

    return load_library("pusht")

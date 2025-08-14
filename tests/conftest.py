# tests/conftest.py
import os
import sys

# Headless SDL so tests don't open a window
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Ensure project root is importable (so NN.* imports work when running from repo root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pygame as pg
import pytest

@pytest.fixture(scope="session", autouse=True)
def _pygame_session():
    pg.init()
    yield
    pg.quit()

@pytest.fixture
def screen():
    # Plain Surface is fine for draw/blit tests (no need for display mode)
    return pg.Surface((800, 600), pg.SRCALPHA)

@pytest.fixture
def node_factory():
    from NN.node import Node
    def make(pos=(100, 100), r=12, color=(0, 255, 0), **kwargs):
        # kwargs lets you pass layer=..., text=..., etc.
        return Node(pos=pos, r=r, color=color, **kwargs)
    return make

@pytest.fixture
def edge_factory():
    from NN.edges import Edge
    def make(a, b, **kwargs):
        return Edge(a, b, **kwargs)
    return make

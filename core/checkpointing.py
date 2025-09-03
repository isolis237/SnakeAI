from __future__ import annotations
import json, os
from typing import Protocol, Dict, Any, Optional

class Checkpointable(Protocol):
    """Objects that can round-trip their state as pure-Python/JSON-serializable dicts."""
    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]) -> None: ...

class CheckpointManager:
    """Saves/loads a named bundle of components. Each component must be Checkpointable."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def save(self, tag: str, components: Dict[str, Checkpointable]) -> None:
        path = os.path.join(self.root_dir, f"{tag}.ckpt.json")
        bundle = {name: comp.get_state() for name, comp in components.items()}
        os.makedirs(self.root_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(bundle, f)

    def load(self, tag: str, components: Dict[str, Checkpointable]) -> None:
        path = os.path.join(self.root_dir, f"{tag}.ckpt.json")
        with open(path, "r") as f:
            bundle = json.load(f)
        for name, comp in components.items():
            if name in bundle:
                comp.set_state(bundle[name])

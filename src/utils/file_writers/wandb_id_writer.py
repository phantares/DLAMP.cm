from pathlib import Path
import json


def write_wandb_id(run_id: str, id_file: Path) -> None:
    id_file.parent.mkdir(parents=True, exist_ok=True)

    with open(id_file, "w") as f:
        json.dump({"run_id": run_id}, f)


def load_wandb_id(id_file: Path) -> str | None:
    if id_file.exists():
        with open(id_file) as f:
            return json.load(f).get("run_id")
    return None

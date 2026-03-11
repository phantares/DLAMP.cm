from pathlib import Path


def find_best_model(exp_name):
    folder = Path("checkpoints", exp_name)
    files = list(folder.glob("*.ckpt"))

    if not files:
        raise FileNotFoundError("No checkpoints found!")

    best_loss = float("inf")
    best_model = ""

    if not files:
        print("No .ckpt files found")
        return

    for f in files:
        try:
            loss = float((f.stem.split("-")[-1]).split("=")[-1])

            if loss < best_loss:
                best_loss = loss
                best_model = f

        except (ValueError, IndexError):
            print(f"Skipping file with unexpected format: {f.name}")
            continue

    print(f"Best model: {best_model.name}")
    return best_model

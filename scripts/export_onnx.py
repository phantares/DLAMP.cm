from pathlib import Path
import argparse
from model import DirectDownscaling


def export_best_model_onnx(exp_name, device="cpu"):
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

    model = DirectDownscaling.load_from_checkpoint(best_model, crop_number=2)
    if device == "cpu":
        model.cpu()
    model.eval()

    input_keys = list(model.example_input_array.keys())
    dynamic_axes_config = {
        "single": {0: "batch_size"},
        "upper": {0: "batch_size"},
        "time": {0: "batch_size"},
        "noise": {0: "batch_size", 1: "crop_number"},
        "sigma": {0: "batch_size", 1: "crop_number"},
        "column_top": {0: "batch_size", 1: "crop_number"},
        "column_left": {0: "batch_size", 1: "crop_number"},
        "output": {0: "batch_size", 1: "crop_number"},
    }

    model.to_onnx(
        folder / f"{exp_name}.onnx",
        export_params=True,
        input_names=input_keys,
        output_names=["output"],
        dynamic_axes=dynamic_axes_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Enter device.",
    )
    args = parser.parse_args()

    export_best_model_onnx(args.exp, args.device)

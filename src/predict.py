import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import onnxruntime as ort
import torch
from dotenv import dotenv_values
from hydra import compose, initialize
from torchvision.transforms.v2 import CenterCrop, Compose, Resize
from utils import write_h5_file


def main(
    exp_name,
    input_dir,
    source_name: str,
    target_time: datetime | None = None,
    initial_time: datetime | None = None,
    batch_size: int = 1,
):
    env = dotenv_values(".env")

    with initialize(config_path=f"../experiments/{exp_name}/.hydra", version_base=None):
        cfg = compose(config_name="config")

    dtype = getattr(torch, cfg.dtype, torch.float32)

    grid_low = cfg.dataset.res.global_grid
    grid_high = int(
        grid_low * cfg.dataset.res.resolution_input / cfg.dataset.res.resolution_target
    )
    transform_grid = Compose([CenterCrop(grid_high), Resize(grid_low, antialias=False)])
    transform_input = Compose([CenterCrop(grid_high), Resize(grid_low)])
    transform_high = CenterCrop(grid_high)

    if initial_time is not None:
        input_files = [input_dir / f"{initial_time:%Y%m%d_%H%M}.h5"]

    elif target_time is None:
        input_files = sorted(input_dir.glob("*.h5"))

    else:
        input_files = [input_dir / f"{target_time:%Y%m}.h5"]

    output_dir = Path(env.get("OUTPUT_DIR"), exp_name, source_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 12
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False

    onnx_path = f"checkpoints/{exp_name}/{exp_name}.onnx"
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    for input_file in input_files:
        inputs = {}
        with h5.File(input_file, "r") as f:
            datas_static = []
            for var in cfg.dataset.var.input_static:
                data = torch.from_numpy(f[var][:])
                if var not in ["longitude", "latitude"]:
                    data = data.clamp(min=0.0)

                if source_name == "RWRF":
                    transform = (
                        transform_grid
                        if var in ["longitude", "latitude"]
                        else transform_input
                    )
                    data = transform(data.unsqueeze(0).to(dtype))

                datas_static.append(data)

            datas_static = torch.stack(datas_static, dim=1)

            time = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]
            if target_time is not None:
                target_index = next(
                    (i for i, t in enumerate(time) if t == target_time), None
                )
                print(time[target_index])
                target_index = slice(target_index, target_index + 1)
            else:
                target_index = slice(None)
            time = time[target_index]
            n = len(time)

            datas_single = []
            for var in cfg.dataset.var.input_single:
                data = torch.from_numpy(f[var][target_index,])
                if var not in ["u10", "v10"]:
                    data = data.clamp(min=0.0)
                if source_name == "RWRF":
                    data = transform_input(data.to(dtype))
                datas_single.append(data)

            datas_single = torch.stack(datas_single, dim=1)
            inputs["single"] = (
                torch.cat(
                    (datas_single, datas_static.expand(n, -1, -1, -1)),
                    axis=1,
                )
                .to(dtype)
                .numpy()
            )

            pressure_in = f["pressure"][:]
            sort_p = np.argsort(pressure_in)
            z_up = np.searchsorted(pressure_in[sort_p], cfg.dataset.var.z_input)
            z_up = sort_p[z_up]

            datas_upper = []
            for var in cfg.dataset.var.input_upper:
                data = []
                for z in z_up:
                    data.append(
                        torch.from_numpy(
                            f[var][
                                target_index,
                                z,
                            ]
                        )
                    )
                data = torch.stack(data, axis=-3)

                if var not in ["u", "v", "w"]:
                    data = data.clamp(min=0.0)
                if source_name == "RWRF":
                    data = transform_input(data.to(dtype))
                datas_upper.append(data)

            inputs["upper"] = torch.stack(datas_upper, dim=1).to(dtype).numpy()

        inputs["column_bottom"] = torch.zeros((n, 1)).to(dtype).numpy()
        inputs["column_left"] = torch.zeros((n, 1)).to(dtype).numpy()
        column_km = (
            torch.tensor([grid_low * cfg.dataset.res.resolution_input])
            .to(dtype)
            .numpy()
        )

        outputs_info = session.get_outputs()
        outputs_chunks = {o.name: [] for o in outputs_info}

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            batch_inputs = {k: inputs[k][start:end] for k in inputs if k != "column_km"}
            batch_inputs["column_km"] = column_km

            batch_output = session.run(None, batch_inputs)
            for i, o in enumerate(outputs_info):
                outputs_chunks[o.name].append(batch_output[i])

        outputs = {
            name: np.concatenate(chunks, axis=0).reshape(-1, *chunks[0].shape[2:])
            for name, chunks in outputs_chunks.items()
        }

        example_file = Path(env.get("INPUT_DIR"), f"{time[0]:%Y%m}.h5")

        with h5.File(example_file, "r") as f:
            time_ref = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]
            target_index_ref = []
            for i, t in enumerate(time_ref):
                if t in time:
                    target_index_ref.append(i)
            print(time_ref[target_index_ref[0]])
            print(time_ref[target_index_ref[-1]])

            lat = torch.from_numpy(f["latitude"][:])
            lat = transform_high(lat.unsqueeze(0)).squeeze(0).numpy()

            lon = torch.from_numpy(f["longitude"][:])
            lon = transform_high(lon.unsqueeze(0)).squeeze(0).numpy()

            pressure = f["pressure"][:]
            z_tar = (
                len(pressure)
                - 1
                - np.searchsorted(pressure[::-1], cfg.dataset.var.z_target)
            )

            datas_target = []
            for var in cfg.dataset.var.target:
                data = []
                for z in z_tar:
                    data.append(
                        torch.from_numpy(
                            f[var][
                                target_index_ref,
                                z,
                            ]
                        )
                    )
                data = torch.stack(data, axis=-3)

                if var not in ["u", "v", "w"]:
                    data = data.clamp(min=0.0)
                data = transform_high(data.to(dtype))
                datas_target.append(data)
            datas_target = torch.stack(datas_target, dim=1).to(dtype).numpy()

        if target_time is not None:
            output_stem = f"{target_time:%Y%m%d_%H%M}"

            if initial_time is not None:
                output_stem += f"_I{initial_time:%Y%m%d_%H%M}"
        else:
            output_stem = input_file.stem
        output_file = output_dir / f"{output_stem}.h5"

        write_h5_file(
            input_file,
            output_file,
            {
                "time": [t.isoformat().encode("utf-8") for t in time],
                "pressure": pressure,
                "latitude": lat,
                "longitude": lon,
            },
            outputs,
            datas_target,
            cfg.dataset.var.target,
            cfg.model.system.get("output_mode", "regress"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--input_dir",
        "-id",
        type=str,
        default=dotenv_values(".env").get("INPUT_DIR"),
        help="Enter input dir path.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="RWRF",
        help="Enter input source name.",
    )
    parser.add_argument(
        "-t",
        "--target_time",
        type=str,
        help="Enter target time in format YYYYmmddHH.",
    )
    parser.add_argument(
        "-i",
        "--initial_time",
        type=str,
        help="Enter initial time in format YYYYmmddHH.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Enter batch size.",
    )
    args = parser.parse_args()

    target_time = (
        datetime.strptime(args.target_time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.target_time
        else None
    )
    initial_time = (
        datetime.strptime(args.initial_time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.initial_time
        else None
    )
    main(
        args.exp,
        Path(args.input_dir),
        args.source,
        target_time,
        initial_time,
        args.batch_size,
    )

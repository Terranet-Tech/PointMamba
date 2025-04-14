"""Export model to ONNX format."""

from argparse import ArgumentParser, Namespace

import torch
from easydict import EasyDict

from tools import builder
from utils.config import cfg_from_yaml_file


def export(config: EasyDict, onnx_path: str) -> None:
    """Export model to ONNX format."""
    base_model = builder.model_builder(config.model)
    base_model.to(device="cuda")
    base_model.eval()
    torch.onnx.export(
        base_model,
        torch.randn(1, 8192, 3, device="cuda"),
        onnx_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model exported to {onnx_path}")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--config_path", type=str, default="cfgs/model.yaml", help="yaml config file"
    )
    parser.add_argument(
        "--onnx_path", type=str, default="model.onnx", help="onnx export path"
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    config = cfg_from_yaml_file(args.config_path)
    export(config, args.onnx_path)
    print(f"Model exported to {args.onnx_path}")


if __name__ == "__main__":
    main()

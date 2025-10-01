import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from typing import List


def export_model_to_onnx(
    model: torch.nn.Module,
    export_path: str,
    input_size: int,
    opset_version: int = 17,
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        export_path (str): Path where the ONNX model will be saved.
        input_size (int): Input image size (height/width).
        opset_version (int, optional): ONNX opset version. Defaults to 17.
    """
    dynamic_axes = {0: "batch_size"}

    torch.onnx.export(
        model,
        torch.rand(2, 3, input_size, input_size),
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": dynamic_axes, "output": dynamic_axes},
    )

    print(f"✅ Model exported to ONNX at {export_path}")


def validate_exported_model(
    model: torch.nn.Module, onnx_path: str, input_size: int
) -> None:
    """
    Validate an exported ONNX model against its PyTorch counterpart.

    Args:
        model (torch.nn.Module): Original PyTorch model.
        onnx_path (str): Path to the exported ONNX model.
        input_size (int): Input image size.
    """
    # Load ONNX model and check integrity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create test sample
    sample = torch.rand(2, 3, input_size, input_size)

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": sample.numpy()}
    ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)

    # PyTorch inference
    with torch.no_grad():
        torch_out = model(sample)

    # Compare results
    np.testing.assert_allclose(
        torch_out.numpy(),
        ort_outputs[0],
        rtol=5e-03,
        atol=5e-05,
    )
    print("✅ ONNXRuntime and PyTorch outputs match within tolerance.")


def process_model_path(model_path: str, input_size: int = 512) -> None:
    """
    Process a single model: load, export to ONNX, and validate.

    Args:
        model_path (str): Path to the PyTorch model (.pth).
        input_size (int, optional): Input size. Defaults to 512.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")

    # Load PyTorch model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Build ONNX output path
    model_dir = os.path.dirname(model_path)
    onnx_dir = os.path.join(os.path.dirname(model_dir), "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_model_path = os.path.join(onnx_dir, f"v{input_size}_{model_name}.onnx")

    # Export & validate
    export_model_to_onnx(model, onnx_model_path, input_size)
    validate_exported_model(model, onnx_model_path, input_size)


if __name__ == "__main__":
    MODEL_PATHS: List[str] = [
        "EXPERIMENTS/spot/weights/spot.pth",
        # Add more model paths here
    ]
    INPUT_SIZE = 512

    for path in MODEL_PATHS:
        process_model_path(path, INPUT_SIZE)

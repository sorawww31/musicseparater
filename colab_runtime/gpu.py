"""colab_runtime/gpu.py: PyTorch と ONNX の両方で GPU 実行を確認する。"""

from .config import GPU_HELP


def check_gpu() -> None:
    """provider の列挙だけでは検出できない CUDA ライブラリ不足も演算で検査する。"""
    import torch
    import numpy as np
    import onnx
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError(GPU_HELP)
    identity = torch.eye(2, device="cuda")
    torch.testing.assert_close(identity @ identity, identity)
    ort.preload_dlls()
    tensor = onnx.helper.make_tensor_value_info
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "x"], ["y"])], "gpu-check",
        [tensor("x", onnx.TensorProto.FLOAT, [2, 2])],
        [tensor("y", onnx.TensorProto.FLOAT, [2, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX が GPU を使えません。準備ログの CUDA / cuDNN エラーを確認してください。")
    np.testing.assert_allclose(session.run(None, {"x": np.eye(2, dtype=np.float32)})[0], np.eye(2))
    print(f"GPU 確認 OK: {torch.cuda.get_device_name(0)} / ONNX CUDA", flush=True)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        print("この GPU では SepACap は使えません。他のモデルを選んでください。", flush=True)


if __name__ == "__main__":
    check_gpu()

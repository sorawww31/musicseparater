# backend/Dockerfile: CUDA PyTorch を再利用して音源分離バックエンドを実行する。
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# CUDA 対応 PyTorch はベースイメージから継承し、uv 環境からも参照できるようにする。
ENV UV_PROJECT_ENVIRONMENT=/opt/musicseparater-venv
ENV VIRTUAL_ENV=/opt/musicseparater-venv
ENV PATH="/opt/musicseparater-venv/bin:${PATH}"
# PyTorch イメージに同梱の CUDA 12 / cuDNN 9 ライブラリを ONNX Runtime からも解決できるようにする。
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

RUN uv venv --system-site-packages "$VIRTUAL_ENV"

COPY pyproject.toml uv.lock ./

# ベースイメージの CUDA 対応 torch を再インストールせず、追加依存だけを導入する。
RUN uv sync --locked --no-install-project

COPY . .

RUN uv sync --locked --no-install-project
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "inference.py", "--help"]

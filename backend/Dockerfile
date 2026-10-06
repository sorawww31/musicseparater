# backend/Dockerfile: CUDA PyTorch を再利用して音源分離バックエンドを実行する。
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app
ENV TZ=Asia/Tokyo
# CUDA 対応 PyTorch はベースイメージから継承し、uv 環境からも参照できるようにする。
ENV UV_PROJECT_ENVIRONMENT=/opt/musicseparater-venv
ENV VIRTUAL_ENV=/opt/musicseparater-venv
ENV PATH="/opt/musicseparater-venv/bin:${PATH}"
# PyTorch イメージに同梱の CUDA 12 / cuDNN 9 ライブラリを ONNX Runtime からも解決できるようにする。
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

RUN uv venv --system-site-packages "$VIRTUAL_ENV"

# 研究コードはライセンス確認が必要なため再配布せず、再現可能なcommitをbuild時に取得する。
ADD https://github.com/jihoojung0106/unmixx.git#8e750521b5942f4717656cac86a23cf0bd90dea5 /opt/model-sources/unmixx
ADD https://github.com/ETH-DISCO/SepACap.git#bb96b6c6ddd37bd9e2ace9e0a5b9080f186cbcc3 /opt/model-sources/sepacap
# 論文が公開するConcatenation系4条件とclean embeddingを固定hashで取得する。
# λはdual lossの学習時重みで、重みごとに別checkpointとして配布されている。
ADD --checksum=sha256:ec69648c9b1628bfb836072be2d66ad341fe23a87ad09934cc6e7edbb837b674 https://drive.usercontent.google.com/download?id=1Lw1oVqsFMr8QKaLEIA5Gx-KE1OaJI4-_&export=download&confirm=t /opt/model-checkpoints/singer-informed/concat-none/vocals.pth
ADD --checksum=sha256:e38c0312577d1f345fe1fbe6baa525398df5ce7b7ab8cea77dc0ba015f591011 https://drive.usercontent.google.com/download?id=1FH4sWEX4vUqMEy9qcXZIHop1IDyZpGJV&export=download&confirm=t /opt/model-checkpoints/singer-informed/concat-0.05/vocals.pth
ADD --checksum=sha256:691e52121e63529d29ea7d06dd4ce08e91789d47fe5ffeed1782a83c14c4ed48 https://drive.usercontent.google.com/download?id=1sbK-YKzJ36jGISuz0ah4JI2Fzw3tsV9n&export=download&confirm=t /opt/model-checkpoints/singer-informed/concat-0.1/vocals.pth
ADD --checksum=sha256:39d5e7a61b5f8470d47233badd4220fda6f617d6bf6b697adcf52f41b936c400 https://drive.usercontent.google.com/download?id=1Q9uMQK-h2u-KdwoD6iO1FHkMhyckVqts&export=download&confirm=t /opt/model-checkpoints/singer-informed/concat-0.2/vocals.pth
ADD --checksum=sha256:5f52e199edc0a753378bfbf335408dd3044828c9989684f602765e7169fbc1fc https://drive.usercontent.google.com/download?id=1DTzLtKgtxBdwuOD5D4K8b9Ugj8PSN-EA&export=download&confirm=t /opt/model-checkpoints/singer-informed/clean-embedding.pth

COPY pyproject.toml uv.lock ./

# ベースイメージの CUDA 対応 torch を再インストールせず、追加依存だけを導入する。
RUN uv sync --locked --no-install-project

COPY . .

RUN uv sync --locked --no-install-project
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "-m", "src.inference", "--help"]

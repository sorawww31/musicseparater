<!-- backend/README.md: BS PolarFormer のローカル実行方法を記録する。 -->

# Backend

`bgkb/bs_polarformer` の ONNX モデルを使い、44.1 kHz stereo 音声を
`vocals.wav` と `instrumental.wav` に分離します。初回実行時は、既定の FP16 なら公式配布済みの
約 108 MB モデル、`--precision fp32` なら約 211 MB モデルを Hugging Face キャッシュへダウンロードします。

推論チャンクは既定で 441,000 samples（10 秒）です。2 倍のオーバーラップを平均して境界ノイズを
抑えます。Python API から同じ設定で連続して呼ぶ場合は ONNX session をプロセス内で再利用し、GPU
推論を一つずつ実行します。CUDA 環境では STFT・mask・iSTFT を GPU に置き、ONNX Runtime の I/O
Binding で mask を CPU へ往復させません。

実装の責務は次のように分けています。`inference.py` は公開 API・CLI・実行の直列化、
`src/separation_request.py` は API 入力の検証と正規化、`src/models.py` は ONNX 推論、
`src/audio_files.py` は音声ファイルの読み書きを担当します。公開する `inference()` と
`BSPolarFormer.separate_file()` の呼び出し方は変わりません。

```bash
docker compose run --rm -v "$PWD:/workspace" backend \
  python inference.py /workspace/song.mp3 --output-dir /workspace/output \
  --cache-dir /app/.model-cache
```

通常は CUDA を使い、GPU を使用できない環境では `--cpu` を付けてください。起動時の
`ONNX Runtime providers: CUDAExecutionProvider, ...` で、実際に CUDA が選択されたことを確認できます。
`--precision fp16` は公式の FP16 モデルをそのまま使うため、ローカル変換時の丸め warning は出ません。
VRAM が足りない場合は、`--chunk-size 220500` のように小さくしてください（小さいほど VRAM は下がり、
処理時間は増えます）。処理中は `tqdm` でチャンク単位の進捗と残り時間を表示します。

Hugging Face の未認証 warning を消してダウンロード上限を上げるには、`backend/.env` に
`HF_TOKEN=hf_...` を設定してください。このトークンはリポジトリへコミットしないでください。
ローカルに FP32 ONNX を配置済みなら、`--model-path /path/to/model.onnx` でダウンロードを省略できます。
この場合の `--precision fp16` は互換性のためローカルで mixed FP16 へ変換します。

Python からは以下の形で呼び出します。

```python
from inference import inference

paths = inference(
    "bs-polarformer",
    {"input_path": "song.wav", "output_dir": "output", "precision": "fp16"},
)
```

テストは `docker compose run --rm backend python -m unittest discover -s tests -v` で実行できます。

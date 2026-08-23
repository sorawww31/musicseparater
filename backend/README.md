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
`src/audio_files.py` は音声ファイルの読み書き、`src/storage.py` は HTTP の保存構造を担当します。
コーデック・ONNX Runtime・モデル変換などの重い依存は、必要な処理を実行するときに遅延ロードします。
`model_path`、`cache_dir`、`providers`、`precision`、`chunk_size` はサーバー設定として
`src/config.py` に置き、HTTP リクエストには公開しません。Python API から辞書で渡す場合も、
推論開始前に型と対応値を検証します。

## FastAPI

起動方法は次の2通りです。

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
python main.py
```

`POST /audios` は `multipart/form-data` で元音源を保存します。返された `audio_id` は、
同じ音源を別設定で再度分離するときにも利用できます。

```bash
curl -X POST http://localhost:8000/audios \
  -F 'uploadfile=@song.wav'
```

```json
{
  "audio_id": "audio123",
  "original_url": "/audios/audio123/original"
}
```

`POST /separations` は JSON で保存済みの `audio_id` と推論設定を受け取ります。処理は現在は
同期実行で、完了後に `job_id` と stem URL を返します。

```bash
curl -X POST http://localhost:8000/separations \
  -H 'Content-Type: application/json' \
  -d '{"audio_id":"audio123","model_id":"bs-polarformer","num_vocals":2}'
```

レスポンスにはサーバー内部のファイルパスを含めず、Reactから取得できるAPI URLを返します。

```json
{
  "job_id": "job123",
  "status": "completed",
  "stems": {
    "vocals": "/separations/job123/stems/vocals",
    "instrumental": "/separations/job123/stems/instrumental"
  }
}
```

`GET /audios/{audio_id}/original` で元音源、
`GET /separations/{job_id}/stems/{stem}` で分離済みWAVを取得できます。Reactでは
`<audio src={original_url} controls />` や `<audio src={stems.vocals} controls />` のように利用します。

保存先は次の構造です。`job_id` はモデル名を含まない不透明な ID で、使用した
`model_id` は `metadata.json` にだけ記録します。

```text
audio/
├── sources/{audio_id}/
│   ├── original.wav
│   └── metadata.json
└── separations/{job_id}/
    ├── metadata.json
    └── stems/
        ├── vocals.wav
        └── instrumental.wav
```

現在の BS PolarFormer は2ステムモデルのため、`drums.wav` と `bass.wav` は生成しません。
4ステムモデルを追加する場合は、モデル層の出力ステム定義も併せて変更します。

```bash
docker compose run --rm -v "$PWD:/workspace" backend \
  python -m src.inference /workspace/song.mp3 --output-dir /workspace/output \
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

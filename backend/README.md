<!-- backend/README.md: 実装済みモデル、非同期API、GPU実行条件を記録する。 -->

# Music Separation Backend

FastAPI と1本のGPUキューで、次のモデルを実行します。

| model_id | 処理 | 公開ステム | 内部精度 |
| --- | --- | --- | --- |
| `bs-polarformer` | 2ステム分離 | vocals, instrumental | ONNX FP16（既定） |
| `unmixx` | 2人の歌声分離 | singer_1, singer_2, instrumental | PyTorch FP32 |
| `sepacap` | アカペラ声部分離 | 7声部, instrumental | PyTorch BF16 |
| `singer-informed` | 参照歌手抽出 | target_vocal, residual | ONNX FP16 → PyTorch FP32 |

UNMIXX と SepACap は、最初に BS PolarFormer で市販曲から vocals を抽出します。歌声モデルには
24 kHz stereo、96,000 samples（4秒）、50% overlapで入力し、完成したWAVは44.1 kHz stereoかつ
instrumentalと同じ長さへ戻します。UNMIXXの匿名出力は左右チャンネルと隣接チャンクの相関で順を
合わせます。SepACapの順序は `alto / bass / finger_snap / lead_vocal / soprano / tenor /
vocal_percussion` 固定です。

`singer-informed` も BS PolarFormerを前段に置きます。論文は混合曲を直接入力しますが、採用モデルは
帯域16 kHz・3層LSTMの条件付きOpen-Unmixで、単体では伴奏が `target_vocal` へ大きく残ります。伴奏除去は
BS PolarFormerに任せ、条件付きモデルには44.1 kHz monoの歌声だけを渡し、外した伴奏は最後に `residual`
へ戻します。参照音声も同じ前段へ通して、伴奏なしの歌声で学習された clean embedding の条件へそろえます。
論文どおり混合曲を直接入力して比較する場合は `SingerInformedConfig.cascade_vocal_extraction` を
`False` にします。

参照音声からは重複しない高エネルギー区間を `enrollment_segment_count` 個選び、32次元embeddingを平均して
1フレーズの音高や母音へ条件付けが偏らないようにします。推定振幅はそのまま使わず、補完推定との比を
ratio maskへ正規化してから適用します。`mask_exponent` がmaskの鋭さで、1.0は従来の直接出力と一致し、
2.0（既定）はWiener相当、大きいほど伴奏優位のbinを強く抑えますが3.0を超えると musical noise が
出やすくなります。`mask_floor` を 0.05 程度にすると、残った伴奏の薄い床をさらに落とせます。

論文でduetのtarget SI-SDRが最良だった Concatenation + dual loss（λ=0.1）を採用しています。出力の
`residual` には対象外の歌手と伴奏が残ります。論文の実証範囲は最大2人であり、3人以上の品質は未検証です。

SepACapはステレオ4秒推論で約6.6 GiBを使うため、CUDAとBF16対応GPUが必須です。RTX 4070 Ti
12 GBで検証しています。複数歌声モデルにCPU fallbackはありません。UNMIXXはFP16化による利点が
確認できないためFP32のまま実行します。

## モデル取得

- BS PolarFormer: `bgkb/bs_polarformer` からHugging Face cacheへ取得。
- SepACap: `Tino3141/sepacap` の固定revisionから設定とcheckpointを取得し、SHA-256を検証。
- UNMIXX: 公式Hugging Face checkpointがないため、公式GitHubの固定commitに同梱された
  `ckpt/best.ckpt` を使用し、SHA-256を検証。
- Singer-Informed: [論文](https://arxiv.org/abs/2608.14516)の公開Google Driveから
  Concatenation λ=0.1 と clean embedding をDocker build時に取得し、Docker `ADD --checksum` と
  実行前検証の両方でSHA-256を固定。公開[研究コード](https://github.com/jocelynxu01/singer-separation-paper)
  では学習scriptが参照する条件付きOpen-Unmix classが欠けているため、公開state dictと標準Open-Unmixの
  層構成から互換モデルを局所実装し、`strict=True` で全parameterの一致を検証。

UNMIXX、SepACap、Singer-Informedの研究成果物には再配布条件を確認できていないものがあります。
SepACapのHugging Face checkpoint metadataはMITですが、現構成はローカルPoC用途に限定し、権利確認前に
Docker imageや組み込みソースを公開しないでください。

初回はモデルソースを固定commitでDocker imageへ取得します。

```bash
docker compose build backend
docker compose up backend
```

SepACap重みとBS PolarFormerは `backend/.model-cache` に保持されます。Hugging Faceの制限を避ける
場合は `backend/.env` に `HF_TOKEN=hf_...` を設定してください。

## 非同期API

音源をアップロードします。

```bash
curl -X POST http://localhost:8000/audios -F 'uploadfile=@song.wav'
```

返された `audio_id` を使ってジョブを作ります。レスポンスは `202 Accepted` です。

```bash
curl -X POST http://localhost:8000/separations \
  -H 'Content-Type: application/json' \
  -d '{"audio_id":"audio123","model_id":"unmixx","num_vocals":2}'
```

```json
{
  "job_id": "job123",
  "status": "queued",
  "status_url": "/separations/job123"
}
```

`GET /separations/{job_id}` をポーリングします。状態は `queued / running / completed / failed`、
段階は `queued / extracting_vocals / separating_singers / finalizing / completed / failed` です。
完了時だけ `stems` にAPI URLが入ります。
`extracting_vocals` の進捗率は、BS PolarFormerで完了したチャンク数から更新します。

```json
{
  "job_id": "job123",
  "status": "completed",
  "phase": "completed",
  "progress_percent": 100,
  "stems": {
    "singer_1": "/separations/job123/stems/singer_1",
    "singer_2": "/separations/job123/stems/singer_2",
    "instrumental": "/separations/job123/stems/instrumental"
  },
  "error": null
}
```

人数指定はUNMIXXとBS PolarFormerで未指定または2、SepACapとSinger-Informedでは未指定だけを受理します。
MedleyVoxは公開checkpointが2出力で3人以上の要件を満たさないため実装していません。

Target singer extractionでは対象楽曲と参照音声を別々にアップロードし、両方のIDを指定します。
参照音声は対象歌手だけが歌う3秒以上の音声を推奨します。別の曲を使用できますが、会話音声と3人以上の
混合は論文の評価範囲外です。

```bash
curl -X POST http://localhost:8000/audios -F 'uploadfile=@target-song.wav'
curl -X POST http://localhost:8000/audios -F 'uploadfile=@reference.wav'
curl -X POST http://localhost:8000/separations \
  -H 'Content-Type: application/json' \
  -d '{"audio_id":"song-id","reference_audio_id":"reference-id","model_id":"singer-informed"}'
```

ジョブはプロセス内 `ThreadPoolExecutor(max_workers=1)` で直列化しています。複数のUvicorn workerを
起動するとGPUキューと状態が分裂するため、現在は1 workerだけで運用してください。起動時に前回の
`queued / running` ジョブは `failed` へ回収され、途中成果物は公開されません。

## 検証

```bash
docker compose run --rm backend python -m unittest discover -s tests -v
docker compose run --rm backend python -m compileall -q main.py src model_runtime
```

UNMIXX、SepACap、Singer-Informedの実checkpointはRTX 4070 Ti上でスモーク確認済みです。
Singer-Informedは6秒の44.1 kHz入力で、全重みのstrict load、出力長、有限値、2ステム生成を確認しています。

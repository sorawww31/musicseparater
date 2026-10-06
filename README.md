<!-- README.md: 初めて使う人向けに、ローカル起動と音源分離の操作手順をまとめる。 -->

# Music Separater

ブラウザから音声ファイルをアップロードし、ボーカル・伴奏や複数の歌声に分離するアプリケーションです。分離した音声は画面上で試聴し、WAV 形式でダウンロードできます。

## 必要な環境

- Docker と Docker Compose
- NVIDIA GPU とドライバー、および Docker から GPU を利用できる環境（NVIDIA Container Toolkit など）
- 初回ビルド・モデル取得用のインターネット接続

現在の Docker Compose 構成は NVIDIA GPU を 1 台使用します。SepACap には BF16 対応 GPU が必要です。既存のバックエンド検証環境は RTX 4070 Ti（VRAM 12 GB）です。詳細は [バックエンド README](backend/README.md) を参照してください。

## 初回セットアップと起動

以下のコマンドは、リポジトリのルート（`compose.yaml` があるディレクトリ）で実行します。

### 1. 環境設定ファイルを用意する

```bash
touch backend/.env
```

`backend/.env` は Docker Compose が読み込むため、トークンを使わない場合も空ファイルとして用意してください。既存のファイルの内容は上記コマンドで変更されません。

Hugging Face のトークンを利用する場合は、`backend/.env` に以下を設定します。

```dotenv
HF_TOKEN=自分のHugging_Faceトークン
```

フロントエンドの API 接続先は、既定で `http://localhost:8000` です。変更が必要な場合は [設定例](frontend/.env.example) を参考に `frontend/.env.local` に `VITE_API_URL` を設定してください。既存の `frontend/.env` に設定がある場合も確認してください。

### 2. ビルドして依存パッケージを導入する

```bash
docker compose build
docker compose run --rm frontend npm install
```

初回ビルドではモデルのソースや一部の重みも取得するため、時間がかかります。フロントエンドの依存パッケージは上記の `npm install` で導入します。

### 3. 起動する

```bash
docker compose up -d
```

- アプリ画面: <http://localhost:5173>
- バックエンド API ドキュメント: <http://localhost:8000/docs>

起動状況や処理ログは次のコマンドで確認できます。

```bash
docker compose ps
docker compose logs -f backend frontend
```

ログ表示は `Ctrl+C` で終了できます。アプリを停止する場合は次を実行してください。

```bash
docker compose down
```

次回からは `docker compose up -d` で起動できます。Dockerfile やバックエンドの依存関係を変更した場合は、再ビルドしてください。

## 基本的な使い方

1. <http://localhost:5173> を開きます。
2. 「対象楽曲」に音声ファイルをドラッグ＆ドロップするか、クリックして選択します。選択できる形式は **MP3 / WAV / FLAC** です。
3. 「分離設定」で分離タイプとモデルを選びます。ボーカルと伴奏に分ける場合は、初期設定の「2ステム」→「BS PolarFormer」を使います。
4. 「分離を開始」をクリックします。音声のアップロード後、処理の段階と進捗が表示されます。
5. 「完了」になったら各音声を再生し、必要なものを「WAVをダウンロード」から保存します。

ステムとは、分離後のボーカルや伴奏などの個別の音声です。初回実行時は追加のモデル取得が発生することがあります。ジョブは 1 件ずつ処理されるため、別のジョブの実行中は「処理待ち」になります。

## 目的別のモデル選択

| 目的 | 分離タイプ / 方式 | モデル | 分離結果 |
| --- | --- | --- | --- |
| ボーカルと伴奏に分ける | 2ステム | BS PolarFormer | Vocals、Instrumental |
| 2 人の歌声を分ける | 複数歌声 / Blind separation | UNMIXX | Singer 1、Singer 2、Instrumental |
| アカペラを声部ごとに分ける | 複数歌声 / Blind separation | SepACap | alto、bass、finger snap、lead vocal、soprano、tenor、vocal percussion、Instrumental |
| 指定した歌手の声を取り出す | 複数歌声 / Target singer extraction | Singer-Informed (Concat λ=0.1) | Target Vocal、Other Singer + Instrumental |

「4ステム」「6ステム」と、モデル一覧で「未実装」と表示されるモデルは現在利用できません。

### 2 人の歌声を分ける

「複数歌声」→「Blind separation」→「UNMIXX」を選択します。参照音声は不要です。出力は匿名の `Singer 1` / `Singer 2` で、歌手名を識別する機能ではありません。

SepACap は歌手の人数を指定して分けるモデルではなく、7 つの固定声部に分けるモデルです。

### 特定の歌手の声を取り出す

1. 「対象楽曲」に分離したい曲を選択します。
2. 「複数歌声」→「Target singer extraction」を選択します。
3. 「Singer-Informed (Concat λ=0.1)」を選択します。
4. 表示された「対象歌手の参照音声」に、**その歌手だけが歌う 3 秒以上の音声**を選択します。対象楽曲とは別の曲でも構いません。
5. 「分離を開始」をクリックします。

`Target Vocal` が対象歌手の声、`Other Singer + Instrumental` が残りの歌声と伴奏です。3 人以上が混ざった曲での品質は未検証です。

## ファイルの保存先

Docker Compose のボリューム設定により、音声とキャッシュはホスト側の次の場所に保存されます。

| 保存内容 | パス |
| --- | --- |
| アップロードした楽曲・参照音声とメタデータ | `backend/audio/sources/<audio_id>/` |
| ジョブのメタデータ | `backend/audio/separations/<job_id>/metadata.json` |
| 完成した WAV ファイル | `backend/audio/separations/<job_id>/stems/` |
| BS PolarFormer・SepACap のモデルキャッシュ | `backend/.model-cache/` |

`docker compose down` で停止しても、これらのファイルは残ります。バックエンドを再起動すると、前回の処理待ち・実行中ジョブは失敗扱いになります。途中からの再開はできないため、画面から分離をやり直してください。

## 困ったとき

| 症状 | 確認すること |
| --- | --- |
| `backend/.env` がないと言われる | ルートで `touch backend/.env` を実行します。 |
| フロントエンドで `vite: not found` が出る | `docker compose run --rm frontend npm install` を実行してから起動し直します。 |
| アップロードや API 接続に失敗する | バックエンドの起動と `VITE_API_URL` を確認します。既定の CORS 設定で許可される画面の URL は `http://localhost:5173` です。 |
| GPU 関連のエラーで起動・分離に失敗する | Docker から NVIDIA GPU を利用できるか確認します。SepACap では BF16 対応も必要です。 |
| 分離が失敗する | 画面のエラーと `docker compose logs --tail=100 backend` を確認します。 |
| 「分離を開始」を押せない | 対象楽曲と実装済みモデルを選択します。Target singer extraction では参照音声も必要です。 |

API の利用例やモデルの詳細は [バックエンド README](backend/README.md)、画面側の設定・検証方法は [フロントエンド README](frontend/README.md) を参照してください。

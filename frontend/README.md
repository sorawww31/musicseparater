<!-- frontend/README.md: 非同期分離画面の起動方法と責務を記録する。 -->

# Music Separation Frontend

音声選択、実装済みモデル選択、アップロード、非同期ジョブの進捗、WAV試聴・ダウンロードを一画面で
扱うReact + TypeScript + Viteフロントエンドです。

実装済みモデルは BS PolarFormer、UNMIXX、SepACap、Singer-Informedです。4/6ステム候補と
MedleyVoxは一覧に残しますが、`未実装` として選択不能にしています。複数歌声では Blind separation と
Target singer extractionを切り替えます。後者は対象楽曲に加えて対象歌手だけの3秒以上の参照音声を
アップロードし、`Target Vocal` と `Other Singer + Instrumental` を返します。

主な責務は次の通りです。

- `src/App.tsx`: 楽曲・参照音声の `audio_id` 再利用とジョブポーリング。
- `src/api/audio.ts`: FastAPIの型付きHTTP契約。
- `src/components/SeparationSettings.tsx`: 実装状態を反映したモデル選択。
- `src/components/JobStatusPanel.tsx`: 段階・進捗・試聴・ダウンロード。
- `src/config.ts`: API URLと1秒のポーリング間隔。

API originは既定で `http://localhost:8000` です。変更する場合は `.env.example` を参考に
`VITE_API_URL` を設定してください。バックエンドが返す相対URLは `resolveApiUrl` で結合します。

```bash
docker compose up frontend
```

検証もホストのNode環境ではなくDocker内で実行します。

```bash
docker compose run --rm frontend npm install
docker compose run --rm frontend npm test
docker compose run --rm frontend npm run lint
docker compose run --rm frontend npm run build
```

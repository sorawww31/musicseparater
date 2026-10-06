// frontend/src/config.test.mjs: API origin と相対パスの結合規則を確認します。
// ブラウザが Vite 側ではなくバックエンド側の音声URLを参照することを守ります。
import assert from 'node:assert/strict'
import test from 'node:test'

import { apiUrl, pollingIntervalMs, resolveApiUrl } from './config.ts'

test('API URL の既定値はローカルバックエンドを参照する', () => {
  assert.equal(apiUrl, 'http://localhost:8000')
})

test('API origin とレスポンスの相対パスを単一スラッシュで結合する', () => {
  const stemPath = '/separations/job123/stems/vocals'

  assert.equal(
    resolveApiUrl(stemPath, 'https://api.example.com/'),
    'https://api.example.com/separations/job123/stems/vocals',
  )
})

test('進捗ポーリング間隔を設定から取得する', () => {
  assert.equal(pollingIntervalMs, 1000)
})

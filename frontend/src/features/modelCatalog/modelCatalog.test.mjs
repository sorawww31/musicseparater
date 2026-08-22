// frontend/src/features/modelCatalog/modelCatalog.test.mjs
// Node 標準 test runner で、モデル一覧が設定値と矛盾していないことを確認します。
// 追加依存なしで Docker 上の npm test から実行できる最小テストです。
import assert from 'node:assert/strict'
import test from 'node:test'

import { acceptedAudioFileTypes, defaultSeparationMode } from '../../config.ts'
import { modelsByMode, separationModes } from './modelCatalog.ts'

test('公開している各分離モードに候補モデルがある', () => {
  assert.deepEqual(separationModes, ['2stem', '4stem', '6stem', 'multi-singer'])

  for (const mode of separationModes) {
    assert.ok(modelsByMode[mode].length > 0, `${mode} にモデル候補が必要です`)
  }
})

test('既定値とアップロード対象拡張子が現在の画面仕様と一致する', () => {
  assert.ok(Object.hasOwn(modelsByMode, defaultSeparationMode))
  assert.equal(acceptedAudioFileTypes, '.mp3,.wav,.flac')
})

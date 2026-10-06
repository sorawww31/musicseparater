// frontend/src/features/modelCatalog/modelCatalog.test.mjs
// Node 標準 test runner で、モデル一覧が設定値と矛盾していないことを確認します。
// 追加依存なしで Docker 上の npm test から実行できる最小テストです。
import assert from 'node:assert/strict'
import test from 'node:test'

import { acceptedAudioFileTypes, defaultSeparationMode } from '../../config.ts'
import {
  firstAvailableModel,
  modelRequiresEnrollment,
  modelsByMode,
  modelsForSelection,
  separationModes,
} from './modelCatalog.ts'

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

test('複数歌声モデルの公開ステムと実装状態がAPI契約に一致する', () => {
  const unmixx = modelsByMode['multi-singer'].find((model) => model.id === 'unmixx')
  const sepacap = modelsByMode['multi-singer'].find((model) => model.id === 'sepacap')
  const dptnet = modelsByMode['multi-singer'].find((model) => model.id === 'jacappella-dptnet')
  const medleyvox = modelsByMode['multi-singer'].find((model) => model.id === 'medleyvox')
  const singerInformed = modelsByMode['multi-singer'].find(
    (model) => model.id === 'singer-informed',
  )

  assert.deepEqual(unmixx?.stems, ['singer_1', 'singer_2', 'instrumental'])
  assert.deepEqual(sepacap?.stems, [
    'alto', 'bass', 'finger_snap', 'lead_vocal', 'soprano', 'tenor',
    'vocal_percussion', 'instrumental',
  ])
  // 声部名つきモデルのステム順は、バックエンドの出力index順と一致させる。
  assert.deepEqual(dptnet?.stems, [
    'vocal_percussion', 'bass', 'alto', 'tenor', 'soprano', 'lead_vocal',
    'instrumental',
  ])
  assert.deepEqual(medleyvox?.stems, ['singer_1', 'singer_2', 'instrumental'])
  assert.equal(unmixx?.availability, 'available')
  assert.equal(sepacap?.availability, 'available')
  assert.equal(dptnet?.availability, 'available')
  assert.equal(medleyvox?.availability, 'available')
  assert.equal(firstAvailableModel('multi-singer')?.id, 'unmixx')
  assert.deepEqual(modelsForSelection('multi-singer', 'target'), [singerInformed])
  assert.equal(firstAvailableModel('multi-singer', 'target')?.id, 'singer-informed')
  assert.deepEqual(singerInformed?.stems, ['target_vocal', 'residual'])
  assert.equal(modelRequiresEnrollment('singer-informed'), true)
  assert.equal(modelRequiresEnrollment('unmixx'), false)
  // Blind separation の4モデルはすべて参照音声なしで動く。
  for (const model of modelsForSelection('multi-singer')) {
    assert.equal(modelRequiresEnrollment(model.id), false)
  }
})

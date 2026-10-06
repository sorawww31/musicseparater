// frontend/src/features/modelCatalog/modelCatalog.ts
// 実装済みモデルと将来候補を区別し、UIから未実装APIを呼ばないための一覧です。
export type SeparationMode = '2stem' | '4stem' | '6stem' | 'multi-singer'
export type MultiSingerStrategy = 'blind' | 'target'

export type ModelOption = {
  id: string
  name: string
  stems: string[]
  source: string
  description: string
  availability: 'available' | 'planned'
  strategy?: MultiSingerStrategy
  requiresEnrollment?: boolean
}

export const modelsByMode: Record<SeparationMode, ModelOption[]> = {
  '2stem': [
    {
      id: 'bs-polarformer',
      name: 'BS PolarFormer',
      stems: ['vocals', 'instrumental'],
      source: 'bgkb/bs_polarformer',
      description: '44.1 kHzでボーカルと伴奏を分離します。',
      availability: 'available',
    },
    {
      id: 'melband-roformer-kim',
      name: 'Mel-Band RoFormer (Kim)',
      stems: ['vocals', 'instrumental'],
      source: 'AEmotionStudio/roformer-models',
      description: '高品質なRoFormer系モデル。現在は実装予定です。',
      availability: 'planned',
    },
    {
      id: 'bs-roformer-viperx',
      name: 'BS-RoFormer (ViperX)',
      stems: ['vocals', 'instrumental'],
      source: 'AEmotionStudio/roformer-models',
      description: '定番のRoFormer系モデル。現在は実装予定です。',
      availability: 'planned',
    },
  ],
  '4stem': [
    {
      id: 'scnet-xl-ihf',
      name: 'SCNet XL IHF',
      stems: ['vocals', 'drums', 'bass', 'other'],
      source: 'noblebarkrr/mvsepless_resources',
      description: '4ステム分離モデル。現在は実装予定です。',
      availability: 'planned',
    },
    {
      id: 'htdemucs',
      name: 'HTDemucs',
      stems: ['vocals', 'drums', 'bass', 'other'],
      source: 'facebookresearch/demucs',
      description: '定番の4ステム分離モデル。現在は実装予定です。',
      availability: 'planned',
    },
  ],
  '6stem': [
    {
      id: 'bs-roformer-sw-6stem',
      name: 'BS-RoFormer SW 6 Stem',
      stems: ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'],
      source: 'elicwhite/bs-roformer-sw-6stem-onnx',
      description: '6ステム分離モデル。現在は実装予定です。',
      availability: 'planned',
    },
  ],
  'multi-singer': [
    {
      id: 'unmixx',
      name: 'UNMIXX',
      stems: ['singer_1', 'singer_2', 'instrumental'],
      source: 'jihoojung0106/unmixx',
      description: '2人の歌声を匿名のSinger 1 / 2へ分離します。',
      availability: 'available',
      strategy: 'blind',
    },
    {
      id: 'sepacap',
      name: 'SepACap',
      stems: [
        'alto', 'bass', 'finger_snap', 'lead_vocal', 'soprano', 'tenor',
        'vocal_percussion', 'instrumental',
      ],
      source: 'Tino3141/sepacap',
      description: 'アカペラを7つの固定声部と伴奏へ分離します。',
      availability: 'available',
      strategy: 'blind',
    },
    {
      id: 'medleyvox',
      name: 'MedleyVox / iSRNet',
      stems: ['singer_1', 'singer_2'],
      source: 'Cyru5/MedleyVox',
      description: '公開checkpointは2出力のため、3人以上の要件を満たさず未実装です。',
      availability: 'planned',
      strategy: 'blind',
    },
    {
      id: 'singer-informed',
      name: 'Singer-Informed (Concat λ=0.1)',
      stems: ['target_vocal', 'residual'],
      source: 'jocelynxu01/singer-separation-paper',
      description: '3秒以上の参照音声を使い、指定した歌手だけを抽出します。',
      availability: 'available',
      strategy: 'target',
      requiresEnrollment: true,
    },
  ],
}

export const separationModes = Object.keys(modelsByMode) as SeparationMode[]

export function modelsForSelection(
  mode: SeparationMode,
  strategy: MultiSingerStrategy = 'blind',
): ModelOption[] {
  if (mode !== 'multi-singer') return modelsByMode[mode]
  return modelsByMode[mode].filter((model) => model.strategy === strategy)
}

export function firstAvailableModel(
  mode: SeparationMode,
  strategy: MultiSingerStrategy = 'blind',
): ModelOption | undefined {
  return modelsForSelection(mode, strategy).find((model) => model.availability === 'available')
}

export function modelRequiresEnrollment(modelId: string): boolean {
  return Object.values(modelsByMode)
    .flat()
    .some((model) => model.id === modelId && model.requiresEnrollment === true)
}

// λは学習時のdual loss重みで、論文はConcatenation系の4条件をcheckpointとして公開しています。
// 推論時に連続で動かせる値ではないため、重みの選択肢として扱います。
export const conditioningLambdas = ['none', '0.05', '0.1', '0.2'] as const
export type ConditioningLambda = (typeof conditioningLambdas)[number]
export const defaultConditioningLambda: ConditioningLambda = '0.1'

export const conditioningLambdaLabels: Record<ConditioningLambda, string> = {
  none: 'なし',
  '0.05': '0.05',
  '0.1': '0.1（論文最良）',
  '0.2': '0.2',
}

export function modelSupportsConditioningLambda(modelId: string): boolean {
  return modelId === 'singer-informed'
}

// frontend/src/components/SeparationSettings.tsx
// モード・方式・実行可能モデルを選び、人数はモデル契約から固定します。
import {
  type ConditioningLambda,
  type SeparationMode,
  type MultiSingerStrategy,
  conditioningLambdaLabels,
  conditioningLambdas,
  modelSupportsConditioningLambda,
  modelsForSelection,
  separationModes,
} from '../features/modelCatalog/modelCatalog.ts'

const modeLabels: Record<SeparationMode, string> = {
  '2stem': '2ステム',
  '4stem': '4ステム',
  '6stem': '6ステム',
  'multi-singer': '複数歌声',
}

const strategyLabels: Record<MultiSingerStrategy, string> = {
  blind: 'Blind separation',
  target: 'Target singer extraction',
}

type SeparationSettingsProps = {
  selectedMode: SeparationMode
  selectedStrategy: MultiSingerStrategy
  selectedModel: string
  selectedLambda: ConditioningLambda
  onModeChange: (mode: SeparationMode) => void
  onStrategyChange: (strategy: MultiSingerStrategy) => void
  onModelChange: (modelId: string) => void
  onLambdaChange: (conditioningLambda: ConditioningLambda) => void
  disabled: boolean
}

export function SeparationSettings({
  selectedMode,
  selectedStrategy,
  selectedModel,
  selectedLambda,
  onModeChange,
  onStrategyChange,
  onModelChange,
  onLambdaChange,
  disabled,
}: SeparationSettingsProps) {
  const modelOptions = modelsForSelection(selectedMode, selectedStrategy)
  const showLambda = modelSupportsConditioningLambda(selectedModel)

  return (
    <section className="panel settings">
      <h2>分離設定</h2>
      <fieldset disabled={disabled}>
        <legend>分離タイプ</legend>
        <div className="option-grid mode-options">
          {separationModes.map((mode) => (
            <label key={mode} className="option-card">
              <input
                type="radio"
                name="separation-mode"
                checked={selectedMode === mode}
                onChange={() => onModeChange(mode)}
              />
              {modeLabels[mode]}
            </label>
          ))}
        </div>
      </fieldset>

      {selectedMode === 'multi-singer' && (
        <fieldset disabled={disabled}>
          <legend>方式</legend>
          <div className="option-grid strategy-options">
            {(Object.keys(strategyLabels) as MultiSingerStrategy[]).map((strategy) => (
              <label key={strategy} className="option-card">
                <input
                  type="radio"
                  name="multi-singer-strategy"
                  checked={selectedStrategy === strategy}
                  onChange={() => onStrategyChange(strategy)}
                />
                {strategyLabels[strategy]}
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <fieldset disabled={disabled}>
        <legend>モデル</legend>
        <div className="option-grid model-options">
          {modelOptions.map((model) => (
            <label
              key={model.id}
              className={`option-card model-card ${model.availability === 'planned' ? 'planned' : ''}`}
            >
              <input
                type="radio"
                name="separation-model"
                checked={selectedModel === model.id}
                disabled={model.availability === 'planned'}
                onChange={() => onModelChange(model.id)}
              />
              <span>
                <strong>{model.name}</strong>
                {model.availability === 'planned' && <small>未実装</small>}
                <span>{model.description}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {showLambda && (
        <fieldset disabled={disabled}>
          <legend>学習条件 λ</legend>
          <div className="option-grid lambda-options">
            {conditioningLambdas.map((conditioningLambda) => (
              <label key={conditioningLambda} className="option-card">
                <input
                  type="radio"
                  name="conditioning-lambda"
                  checked={selectedLambda === conditioningLambda}
                  onChange={() => onLambdaChange(conditioningLambda)}
                />
                {conditioningLambdaLabels[conditioningLambda]}
              </label>
            ))}
          </div>
          <p className="field-help">
            dual lossの学習時重みで、条件ごとに別のcheckpointへ切り替えます。
          </p>
        </fieldset>
      )}
    </section>
  )
}

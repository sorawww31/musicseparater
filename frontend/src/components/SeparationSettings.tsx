import { useState } from 'react'
import {
  type SeparationMode,
  separationModes,
  modelsByMode,
} from '../features/modelCatalog/modelCatalog'

type SeparationTypeSelectorProps = {
  selectedMode: SeparationMode
  setselectedMode: React.Dispatch<React.SetStateAction<SeparationMode>>
}
function SeparationTypeSelector({ selectedMode, setselectedMode }: SeparationTypeSelectorProps) {
    
    return (
    <div>
      <h3>分離タイプ</h3>

      <div style={{ display: 'flex', gap: '8px' }}>
        {/* mapで4個のボタンを並べてる */}
        {separationModes.map((mode) => (
          <button
            key={mode}
            type="button"
            onClick={() => setselectedMode(mode)}
            aria-pressed={selectedMode === mode}
          >
            {mode}
          </button>
        ))}
      </div>

      <p>Selected: {selectedMode}</p>
    </div>
  )
}

type SeparationModelSelectorProps = {
  selectedMode: SeparationMode
  selectedModel: string
  setSelectedModel: React.Dispatch<React.SetStateAction<string>>
}

function SeparationModelSelector({ selectedMode, selectedModel, setSelectedModel }: SeparationModelSelectorProps) {
  return (
    <div>
      <h3>分離モデル</h3>

      <div style={{ display: 'flex', gap: '8px' }}>
        {/* mapで4個のボタンを並べてる */}
        {modelsByMode[selectedMode].map((model) => (
          <button
            key={model.id}
            type="button"
            onClick={() => setSelectedModel(model.id)}
            aria-pressed={selectedModel === model.id}
          >
            {model.name}
          </button>
        ))}
      </div>

      <p>Selected: {selectedModel}</p>
    </div>  
    )
}

export function SeparationSettings() {
    const [selectedMode, setselectedMode] = useState<SeparationMode>('2stem');
    const [selectedModel, setSelectedModel] = useState<string>(modelsByMode[selectedMode][0].id);
    return (
    <section>
      <h2>分離設定</h2>
      <SeparationTypeSelector 
        selectedMode={selectedMode}
        setselectedMode={setselectedMode}
      />
      <SeparationModelSelector
        selectedMode={selectedMode}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />
    </section>
    )
}
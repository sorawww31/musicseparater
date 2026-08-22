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

      <div style={{ display: 'flex', gap: '16px' }}>
        {separationModes.map((mode) => (
          <label key={mode}>
            <input
              type="radio"
              name="separation-mode"
              value={mode}
              checked={selectedMode === mode}
              onChange={() => setselectedMode(mode)}
            />

            {mode}
          </label>
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

      <div>
        {/* mapで4個のボタンを並べてる */}
        {modelsByMode[selectedMode].map((model) => (
            <label key={model.id}>
            <input
              type="radio"
              name="separation-model"
              value={model.name}
              checked={selectedModel === model.id}
              onChange={() => setSelectedModel(model.id)}
            />

            {model.name}
          </label>
        )
        )}
      </div>

      <p>Selected: {selectedModel}</p>
    </div>  
    )
}

type SingerCountSelectorProps = {
  selectedMode: SeparationMode
  singerCount: number | string
  setSingerCount: React.Dispatch<React.SetStateAction<number | string>>
}

function SingerCountSelector({ selectedMode, singerCount, setSingerCount }: SingerCountSelectorProps) {
    const labels = [2, 3, 4, 5, 'auto']
    const displayedSingerCount =
            selectedMode === 'multi-singer'
            ? singerCount
            : 'not multi-singer mode'
    
    return (
    <div>
        <h3>vocal人数</h3>

        <div style={{ display: 'flex', gap: '16px' }}>
            {labels.map((label) => (
                <label key={label}>
                    <input
                        type="radio"
                        name="singer-count"
                        value={label}
                        checked={singerCount === label && selectedMode === 'multi-singer'}
                        onChange={() => setSingerCount(label)}
                    />
                    {label}
                </label>
            ))}
        </div>

        <p>Selected: {displayedSingerCount}</p>
    </div>
    )
}

export function SeparationSettings() {
    const [selectedMode, setselectedMode] = useState<SeparationMode>('2stem');
    const [selectedModel, setSelectedModel] = useState<string>(modelsByMode[selectedMode][0].id);
    const [singerCount, setSingerCount] = useState<number | string>("not multi-singer mode");

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
      
    <SingerCountSelector
        selectedMode={selectedMode}
        singerCount={singerCount}
        setSingerCount={setSingerCount}
    />
      
    </section>
    )
}
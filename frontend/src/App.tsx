// frontend/src/App.tsx
// 選択状態と非同期APIフローを集約し、表示責務は各componentへ渡します。
import { useEffect, useRef, useState } from 'react'
import './App.css'
import {
  createSeparation,
  getSeparationJob,
  type SeparationJob,
  uploadAudio,
} from './api/audio.ts'
import { Header } from './components/Header.tsx'
import { JobStatusPanel } from './components/JobStatusPanel.tsx'
import { SeparationSettings } from './components/SeparationSettings.tsx'
import { UploadArea } from './components/UploadArea.tsx'
import { defaultSeparationMode, pollingIntervalMs } from './config.ts'
import {
  defaultConditioningLambda,
  firstAvailableModel,
  modelRequiresEnrollment,
  modelSupportsConditioningLambda,
  type ConditioningLambda,
  type MultiSingerStrategy,
  type SeparationMode,
} from './features/modelCatalog/modelCatalog.ts'

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(resolve, milliseconds)
    signal.addEventListener('abort', () => {
      window.clearTimeout(timeout)
      reject(new DOMException('Aborted', 'AbortError'))
    }, { once: true })
  })
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [audioId, setAudioId] = useState<string | null>(null)
  const [referenceFile, setReferenceFile] = useState<File | null>(null)
  const [referenceAudioId, setReferenceAudioId] = useState<string | null>(null)
  const [selectedMode, setSelectedMode] = useState<SeparationMode>(defaultSeparationMode)
  const [selectedStrategy, setSelectedStrategy] = useState<MultiSingerStrategy>('blind')
  const [selectedLambda, setSelectedLambda] = useState<ConditioningLambda>(
    defaultConditioningLambda,
  )
  const [selectedModel, setSelectedModel] = useState(
    firstAvailableModel(defaultSeparationMode)?.id ?? '',
  )
  const [job, setJob] = useState<SeparationJob | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => () => controller.current?.abort(), [])

  function changeFile(nextFile: File | null) {
    controller.current?.abort()
    setFile(nextFile)
    setAudioId(null)
    setJob(null)
    setError(null)
    setBusy(false)
  }

  function changeMode(mode: SeparationMode) {
    const strategy = mode === 'multi-singer' ? selectedStrategy : 'blind'
    setSelectedMode(mode)
    setSelectedModel(firstAvailableModel(mode, strategy)?.id ?? '')
    setJob(null)
    setError(null)
  }

  function changeReferenceFile(nextFile: File | null) {
    controller.current?.abort()
    setReferenceFile(nextFile)
    setReferenceAudioId(null)
    setJob(null)
    setError(null)
    setBusy(false)
  }

  function changeStrategy(strategy: MultiSingerStrategy) {
    setSelectedStrategy(strategy)
    setSelectedModel(firstAvailableModel('multi-singer', strategy)?.id ?? '')
    setJob(null)
    setError(null)
  }

  function changeModel(modelId: string) {
    setSelectedModel(modelId)
    setJob(null)
    setError(null)
  }

  async function startSeparation() {
    const needsEnrollment = modelRequiresEnrollment(selectedModel)
    if (!file || !selectedModel || busy || (needsEnrollment && !referenceFile)) return
    controller.current?.abort()
    const requestController = new AbortController()
    controller.current = requestController
    setBusy(true)
    setError(null)
    setJob(null)
    try {
      let currentAudioId = audioId
      if (!currentAudioId) {
        currentAudioId = (await uploadAudio(file, requestController.signal)).audio_id
        setAudioId(currentAudioId)
      }
      let currentReferenceAudioId = referenceAudioId
      if (needsEnrollment && !currentReferenceAudioId && referenceFile) {
        currentReferenceAudioId = (
          await uploadAudio(referenceFile, requestController.signal)
        ).audio_id
        setReferenceAudioId(currentReferenceAudioId)
      }
      const created = await createSeparation(
        currentAudioId,
        selectedModel,
        requestController.signal,
        currentReferenceAudioId ?? undefined,
        modelSupportsConditioningLambda(selectedModel) ? selectedLambda : undefined,
      )
      while (!requestController.signal.aborted) {
        const currentJob = await getSeparationJob(created.status_url, requestController.signal)
        setJob(currentJob)
        if (currentJob.status === 'completed' || currentJob.status === 'failed') break
        await wait(pollingIntervalMs, requestController.signal)
      }
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === 'AbortError')) {
        setError(caught instanceof Error ? caught.message : '処理を開始できませんでした')
      }
    } finally {
      if (!requestController.signal.aborted) setBusy(false)
    }
  }

  const needsEnrollment = modelRequiresEnrollment(selectedModel)

  return (
    <main>
      <Header />
      <div className="workspace">
        <UploadArea
          file={file}
          onFileChange={changeFile}
          disabled={busy}
          title="対象楽曲"
        />
        <SeparationSettings
          selectedMode={selectedMode}
          selectedStrategy={selectedStrategy}
          selectedModel={selectedModel}
          selectedLambda={selectedLambda}
          onModeChange={changeMode}
          onStrategyChange={changeStrategy}
          onModelChange={changeModel}
          onLambdaChange={setSelectedLambda}
          disabled={busy}
        />
        {needsEnrollment && (
          <UploadArea
            file={referenceFile}
            onFileChange={changeReferenceFile}
            disabled={busy}
            title="対象歌手の参照音声"
            prompt="参照音声をここにドロップ"
            helpText="対象歌手だけが歌う3秒以上の音声を選んでください。別の曲でも構いません。"
          />
        )}
        {!selectedModel && <p className="notice">この分離タイプは現在準備中です。</p>}
        <button
          className="start-button"
          type="button"
          disabled={!file || !selectedModel || busy || (needsEnrollment && !referenceFile)}
          onClick={startSeparation}
        >
          {busy ? '処理中…' : '分離を開始'}
        </button>
        {error && <p className="error-message" role="alert">{error}</p>}
        <JobStatusPanel job={job} />
      </div>
    </main>
  )
}

export default App

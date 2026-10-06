// frontend/src/components/UploadArea.tsx
// ファイル選択とドラッグ&ドロップだけを扱い、通信状態はAppに集約します。
import { acceptedAudioFileTypes } from '../config.ts'

type UploadAreaProps = {
  file: File | null
  onFileChange: (file: File | null) => void
  disabled: boolean
  title?: string
  prompt?: string
  helpText?: string
}

export function UploadArea({
  file,
  onFileChange,
  disabled,
  title = '音声ファイル',
  prompt = 'ここに音声をドロップ',
  helpText,
}: UploadAreaProps) {
  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    onFileChange(event.target.files?.[0] ?? null)
  }

  function handleDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault()
    if (!disabled) onFileChange(event.dataTransfer.files?.[0] ?? null)
  }

  return (
    <section className="panel upload-panel">
      <h2>{title}</h2>
      {helpText && <p className="upload-help">{helpText}</p>}
      <label
        className={`drop-zone ${disabled ? 'disabled' : ''}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
      >
        <strong>{prompt}</strong>
        <span>またはクリックして選択</span>
        <input
          type="file"
          accept={acceptedAudioFileTypes}
          onChange={handleFileChange}
          disabled={disabled}
        />
      </label>
      {file && <p className="selected-file">選択中: {file.name}</p>}
    </section>
  )
}

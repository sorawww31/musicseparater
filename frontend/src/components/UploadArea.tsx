// frontend/src/components/UploadArea.tsx
// 音声ファイルの選択とドラッグ&ドロップだけを担当します。
// アップロード先 API はまだ無いため、選択中ファイル名の表示までを保持します。
import { useState } from 'react'
import type { CSSProperties} from 'react'

const acceptedAudioFileTypes = '.mp3,.wav,.flac'

const dropZoneStyle: CSSProperties = {
  border: '2px dashed #ccc',
  padding: '20px',
  marginTop: '10px',
}

export function UploadArea() {
  // これがstate file: 現在リアクトが覚えている値 setFile: これを使ってfileの値を更新する null: 初期値はnull, File: ファイルオブジェクト
  const [file, setFile] = useState<File | null>(null); 
  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] || null; // 選択されたファイルを取得する。もし選択されていなければnullを返す。
    // event.target.filesはFileList型で、<input type=file　で選ばれたファイルの全て。 複数のファイルが選択される可能性があるため、[0]で最初のファイルを取得する。
    setFile(selectedFile); // 選択されたファイルをstateに保存する。
  }
  function handleDragOver(event: React.DragEvent<HTMLElement>) {
    // 普通のHTML要素はデフォルトではドロップ先ではないからです。dragover のデフォルト動作をキャンセルすると、「この要素にはdropしていい」とブラウザに伝えられます
    event.preventDefault(); // デフォルトの動作をキャンセルする。これにより、ブラウザがファイルを開くのを防ぐ。
  }
  function handleDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault(); // デフォルトの動作をキャンセルする。これにより、ブラウザがファイルを開くのを防ぐ。
    const droppedFile = event.dataTransfer.files?.[0] || null; // ドロップされたファイルを取得する。もしドロップされていなければnullを返す。
    setFile(droppedFile); // ドロップされたファイルをstateに保存する。
  }

  return (
    <section>
      <h2>音声ファイルをアップロードしてください</h2>

      <label onDragOver={handleDragOver} onDrop={handleDrop} style={dropZoneStyle}>
        ここにファイルをドロップ
        またはクリックして選択

        <input type="file" accept={acceptedAudioFileTypes} onChange={handleFileChange} />
      </label>
      {file && <p>選択中: {file.name}</p>}
    </section>
  )
}

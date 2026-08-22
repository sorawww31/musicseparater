import {useState} from 'react'

function Header() {
  return (
    <header>
      <h1> AI音楽分離アプリケーション</h1>
      <p>
      このアプリケーションは、AIを使用して音楽トラックを分離することができます。<br />
      音声ファイルをアップロードすると、ボーカルと伴奏に分離された音声ファイルをダウンロードできます。<br />
      </p>
    </header>
  )
}

function UploadArea(){
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

      <label
  onDragOver={handleDragOver}
  onDrop={handleDrop}
  style={{border: '2px dashed #ccc', padding: '20px', marginTop: '10px'}}
>
  ここにファイルをドロップ
  またはクリックして選択

  <input
    type="file"
    accept=".mp3,.wav,.flac"
    onChange={handleFileChange}
  />
</label>
    {file && <p>選択中: {file.name}</p>}
    </section>
  )
}

type SeparationMode =
  | "2stem"
  | "4stem"
  | "6stem"
  | "multi-singer";

type ModelOption = {
  id: string;
  name: string;
  stems: string[];
  hfRepo: string;
  description: string;
};

const modelsByMode: Record<SeparationMode, ModelOption[]> = {
  "2stem": [
    {
      id: "bs-polarformer",
      name: "BS PolarFormer",
      stems: ["vocals", "instrumental"],
      hfRepo: "bgkb/bs_polarformer",
      description:
        "高品質なボーカル分離モデル。Multisong DatasetでVocal SDR 11.00 dB。51Mパラメータと比較的軽量で、ボーカル抽出の第一候補。",
    },
    {
      id: "melband-roformer-kim",
      name: "Mel-Band RoFormer (Kim)",
      stems: ["vocals", "instrumental"],
      hfRepo: "AEmotionStudio/roformer-models",
      description:
        "Mel周波数帯を利用する高性能RoFormer。Multisong DatasetでVocal SDR 10.98 dB。BS PolarFormerとほぼ同水準の高品質なボーカル分離。",
    },
    {
      id: "bs-roformer-viperx",
      name: "BS-RoFormer (ViperX)",
      stems: ["vocals", "instrumental"],
      hfRepo: "AEmotionStudio/roformer-models",
      description:
        "定番の高性能RoFormer系モデル。Multisong DatasetではVocal SDR 10.87 dB。高品質なボーカル/伴奏分離に向く。",
    },
  ],

  "4stem": [
    {
      id: "scnet-xl-ihf",
      name: "SCNet XL IHF",
      stems: ["vocals", "drums", "bass", "other"],
      hfRepo: "noblebarkrr/mvsepless_resources",
      description:
        "4-stem分離の高性能モデル。MUSDB18HQ testで平均SDR 10.08 dB。特にdrums 11.81 dB、vocals 11.42 dBと強く、4-stemの第一候補。",
    },
    {
      id: "bs-roformer-4stem",
      name: "BS-RoFormer 4 Stem",
      stems: ["vocals", "drums", "bass", "other"],
      hfRepo: "AEmotionStudio/roformer-models",
      description:
        "RoFormerベースの4-stemモデル。MUSDB18HQ testで平均SDR 9.65 dB。特にdrumsとvocalsの分離性能が高い。",
    },
    {
      id: "htdemucs",
      name: "HTDemucs",
      stems: ["vocals", "drums", "bass", "other"],
      hfRepo: "puar-playground/htdemucs",
      description:
        "Metaの定番Hybrid Transformer Demucs。公式報告ではMUSDB HQで約9.0 dB SDR。最新モデルより精度は劣るが、実績があり扱いやすいベースライン。",
    },
  ],

  "6stem": [
    {
      id: "bs-roformer-sw-6stem",
      name: "BS-RoFormer SW 6 Stem",
      stems: [
        "vocals",
        "drums",
        "bass",
        "guitar",
        "piano",
        "other",
      ],
      hfRepo: "elicwhite/bs-roformer-sw-6stem-onnx",
      description:
        "vocals・drums・bassに加えてguitarとpianoまで個別に分離できるRoFormer系6-stemモデル。高品質な詳細ステム分離向け。",
    },
    {
      id: "htdemucs-6s",
      name: "HTDemucs 6s",
      stems: [
        "vocals",
        "drums",
        "bass",
        "guitar",
        "piano",
        "other",
      ],
      hfRepo: "puar-playground/htdemucs",
      description:
        "HTDemucsの実験的6-stem版。guitarは比較的良好だが、公式にもpianoはbleedingやartifactが多いとされているため品質面では注意が必要。",
    },
  ],

  "multi-singer": [
    {
      id: "sepacap",
      name: "SepACap",
      stems: ["singer1", "singer2", "..."],
      hfRepo: "Tino3141/sepacap",
      description:
        "アカペラ楽曲に特化した複数歌手分離モデル。JaCappellaでfull-ensemble・subsetの両条件において論文上SOTAを報告。複数人ボーカルの分離に最も特化した候補。",
    },
    {
      id: "medleyvox",
      name: "MedleyVox / iSRNet",
      stems: ["singer1", "singer2", "..."],
      hfRepo: "Cyru5/MedleyVox",
      description:
        "duet・unison・N-singer separationを対象とした複数歌唱音声分離モデル。研究用ベースラインとして有用だが、Hugging FaceのModel Cardには統一された性能値が明記されていない。",
    },
  ],
};


function ModelSelector() {
  const [selectedModel, setSelectedModel] = useState<ModelId>('demucs'); // 初期値は'demucs'
  const [selectedSeparationMode, setSelectedSeparationMode] = useState<SeparationMode>('vocal'); // 初期値は'vocal'
}

function App() {
  return (
    <main>  {/* セマンティックhtmlだって。 */}
      <Header />
      <UploadArea />
    </main>
  )
}

export default App
// frontend/src/App.tsx
// アプリ全体の画面構成だけを置くファイルです。
// 個別の表示や入力処理は components 配下へ分け、差分を小さく保ちます。
import { Header } from './components/Header.tsx'
import { UploadArea } from './components/UploadArea.tsx'
import {SeparationSettings} from './components/SeparationSettings.tsx'

function App() {
  return (
    <main>
      {/* セマンティック HTML として、ページの主要コンテンツを main にまとめます。 */}
      <Header />
      <UploadArea />
      <SeparationSettings />
    </main>
  )
}

export default App

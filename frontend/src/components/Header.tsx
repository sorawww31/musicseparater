// frontend/src/components/Header.tsx
// アプリの説明ヘッダーだけを担当する表示コンポーネントです。
// 文言と改行は分離前の App.tsx と同じ内容を維持します。
export function Header() {
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

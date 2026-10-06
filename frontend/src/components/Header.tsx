// frontend/src/components/Header.tsx
// アプリの説明ヘッダーだけを担当する表示コンポーネントです。
// 対応済みの2ステム分離と複数歌声分離を短く案内します。
export function Header() {
  return (
    <header>
      <h1>AI音楽分離アプリケーション</h1>
      <p>
        このアプリケーションは、AIを使用して音楽トラックを分離することができます。<br />
        ボーカルと伴奏の分離に加え、blind分離や参照音声を使った対象歌手の抽出ができます。<br />
      </p>
    </header>
  )
}

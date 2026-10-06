// frontend/src/components/JobStatusPanel.tsx
// 非同期ジョブの段階・進捗・完成ステムを一箇所に表示します。
import type { SeparationJob } from '../api/audio.ts'
import { resolveApiUrl } from '../config.ts'

const phaseLabels = {
  queued: '処理待ち',
  extracting_vocals: 'ボーカルを抽出中',
  separating_singers: '歌声を分離中',
  finalizing: 'ファイルを仕上げ中',
  completed: '完了',
  failed: '失敗',
} as const

const stemLabels: Record<string, string> = {
  singer_1: 'Singer 1',
  singer_2: 'Singer 2',
  lead_vocal: 'Lead Vocal',
  finger_snap: 'Finger Snap',
  vocal_percussion: 'Vocal Percussion',
  instrumental: 'Instrumental',
  vocals: 'Vocals',
  target_vocal: 'Target Vocal',
  residual: 'Other Singer + Instrumental',
}

export function JobStatusPanel({ job }: { job: SeparationJob | null }) {
  if (!job) return null

  return (
    <section className="panel job-panel" aria-live="polite">
      <div className="job-heading">
        <h2>{phaseLabels[job.phase]}</h2>
        <strong>{job.progress_percent}%</strong>
      </div>
      <progress max="100" value={job.progress_percent} />
      {job.error && <p className="error-message">{job.error}</p>}
      {job.stems && (
        <div className="stem-list">
          {Object.entries(job.stems).map(([stem, path]) => {
            const url = resolveApiUrl(path)
            return (
              <article className="stem" key={stem}>
                <h3>{stemLabels[stem] || stem.replaceAll('_', ' ')}</h3>
                <audio controls preload="none" src={url} />
                <a href={`${url}?download=true`}>WAVをダウンロード</a>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}

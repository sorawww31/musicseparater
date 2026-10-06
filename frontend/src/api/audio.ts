// frontend/src/api/audio.ts
// アップロード・ジョブ作成・進捗取得のHTTP契約を型付きで提供します。
import { apiUrl, resolveApiUrl } from '../config.ts'

export type SeparationStatus = 'queued' | 'running' | 'completed' | 'failed'
export type SeparationPhase =
  | 'queued'
  | 'extracting_vocals'
  | 'separating_singers'
  | 'finalizing'
  | 'completed'
  | 'failed'

export type UploadResponse = {
  audio_id: string
  original_url: string
}

export type CreatedSeparation = {
  job_id: string
  status: 'queued'
  status_url: string
}

export type SeparationJob = {
  job_id: string
  status: SeparationStatus
  phase: SeparationPhase
  progress_percent: number
  stems: Record<string, string> | null
  error: string | null
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail || `API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export async function uploadAudio(file: File, signal?: AbortSignal): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('uploadfile', file)
  const response = await fetch(`${apiUrl}/audios`, { method: 'POST', body: formData, signal })
  return parseResponse<UploadResponse>(response)
}

export async function createSeparation(
  audioId: string,
  modelId: string,
  signal?: AbortSignal,
  referenceAudioId?: string,
): Promise<CreatedSeparation> {
  const numVocals = modelId === 'unmixx' ? 2 : undefined
  const response = await fetch(`${apiUrl}/separations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_id: audioId,
      model_id: modelId,
      num_vocals: numVocals,
      reference_audio_id: referenceAudioId,
    }),
    signal,
  })
  return parseResponse<CreatedSeparation>(response)
}

export async function getSeparationJob(
  statusPath: string,
  signal?: AbortSignal,
): Promise<SeparationJob> {
  const response = await fetch(resolveApiUrl(statusPath), { signal })
  return parseResponse<SeparationJob>(response)
}

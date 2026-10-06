// frontend/src/config.ts: フロントエンドで共有する変更可能な既定値を集約します。
// API の origin とレスポンス内の相対パスを安全に結合し、配信元の分離に対応します。

export const acceptedAudioFileTypes = '.mp3,.wav,.flac'
export const defaultSeparationMode = '2stem' as const
export const pollingIntervalMs = 1000

const defaultApiUrl = 'http://localhost:8000'

export const apiUrl = (import.meta.env?.VITE_API_URL || defaultApiUrl).replace(/\/+$/, '')

export function resolveApiUrl(path: string, baseUrl = apiUrl): string {
  const normalizedBaseUrl = baseUrl.replace(/\/+$/, '')
  const normalizedPath = path.startsWith('/') ? path : `/${path}`

  return `${normalizedBaseUrl}${normalizedPath}`
}

import type { Answer, Backend, Health, Language, ModelList } from './types'

const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, init)
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(typeof data?.detail === 'string' ? data.detail : 'Chưa nhận được phản hồi hợp lệ từ máy chủ. Vui lòng thử lại sau.')
  }
  if (!data) throw new Error('Chưa nhận được phản hồi hợp lệ từ máy chủ. Vui lòng thử lại sau.')
  return data
}

export const getHealth = (signal: AbortSignal) => request<Health>('/api/health', { signal })
export const getModels = (signal: AbortSignal) => request<ModelList>('/api/models', { signal })

export async function ask(question: string, language: Language, backend: Backend): Promise<Answer> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 600_000)
  try {
    return await request<Answer>('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, language, backend }), signal: controller.signal,
    })
  } finally {
    window.clearTimeout(timeout)
  }
}

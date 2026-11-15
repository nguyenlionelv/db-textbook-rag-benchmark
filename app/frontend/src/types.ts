export type Backend = 'local' | 'api'
export type Language = 'vi' | 'en'

export interface Source {
  source_id: string
  point_id: string
  text: string
  pdf_pages: number[]
  chapter: string
  section: string
  title: string
}

export interface Answer {
  request_id: string
  query: string
  language: Language
  backend: Backend
  model: string
  status: 'answered' | 'insufficient_evidence' | 'error'
  answer: string
  citations: string[]
  sources: Source[]
  retrieval_seconds: number
  generation_seconds: number
  total_seconds: number
  error: string | null
}

export interface ModelOption {
  backend: Backend
  model: string
  available: boolean
  reason: string | null
}

export interface Health {
  status: 'starting' | 'ready' | 'unavailable' | 'examples'
  ready: boolean
  busy: boolean
}

export interface ModelList {
  models: ModelOption[]
  default_backend: Backend | null
}

export interface Example {
  title: string
  topic: string
  recorded_at: string
  result: Answer
}

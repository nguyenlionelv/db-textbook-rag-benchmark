import type { Backend, Language, ModelOption } from '../types'

interface Props {
  backend: Backend
  language: Language
  models: ModelOption[]
  pending: boolean
  onBackend: (backend: Backend) => void
  onLanguage: (language: Language) => void
}

export default function Sidebar({ backend, language, models, pending, onBackend, onLanguage }: Props) {
  return <aside className="sidebar" aria-label="Tùy chỉnh">
    <div className="setting">
      <label htmlFor="backend">Mô hình trả lời</label>
      <select id="backend" value={backend} disabled={pending} onChange={event => onBackend(event.target.value as Backend)}>
        {(['local', 'api'] as const).map(kind => {
          const option = models.find(model => model.backend === kind)
          return <option key={kind} value={kind} disabled={!option?.available}>
            {kind === 'local' ? 'Local' : 'API'}
            {option?.model ? ` · ${option.model}` : ''}
            {option?.available ? '' : ' · chưa sẵn sàng'}
          </option>
        })}
      </select>
    </div>
    <div className="setting">
      <label htmlFor="language">Ngôn ngữ trả lời</label>
      <select id="language" value={language} disabled={pending} onChange={event => onLanguage(event.target.value as Language)}>
        <option value="vi">Tiếng Việt</option>
        <option value="en">English</option>
      </select>
    </div>
  </aside>
}

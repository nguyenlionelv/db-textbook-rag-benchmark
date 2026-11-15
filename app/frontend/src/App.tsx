import { ArrowUp, LoaderCircle } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { ask, getHealth, getModels } from './api'
import AnswerCard from './components/AnswerCard'
import Sidebar from './components/Sidebar'
import SourcePanel from './components/SourcePanel'
import type { Answer, Backend, Health, Language, ModelOption } from './types'

export default function App() {
  const [backend, setBackend] = useState<Backend>('local')
  const [language, setLanguage] = useState<Language>('vi')
  const [models, setModels] = useState<ModelOption[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [history, setHistory] = useState<Answer[]>([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState('')
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Answer | null>(null)
  const [activeSource, setActiveSource] = useState<string | null>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const end = useRef<HTMLDivElement>(null)
  const sending = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    async function connect() {
      try {
        const [status, options] = await Promise.all([getHealth(controller.signal), getModels(controller.signal)])
        if (controller.signal.aborted) return
        setHealth(status)
        setModels(options.models)
        setOffline(false)
        setBackend(current => options.models.some(model => model.backend === current && model.available)
          ? current : options.default_backend || current)
      } catch {
        if (!controller.signal.aborted) setOffline(true)
      }
    }
    void connect()
    const timer = window.setInterval(() => void connect(), 5000)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [refresh])

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [history.length, pending])

  const ready = !offline && !!health?.ready && !!models.find(model => model.backend === backend)?.available
  const statusLabel = offline ? 'Chưa kết nối' : !health || health.status === 'starting'
    ? 'Đang khởi động' : ready ? 'Sẵn sàng' : 'Chưa sẵn sàng'

  function reset() {
    setHistory([])
    setSelected(null)
    setActiveSource(null)
    setDraft('')
    setError('')
    input.current?.focus()
  }

  function openSource(result: Answer, id: string) {
    setSelected(result)
    setActiveSource(id)
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const question = draft.trim()
    if (!question || sending.current || !ready) return
    sending.current = true
    setPending(true)
    setPendingQuestion(question)
    setError('')
    try {
      const result = await ask(question, language, backend)
      if (result.status === 'error') throw new Error(result.error || 'Không xử lý được câu hỏi. Vui lòng thử lại.')
      setHistory(current => [...current, result])
      setDraft('')
      setSelected(result)
      setActiveSource(null)
    } catch (failure) {
      const network = failure instanceof TypeError || (failure instanceof DOMException && failure.name === 'AbortError')
      setError(network ? 'Không nhận được phản hồi. Kiểm tra kết nối trước khi thử lại.'
        : failure instanceof Error ? failure.message : 'Không xử lý được câu hỏi. Vui lòng thử lại.')
    } finally {
      sending.current = false
      setPending(false)
      setPendingQuestion('')
    }
  }

  return <div className="app-shell">
    <Sidebar backend={backend} language={language} models={models} pending={pending}
      onBackend={setBackend} onLanguage={setLanguage} />
    <div className="workspace">
      <header className="topbar">
        <h1>Hỏi đáp</h1>
        <div className="workspace-actions">
          <span className={`connection-status ${ready ? 'connection-ready' : ''}`} role="status">{statusLabel}</span>
          {offline ? <button onClick={() => setRefresh(value => value + 1)}>Thử lại</button> : null}
          {history.length ? <button onClick={reset} disabled={pending}>Xóa hội thoại</button> : null}
        </div>
      </header>
      <div className="workspace-body">
        <main className="main-panel">
          <div className="conversation-scroll">
            {history.map(result => <AnswerCard key={result.request_id} result={result}
              onSource={id => openSource(result, id)} />)}
            {pending ? <div className="pending-exchange">
              <div className="user-question"><p>{pendingQuestion}</p></div>
              <div className="pending-answer" role="status"><LoaderCircle size={17} className="spin" />Đang trả lời…</div>
            </div> : null}
            <div ref={end} />
          </div>
          <div className="composer-area">
            {error ? <div className="error-notice" role="alert">{error}</div> : null}
            <form className="composer" onSubmit={submit}>
              <label className="sr-only" htmlFor="question">Câu hỏi của bạn</label>
              <textarea id="question" ref={input} value={draft} onChange={event => setDraft(event.target.value)}
                placeholder="Nhập câu hỏi…" maxLength={1000} rows={2} disabled={pending}
                onKeyDown={event => {
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault()
                    event.currentTarget.form?.requestSubmit()
                  }
                }} />
              <button className="send-button" type="submit" disabled={!draft.trim() || pending || !ready} aria-label="Gửi câu hỏi">
                {pending ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={19} />}
              </button>
            </form>
          </div>
        </main>
        <SourcePanel sources={selected?.sources || []} citations={selected?.citations || []}
          active={activeSource} onSelect={setActiveSource} onClose={() => setActiveSource(null)} />
      </div>
    </div>
  </div>
}

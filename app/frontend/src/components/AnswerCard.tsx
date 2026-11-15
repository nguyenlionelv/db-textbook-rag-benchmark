import { Check, Copy } from 'lucide-react'
import { useState } from 'react'
import type { Answer } from '../types'

interface Props {
  result: Answer
  onSource: (sourceId: string) => void
}

export default function AnswerCard({ result, onSource }: Props) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(result.answer)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return <article className="exchange">
    <div className="user-question"><p>{result.query}</p></div>
    <div className="answer-card">
      <div className="answer-heading">{result.model}</div>
      {result.status === 'insufficient_evidence' ? <p className="insufficient">Chưa đủ bằng chứng</p> : null}
      <div className="answer-text">{result.answer}</div>
      {result.citations.length ? <div className="citation-row">
        <span>Nguồn</span>
        {result.citations.map(id => <button key={id} onClick={() => onSource(id)} aria-label={`Mở nguồn ${id}`}>
          {id}
        </button>)}
      </div> : null}
      {result.sources.length ? <button className="context-button" onClick={() => onSource(result.sources[0].source_id)}>
        Xem ngữ cảnh ({result.sources.length})
      </button> : null}
      <div className="answer-bottom">
        <span className="timings">{result.total_seconds.toFixed(1)}s</span>
        <button className="icon-button" onClick={copy} aria-label={copied ? 'Đã sao chép' : 'Sao chép câu trả lời'}>
          {copied ? <Check size={16} /> : <Copy size={16} />}
        </button>
      </div>
    </div>
  </article>
}

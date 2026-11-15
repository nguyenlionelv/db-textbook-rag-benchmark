import { X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { Source } from '../types'

interface Props {
  sources: Source[]
  citations: string[]
  active: string | null
  onSelect: (id: string) => void
  onClose: () => void
}

export default function SourcePanel({ sources, citations, active, onSelect, onClose }: Props) {
  const detail = useRef<HTMLDivElement>(null)
  const selected = sources.find(source => source.source_id === active)

  useEffect(() => {
    if (selected) detail.current?.focus({ preventScroll: true })
  }, [selected])

  return <aside className={`source-panel ${selected ? 'source-panel-open' : ''}`} aria-label="Nguồn tham khảo">
    <div className="source-panel-heading">
      <h2>Nguồn tham khảo</h2>
      {sources.length ? <span className="source-count">{sources.length}</span> : null}
      <button className="icon-button mobile-close" onClick={onClose} aria-label="Đóng nguồn"><X size={18} /></button>
    </div>
    {sources.length ? <>
      <div className="source-list">
        {sources.map(source => <button
          className={`source-item ${source.source_id === active ? 'source-item-active' : ''}`}
          key={source.source_id}
          onClick={() => onSelect(source.source_id)}
          aria-pressed={source.source_id === active}
        >
          <span className="source-item-top">
            <span className="source-label">{source.source_id}</span>
            <span>Trang {source.pdf_pages.join(', ')}</span>
            {citations.includes(source.source_id) ? <span className="source-cited">Trích dẫn</span> : null}
          </span>
          <strong>{source.title}</strong>
          <span className="source-location">
            {source.chapter ? `Chương ${source.chapter}` : ''}
            {source.section ? ` · Mục ${source.section}` : ''}
          </span>
        </button>)}
      </div>
      {selected ? <div className="source-detail" ref={detail} tabIndex={-1}>
        <h3>Đoạn nguồn {selected.source_id}</h3>
        <p>{selected.text}</p>
        <span className="source-page">Trang PDF {selected.pdf_pages.join(', ')}</span>
      </div> : null}
    </> : <p className="source-empty">Chưa có nguồn.</p>}
  </aside>
}

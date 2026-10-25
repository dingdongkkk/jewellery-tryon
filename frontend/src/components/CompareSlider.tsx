import { useState } from 'react'

export function CompareSlider({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50)
  return (
    <div className="compare">
      <div className="compare-box">
        <img src={before} alt="Before" draggable={false} />
        <img src={after} alt="After" draggable={false} className="after" style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <div className="compare-line" style={{ left: `${pos}%` }} />
        <span className="tag left">Before</span>
        <span className="tag right">After</span>
      </div>
      <input type="range" min={0} max={100} value={pos} onChange={(e) => setPos(+e.target.value)} aria-label="Compare" />
    </div>
  )
}

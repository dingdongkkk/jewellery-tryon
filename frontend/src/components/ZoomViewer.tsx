import { useCallback, useEffect, useRef, useState } from 'react'

type Overlay = { url: string; region: [number, number, number, number] }

/** Wheel zoom and drag-pan. "100%" shows true image pixels.
 *  overlays: high-res tiles (same composition, rendered at up to 8x) laid exactly over their region,
 *  so zooming into the jewellery shows the product photo's real detail instead of upscaled pixels. */
export function ZoomViewer({
  src,
  alt,
  height = '68vh',
  overlays = [],
  focusRegions = [],
}: {
  src: string
  alt: string
  height?: string
  overlays?: Overlay[]
  focusRegions?: [number, number, number, number][]
}) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null)
  const [view, setView] = useState({ s: 1, x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null)

  const fitView = useCallback(() => {
    const box = boxRef.current
    if (!box || !nat) return
    const s = Math.min(box.clientWidth / nat.w, box.clientHeight / nat.h)
    setView({ s, x: (box.clientWidth - nat.w * s) / 2, y: (box.clientHeight - nat.h * s) / 2 })
  }, [nat])

  useEffect(() => fitView(), [fitView])
  useEffect(() => {
    const box = boxRef.current
    if (!box) return
    const observer = new ResizeObserver(fitView)
    observer.observe(box)
    return () => observer.disconnect()
  }, [fitView])

  const zoomAt = (factor: number, px: number, py: number) =>
    setView((v) => {
      const s = Math.min(Math.max(v.s * factor, 0.02), 16)
      const f = s / v.s
      return { s, x: px - (px - v.x) * f, y: py - (py - v.y) * f }
    })

  useEffect(() => {
    const box = boxRef.current
    if (!box) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = box.getBoundingClientRect()
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top)
    }
    box.addEventListener('wheel', onWheel, { passive: false })
    return () => box.removeEventListener('wheel', onWheel)
  }, [])

  const centre = () => {
    const b = boxRef.current!
    return [b.clientWidth / 2, b.clientHeight / 2] as const
  }

  const focus = (r: [number, number, number, number]) => {
    const box = boxRef.current
    if (!box) return
    const s = Math.min(2, box.clientWidth * 0.85 / r[2], box.clientHeight * 0.85 / r[3])
    setView({ s, x: box.clientWidth / 2 - (r[0] + r[2] / 2) * s,
      y: box.clientHeight / 2 - (r[1] + r[3] / 2) * s })
  }

  return (
    <div className="zoom">
      <div
        className="zoom-box"
        ref={boxRef}
        style={{ height }}
        onPointerDown={(e) => {
          ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (d) setView((v) => ({ ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y }))
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onDoubleClick={fitView}
      >
        <div
          className="zoom-layer"
          style={{
            width: nat?.w,
            height: nat?.h,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})`,
          }}
        >
          <img
            src={src}
            alt={alt}
            draggable={false}
            onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          />
          {nat &&
            overlays.map((o) => (
              <img
                key={o.url}
                src={o.url}
                alt=""
                draggable={false}
                className="zoom-tile"
                style={{ left: o.region[0], top: o.region[1], width: o.region[2], height: o.region[3] }}
              />
            ))}
        </div>
      </div>
      <div className="zoom-bar">
        <button onClick={() => zoomAt(1 / 1.5, ...centre())}>−</button>
        <span>{Math.round(view.s * 100)}%</span>
        <button onClick={() => zoomAt(1.5, ...centre())}>+</button>
        <button onClick={() => zoomAt(1 / view.s, ...centre())}>100% pixels</button>
        <button onClick={fitView}>Fit</button>
        {focusRegions.map((r, i) => <button key={i} onClick={() => focus(r)}>Inspect piece {i + 1}</button>)}
        {nat && (
          <span className="muted">
            {nat.w}×{nat.h}px{view.s > 1 && ' · enlarged beyond image pixels'}
          </span>
        )}
      </div>
    </div>
  )
}

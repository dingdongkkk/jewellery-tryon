import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Chain, PlacementItem, Product, Upload } from '../api'

type Props = {
  upload: Upload
  product: Product
  items: PlacementItem[]
  onChange: (items: PlacementItem[]) => void
  active: number
  setActive: (i: number) => void
  chain?: Chain
  onChainChange?: (c: Chain) => void
  occluder?: [number, number][]
  showOverlay?: boolean
}

const ZOOMS = [1, 2, 4]

/** Drag / resize / rotate the real product cutout over the photo. All maths in photo pixels. */
export function PlacementEditor({ upload, product, items, onChange, active, setActive, chain, onChainChange, occluder, showOverlay = true }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [fit, setFit] = useState(1)
  const [zoom, setZoom] = useState(1)
  const drag = useRef<{ i: number; x: number; y: number; cx: number; cy: number } | null>(null)
  const aspect = product.dimensions.cutout_h / product.dimensions.cutout_w

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const update = () => {
      const maxH = window.innerHeight * 0.72
      setFit(Math.min(el.clientWidth / upload.width, maxH / upload.height))
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [upload.width, upload.height])

  const scale = fit * zoom

  // keep the active item in view when zooming
  useEffect(() => {
    const el = wrapRef.current
    const it = items[active]
    if (!el || !it) return
    el.scrollTo({ left: it.cx * scale - el.clientWidth / 2, top: it.cy * scale - el.clientHeight / 2 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom])

  const update = (i: number, patch: Partial<PlacementItem>) => {
    const prev = items[i]
    onChange(items.map((it, k) => (k === i ? { ...it, ...patch } : it)))
    // the chain follows the pendant: its middle (the bail) moves fully, the ends at the neck stay put
    if (chain && onChainChange && i === 0 && (patch.cx !== undefined || patch.cy !== undefined)) {
      const dx = (patch.cx ?? prev.cx) - prev.cx
      const dy = (patch.cy ?? prev.cy) - prev.cy
      const n = chain.points.length - 1
      onChainChange({
        ...chain,
        points: chain.points.map(([x, y], k) => {
          const w = 1 - Math.abs(k - n / 2) / (n / 2)
          return [x + dx * w, y + dy * w] as [number, number]
        }),
      })
    }
  }

  const onPointerDown = (e: React.PointerEvent, i: number) => {
    e.preventDefault()
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    setActive(i)
    drag.current = { i, x: e.clientX, y: e.clientY, cx: items[i].cx, cy: items[i].cy }
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const cx = Math.min(upload.width, Math.max(0, d.cx + (e.clientX - d.x) / scale))
    const cy = Math.min(upload.height, Math.max(0, d.cy + (e.clientY - d.y) / scale))
    update(d.i, { cx, cy })
  }
  const onPointerUp = () => (drag.current = null)

  const onWheel = (e: React.WheelEvent, i: number) => {
    if (!e.shiftKey && !e.altKey) return // plain wheel scrolls the page / zoomed view
    e.preventDefault()
    const it = items[i]
    if (e.altKey) update(i, { rotation: Math.round(it.rotation + Math.sign(e.deltaY) * 2) })
    else update(i, { width: Math.max(8, it.width * (e.deltaY > 0 ? 0.96 : 1.04)) })
  }

  const onKey = (e: React.KeyboardEvent) => {
    const it = items[active]
    if (!it) return
    const step = (e.shiftKey ? 10 : 1) / Math.max(fit, 0.01) / zoom
    const moves: Record<string, Partial<PlacementItem>> = {
      ArrowLeft: { cx: it.cx - step },
      ArrowRight: { cx: it.cx + step },
      ArrowUp: { cy: it.cy - step },
      ArrowDown: { cy: it.cy + step },
    }
    if (moves[e.key]) {
      e.preventDefault()
      update(active, moves[e.key])
    }
  }

  const it = items[active]
  return (
    <div className="editor">
      <div
        className="stage-wrap"
        ref={wrapRef}
        tabIndex={0}
        onKeyDown={onKey}
        style={{ maxHeight: '72vh', overflow: zoom > 1 ? 'auto' : 'hidden' }}
      >
        <div
          className="stage"
          style={{ width: upload.width * scale, height: upload.height * scale }}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <img src={upload.url} className="stage-photo" draggable={false} alt="Your photo" />
          {showOverlay && chain && (
            <svg className="chain-preview" width={upload.width * scale} height={upload.height * scale}>
              <polyline
                points={chain.points.map(([x, y]) => `${x * scale},${y * scale}`).join(' ')}
                fill="none"
                stroke="#c9a14a"
                strokeWidth={Math.max(1, chain.thickness * scale)}
                strokeLinecap="round"
              />
            </svg>
          )}
          {showOverlay && items.map((p, i) => {
            const w = p.width * scale
            const h = w * aspect
            return (
              <img
                key={i}
                src={product.cutout_url}
                draggable={false}
                alt={product.name}
                className={`cutout ${i === active ? 'active' : ''}`}
                onPointerDown={(e) => onPointerDown(e, i)}
                onWheel={(e) => onWheel(e, i)}
                style={{
                  left: p.cx * scale - w / 2,
                  top: p.cy * scale - h / 2,
                  width: w,
                  height: h,
                  transform: `rotate(${p.rotation}deg) scaleX(${p.flip ? -1 : 1})`,
                }}
              />
            )
          })}
          {showOverlay && occluder && (
            // the face is in front of the earlobes: redraw it over the earrings
            <img
              src={upload.url}
              className="stage-photo occluder"
              draggable={false}
              alt=""
              style={{ clipPath: `polygon(${occluder.map(([x, y]) => `${x * scale}px ${y * scale}px`).join(',')})` }}
            />
          )}
        </div>
      </div>

      {showOverlay && <div className="controls">
        <div className="row">
          <span className="label">View</span>
          {ZOOMS.map((z) => (
            <button key={z} className={`chip ${zoom === z ? 'on' : ''}`} onClick={() => setZoom(z)}>
              {z}×
            </button>
          ))}
          <span className="hint">Placed automatically. Drag to fine-tune if you like.</span>
        </div>
        {items.length > 1 && (
          <div className="row">
            <span className="label">Editing</span>
            {items.map((_, i) => (
              <button key={i} className={`chip ${active === i ? 'on' : ''}`} onClick={() => setActive(i)}>
                {i === 0 ? 'First' : 'Second'} {product.category}
              </button>
            ))}
          </div>
        )}
        {it && (
          <>
            <label className="row">
              <span className="label">Size</span>
              <input
                type="range"
                min={Math.max(8, upload.width * 0.01)}
                max={upload.width * 0.6}
                step={0.5}
                value={it.width}
                onChange={(e) => update(active, { width: +e.target.value })}
              />
              <span className="val">{Math.round(it.width)} px</span>
            </label>
            <label className="row">
              <span className="label">Rotate</span>
              <input
                type="range"
                min={-60}
                max={60}
                step={0.5}
                value={it.rotation}
                onChange={(e) => update(active, { rotation: +e.target.value })}
              />
              <span className="val">{it.rotation.toFixed(1)}°</span>
            </label>
            <div className="row">
              <label className="check">
                <input type="checkbox" checked={it.flip} onChange={(e) => update(active, { flip: e.target.checked })} />
                Mirror
              </label>
              {product.category === 'earring' && items.length === 1 && (
                <button
                  className="ghost"
                  onClick={() => {
                    onChange([...items, { ...it, cx: upload.width - it.cx, rotation: -it.rotation, flip: !it.flip }])
                    setActive(1)
                  }}
                >
                  + Add the other earring
                </button>
              )}
              {items.length > 1 && (
                <button
                  className="ghost"
                  onClick={() => {
                    onChange(items.filter((_, k) => k !== active))
                    setActive(0)
                  }}
                >
                  Remove this one
                </button>
              )}
            </div>
            <p className="hint">
              Drag the piece to move it · Shift + scroll to resize · Alt + scroll to rotate · arrow keys nudge.
            </p>
          </>
        )}
      </div>}
    </div>
  )
}

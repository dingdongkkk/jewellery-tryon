import { useEffect, useRef, useState } from 'react'
import { api, type Category, type Chain, type Job, type Placement, type PlacementItem, type Product, type Upload } from './api'
import { CameraCapture } from './components/CameraCapture'
import { CompareSlider } from './components/CompareSlider'
import { PlacementEditor } from './components/PlacementEditor'
import { ZoomViewer } from './components/ZoomViewer'

const STAGES: { id: Job['status']; label: string }[] = [
  { id: 'queued', label: 'Queued' },
  { id: 'preparing', label: 'Preparing' },
  { id: 'generating', label: 'AI generating' },
  { id: 'compositing', label: 'Detail-lock composite' },
  { id: 'complete', label: 'Done' },
]

type Tab = 'compare' | 'zoom' | 'detail' | 'ai'

export default function App() {
  const [products, setProducts] = useState<Product[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [category, setCategory] = useState('earring')
  const [samples, setSamples] = useState<string[]>([])
  const [upload, setUpload] = useState<Upload | null>(null)
  const [product, setProduct] = useState<Product | null>(null)
  const [items, setItems] = useState<PlacementItem[]>([])
  const [chain, setChain] = useState<Chain | undefined>(undefined)
  const [occluder, setOccluder] = useState<[number, number][] | undefined>(undefined)
  const userChoseNoAI = useRef(false)
  const [outputSize, setOutputSize] = useState(6144)
  const resultRef = useRef<HTMLDivElement>(null)
  const [placing, setPlacing] = useState(false)
  const [aiAvailable, setAiAvailable] = useState(true)
  const [active, setActive] = useState(0)
  const [job, setJob] = useState<Job | null>(null)
  const [provider, setProvider] = useState('qwen-fast')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('compare')
  const [notice, setNotice] = useState<string | null>(null)
  const poll = useRef<number | null>(null)

  useEffect(() => {
    api.products().then((r) => {
      setProducts(r.products)
      setCategories(r.categories)
    }, (e) => setError(`Backend not reachable: ${e.message}`))
    api.samples().then((r) => setSamples(r.samples), () => {})
    api.health().then((h) => {
      // the server picks the default AI provider; nothing without one
      const ok = h.fal_key || h.hf_token
      setAiAvailable(ok)
      setProvider(ok ? h.provider : 'none')
    }, () => {})
    return () => {
      if (poll.current) window.clearInterval(poll.current)
    }
  }, [])

  const autoPlace = async (u: Upload, p: Product) => {
    setPlacing(true)
    setError(null)
    setJob(null)
    try {
      // re-check every time: the token may have been added (or quota reset) since the page loaded
      let prov = provider
      try {
        const h = await api.health()
        const ok = h.fal_key || h.hf_token
        setAiAvailable(ok)
        prov = !ok ? 'none' : provider === 'none' && !userChoseNoAI.current ? h.provider : provider
        setProvider(prov)
      } catch {
        /* keep current provider */
      }
      const r = await api.autoplace(u.id, p.id)
      setItems(r.items)
      setChain(r.chain)
      setOccluder(r.occluder)
      setActive(0)
      const where =
        p.category === 'earring'
          ? 'at your ears'
          : p.category === 'pendant'
            ? 'around your neck'
            : 'on your ring finger'
      setNotice(`Found the spot ${where}. Press “Generate AI try-on”.`)
      void prov // generation is started by the "Generate AI try-on" button (each AI image costs money)
    } catch (e) {
      setItems([])
      setChain(undefined)
      setOccluder(undefined)
      setError((e as Error).message)
    } finally {
      setPlacing(false)
    }
  }

  const choosePhoto = async (blob: Blob, name: string) => {
    setError(null)
    setBusy(true)
    try {
      const u = await api.upload(blob, name)
      setUpload(u)
      setJob(null)
      if (product) await autoPlace(u, product)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const pickProduct = (p: Product) => {
    if (!p.supported) return
    setProduct(p)
    if (upload) autoPlace(upload, p)
  }

  const startPolling = (id: string) => {
    if (poll.current) window.clearInterval(poll.current)
    poll.current = window.setInterval(async () => {
      try {
        const j = await api.getTryon(id)
        setJob(j)
        if (['complete', 'failed', 'quota_exceeded'].includes(j.status)) {
          window.clearInterval(poll.current!)
          poll.current = null
          if (j.status === 'complete') {
            setTab('zoom')
            setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
          }
          if (j.status === 'quota_exceeded') {
            // keep going without the AI pass: light-matched composite, instant
            setAiAvailable(false)
            setNotice('The AI provider quota or balance is used up. This result uses local lighting instead.')
            const r = await api.createTryon({
              upload_id: j.upload_id,
              product_id: j.product_id,
              placement: j.placement,
              provider: 'none',
              settings: { output_long_side: Number(j.settings.output_long_side ?? 6144) },
            })
            setJob(r)
            startPolling(r.id)
          }
        }
      } catch (e) {
        setError((e as Error).message)
      }
    }, 1200)
  }

  const tryOn = async (prov = provider, placement?: Placement, u = upload, p = product) => {
    const pl = placement ?? { items, chain }
    if (!u || !p || !pl.items.length) return
    setError(null)
    try {
      const j = await api.createTryon({ upload_id: u.id, product_id: p.id, placement: pl, provider: prov, settings: { output_long_side: outputSize } })
      setJob(j)
      startPolling(j.id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const deleteResult = async () => {
    if (!job) return
    const r = await api.deleteTryon(job.id)
    setJob(null)
    if (r.upload_deleted) {
      setUpload(null)
      setItems([])
      setChain(undefined)
    }
    setNotice(r.upload_deleted ? 'Deleted the result and your uploaded photo from this server.' : 'Deleted the result.')
  }

  const deletePhoto = async () => {
    if (!upload) return
    await api.deleteUpload(upload.id)
    setUpload(null)
    setJob(null)
    setItems([])
    setChain(undefined)
    setNotice('Deleted your photo and every result made from it.')
  }

  const running = job && !['complete', 'failed', 'quota_exceeded'].includes(job.status)
  const stageIdx = job ? STAGES.findIndex((s) => s.id === job.status) : -1
  const shown = products.filter((p) => p.category === category)

  return (
    <div className="app">
      <header>
        <div>
          <h1>Jewellery Try-On</h1>
          <p className="sub">Try it on. Look closer. Original jewellery detail, with lighting matched to your photo.</p>
        </div>
      </header>

      <section className="privacy">
        <strong>Privacy:</strong> AI lighting sends a close crop around each piece to your selected
        image service. Local lighting keeps processing on this server. Your photo and results are stored on
        this server until you delete them. You can delete them at any time with the buttons below.
      </section>

      {error && <div className="error">{error}</div>}
      {notice && <div className="notice">{notice}</div>}

      <div className="grid">
        <div className="panel">
          <h2>1 · Your photo</h2>
          <p className="hint">
            Earrings: front-facing with ears visible. Pendants: neck and upper chest visible. Rings: back of your hand,
            fingers spread.
          </p>
          <label className="upload">
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => e.target.files?.[0] && choosePhoto(e.target.files[0], e.target.files[0].name)}
            />
            {busy ? 'Uploading…' : upload ? 'Choose a different photo' : 'Upload a photo'}
          </label>
          <button className="camera-btn" onClick={() => setCameraOpen(true)} disabled={busy}>
            📷 Take a photo with your camera
          </button>
          {cameraOpen && (
            <CameraCapture
              onClose={() => setCameraOpen(false)}
              onCapture={(blob) => {
                setCameraOpen(false)
                choosePhoto(blob, 'camera.jpg')
              }}
            />
          )}
          {samples.length > 0 && (
            <div className="samples">
              <span className="hint">or try a sample:</span>
              {samples.map((s) => (
                <button
                  key={s}
                  className="sample"
                  onClick={async () => choosePhoto(await (await fetch(s)).blob(), s.split('/').pop()!)}
                >
                  <img src={s} alt="sample" />
                </button>
              ))}
            </div>
          )}
          {upload && (
            <p className="hint">
              {upload.width}×{upload.height}px ·{' '}
              <button className="link" onClick={deletePhoto}>
                Delete my photo
              </button>
            </p>
          )}

          <h2>2 · Pick a piece</h2>
          <div className="tabs">
            {categories.map((c) => (
              <button
                key={c.id}
                disabled={!c.supported}
                title={c.note}
                className={`chip ${category === c.id ? 'on' : ''}`}
                onClick={() => setCategory(c.id)}
              >
                {c.label}
                {!c.supported && <small> · soon</small>}
              </button>
            ))}
          </div>
          <p className="hint">Pick a piece and it's placed on you automatically. Bracelets aren't supported yet.</p>
          <div className="products">
            {shown.map((p) => (
              <button
                key={p.id}
                className={`product ${product?.id === p.id ? 'on' : ''}`}
                onClick={() => pickProduct(p)}
                title={p.credit}
              >
                <img src={p.thumb_url} alt={p.name} />
                <span>{p.name}</span>
              </button>
            ))}
          </div>
          {product && (
            <p className="hint">
              {product.credit} ·{' '}
              <a href={product.original_url} target="_blank" rel="noreferrer">
                full-res product photo
              </a>
            </p>
          )}

          <label className="row">
            <span className="label">Export detail</span>
            <select value={outputSize} onChange={(e) => setOutputSize(Number(e.target.value))} disabled={!!running}>
              <option value={4096}>4K · smaller file</option>
              <option value={6144}>6K · detailed (recommended)</option>
              <option value={8192}>8K · maximum detail</option>
            </select>
          </label>
          <p className="hint">Jewellery is rendered from the original product photo. Export size is limited by source detail and 64 megapixels; enlarging skin does not add new detail.</p>
          <details className="advanced">
            <summary>Advanced</summary>
            <label className="row">
              <span className="label">Provider</span>
              <select
                value={provider}
                onChange={(e) => {
                  userChoseNoAI.current = e.target.value === 'none'
                  setProvider(e.target.value)
                }}
              >
                <option value="fal-nano-banana">AI try-on (recommended)</option>
                <option value="fal-qwen-edit">AI try-on (alternative model)</option>
                <option value="qwen-fast">AI try-on (free, slower, daily limit)</option>
                <option value="none">Instant preview, no AI</option>
              </select>
            </label>
          </details>
        </div>

        <div className="panel wide">
          <h2>3 · Wear it</h2>
          {!aiAvailable && (
            <p className="hint">
              AI try-on isn&rsquo;t available right now. Results use the instant preview.
            </p>
          )}
          {placing && <p className="hint">Finding your {product?.category === 'ring' ? 'ring finger' : product?.category === 'pendant' ? 'neck' : 'ears'}…</p>}
          {upload && product && (
            <div className="generate">
              <button className="primary big" disabled={!!running || placing} onClick={() => tryOn()}>
                {running || placing ? 'Generating…' : job?.status === 'complete' ? '✨ Generate again' : '✨ Generate AI try-on'}
              </button>
              <span className="hint">
                {provider === 'none'
                  ? 'Uses local lighting; no photo is sent to an AI service.'
                  : provider.startsWith('fal')
                    ? 'The AI draws you wearing the piece. Usually 20–40 seconds.'
                    : 'Free AI service; queue times and daily limits apply.'}
              </span>
            </div>
          )}
          {job && (
            <div className="job" ref={resultRef}>
              <ol className="stages">
                {STAGES.map((s, i) => (
                  <li
                    key={s.id}
                    className={
                      job.status === s.id ? 'now' : stageIdx > i || job.status === 'complete' ? 'done' : ''
                    }
                  >
                    {s.id === 'generating' && job.provider === 'none' ? 'AI (skipped)' : s.label}
                  </li>
                ))}
              </ol>
              {job.status === 'queued' && job.queue_position ? <p className="hint">{job.queue_position} ahead of you</p> : null}
              {running && job.error && <p className="hint">{job.error}</p>}
              {job.status === 'generating' && (
                <p className="hint">
                  {job.provider.startsWith('fal')
                    ? 'The AI is putting the piece on you…'
                    : 'Waiting for a free AI server. Queues and daily limits apply.'}
                </p>
              )}
              {job.status === 'quota_exceeded' && (
                <div className="error">
                  {job.error}
                  <div>
                    <button className="link-plain" onClick={() => tryOn('none')}>
                      Instant preview without AI (pasted, won&rsquo;t look worn)
                    </button>
                  </div>
                </div>
              )}
              {job.status === 'failed' && (
                <div className="error">
                  {job.error}
                  <div>
                    <button className="primary" onClick={() => tryOn()}>
                      ✨ Generate again
                    </button>{' '}
                    <button className="link-plain" onClick={() => tryOn('none')}>
                      Instant preview without AI (pasted, won&rsquo;t look worn)
                    </button>
                  </div>
                </div>
              )}

              {job.status === 'complete' && upload && (
                <div className="result">
                  <div className="tabs">
                    {(
                      [
                        ['zoom', 'You wearing it (zoom in)'],
                        ['compare', 'Before / after'],
                        ['detail', 'Inspect texture'],
                        ...(job.assets.ai ? ([['ai', 'Raw AI image']] as [Tab, string][]) : []),
                      ] as [Tab, string][]
                    ).map(([id, label]) => (
                      <button key={id} className={`chip ${tab === id ? 'on' : ''}`} onClick={() => setTab(id)}>
                        {label}
                      </button>
                    ))}
                  </div>
                  {tab === 'compare' && (
                    <CompareSlider
                      before={upload.url}
                      after={job.assets.preview ?? job.assets.tryon!}
                    />
                  )}
                  {tab === 'zoom' && (
                    <ZoomViewer
                      src={job.assets.tryon!}
                      alt="Try-on result"
                      focusRegions={job.tiles.map((t) => t.region)}
                      overlays={job.timings?.info?.quality ? [] : job.tiles.map((t) => ({ url: t.url, region: t.region }))}
                    />
                  )}
                  {tab === 'detail' && (
                    <div className="pair">
                      <figure>
                        <ZoomViewer src={job.assets.detail!} alt="Close-up from the try-on" height="56vh" />
                        <figcaption>A pixel crop from the downloaded result</figcaption>
                      </figure>
                      <figure>
                        <ZoomViewer src={job.assets.reference!} alt="Original product photo" height="56vh" />
                        <figcaption>Original product photo</figcaption>
                      </figure>
                    </div>
                  )}
                  {tab === 'ai' && job.assets.ai && (
                    <figure className="ai-result">
                      <ZoomViewer src={job.assets.ai} alt="AI image of you wearing the piece" />
                      <figcaption>
                        The AI&rsquo;s own image before texture lock. In &ldquo;You wearing it&rdquo; the AI&rsquo;s drawing of the
                        piece is replaced by the real product photo (or a sharper AI detail pass), so zooming shows the real
                        texture.
                        {job.timings?.info?.ai_output === 'model framing (not aligned to the photo)' &&
                          ' The AI re-framed this shot, so it is shown as the AI drew it.'}
                      </figcaption>
                    </figure>
                  )}
                  <p className="hint">
                    The AI put the piece on you (placement, shadows, chain, hair in front); the jewellery pixels then
                    come from the real product photo{' '}
                    {(job.timings?.info?.worn as { pieces?: { mode: string }[] } | undefined)?.pieces?.some((p) => p.mode !== 'real product pixels')
                      ? 'where it matched, and from a high-res AI detail pass elsewhere'
                      : ''}
                    . Zoom in to see the texture.
                  </p>
                  {Boolean(job.timings?.info?.ai_rejected) && (
                    <p className="notice">
                      The AI image couldn&rsquo;t be used ({String(job.timings?.info?.ai_rejected)}), so this is the instant
                      non-AI version. Press &ldquo;Generate again&rdquo; for another attempt.
                    </p>
                  )}
                  <div className="downloads">
                    {job.assets.ai && (
                      <a className="button" href={job.assets.ai} download={`ai-tryon-${job.id}.png`}>
                        Download raw AI image
                      </a>
                    )}
                    <a className="button" href={job.assets.tryon} download={`tryon-${job.id}.png`}>
                      Download full-resolution PNG
                    </a>
                    <a className="button" href={job.assets.detail} download={`closeup-${job.id}.png`}>
                      Download close-up
                    </a>
                    <a className="button" href={job.assets.reference} download>
                      Product reference
                    </a>
                    <button className="danger" onClick={deleteResult}>
                      Delete result &amp; my photo
                    </button>
                  </div>
                  <p className="meta">
                    seed {job.seed}
                    {job.timings &&
                      ' · ' +
                        Object.entries(job.timings)
                          .filter(([k]) => k !== 'info')
                          .map(([k, v]) => `${k} ${v}s`)
                          .join(' · ')}
                  </p>
                </div>
              )}
            </div>
          )}
          {!upload || !product ? (
            <p className="empty">Upload a photo and pick a piece, then press “Generate AI try-on”.</p>
          ) : (
            <details className="placement" open={job?.status !== 'complete'}>
              <summary>{job?.status === 'complete' ? 'Adjust placement' : 'Your photo'}</summary>
              <PlacementEditor
                upload={upload}
                product={product}
                items={items}
                onChange={setItems}
                active={active}
                setActive={setActive}
                chain={chain}
                onChainChange={setChain}
                occluder={occluder}
                showOverlay={true}
              />
              <p className="hint">Check the attachment point before generating. Drag, rotate or resize the piece if needed; this placement is used for both lighting and the final detail.</p>
            </details>
          )}

        </div>
      </div>
    </div>
  )
}

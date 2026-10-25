export type Category = { id: string; label: string; supported: boolean; note?: string }

export type Product = {
  id: string
  name: string
  category: 'earring' | 'pendant' | 'ring' | 'bracelet'
  supported: boolean
  cutout_url: string
  thumb_url: string
  original_url: string
  dimensions: { cutout_w: number; cutout_h: number; approx_width_mm?: number }
  anchors: { attach: [number, number]; center: [number, number] }
  credit?: string
  source_url?: string
}

export type Upload = { id: string; width: number; height: number; url: string }

export type PlacementItem = {
  cx: number
  cy: number
  width: number
  rotation: number
  flip: boolean
  occlude?: Record<string, unknown>
}

export type Chain = { points: [number, number][]; thickness: number }

export type Placement = { items: PlacementItem[]; chain?: Chain; whole_photo?: boolean; face_oval?: [number, number][] }

export type AutoPlacement = Placement & { detected: Record<string, unknown>; occluder?: [number, number][] }

export type Tile = { file: string; url: string; region: [number, number, number, number]; scale: number; size: [number, number] }

export type JobStatus =
  | 'queued'
  | 'preparing'
  | 'generating'
  | 'compositing'
  | 'complete'
  | 'quota_exceeded'
  | 'failed'

export type Job = {
  id: string
  upload_id: string
  product_id: string
  status: JobStatus
  placement: Placement
  provider: string
  seed: number
  settings: Record<string, unknown>
  error: string | null
  attempts: number
  timings: (Record<string, number> & { info?: Record<string, unknown> }) | null
  assets: { tryon?: string; preview?: string | null; detail?: string; reference?: string; ai?: string | null }
  tiles: Tile[]
  debug: { preplaced?: string; ai_context?: string }
  queue_position?: number
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  return r.json()
}

const serverApi = {
  products: () => req<{ products: Product[]; categories: Category[] }>('/products'),
  samples: () => req<{ samples: string[] }>('/samples-list'),
  upload: (file: Blob, name = 'photo.jpg') => {
    const fd = new FormData()
    fd.append('file', file, name)
    return req<Upload>('/uploads', { method: 'POST', body: fd })
  },
  deleteUpload: (id: string) => req<{ deleted: boolean }>(`/uploads/${id}`, { method: 'DELETE' }),
  createTryon: (body: {
    upload_id: string
    product_id: string
    placement: Placement
    provider?: string
    seed?: number
    settings?: { output_long_side: number; detail_pass?: boolean }
  }) =>
    req<Job>('/tryons', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  autoplace: (upload_id: string, product_id: string) =>
    req<AutoPlacement>('/autoplace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id, product_id }),
    }),
  health: () => req<{ hf_token: boolean; fal_key: boolean; provider: string }>('/health'),
  getTryon: (id: string) => req<Job>(`/tryons/${id}`),
  deleteTryon: (id: string) => req<{ deleted: boolean; upload_deleted: boolean }>(`/tryons/${id}`, { method: 'DELETE' }),
}

// ---------------------------------------------------------------------------------------------------------------
// Serverless mode (Vercel): the photo stays in the browser and is sent with each request; landmarks are detected
// in the browser; the API is stateless and returns results directly. Built with VITE_STATELESS=1.

const STATELESS = import.meta.env.VITE_STATELESS === '1'
const PW_KEY = 'jewellery-tryon-password'
const local = {
  uploads: new Map<string, { dataUrl: string; url: string; width: number; height: number }>(),
  jobs: new Map<string, Job>(),
  products: [] as Product[],
}

function password() {
  try {
    return localStorage.getItem(PW_KEY) ?? ''
  } catch {
    return ''
  }
}

async function post<T>(path: string, body: unknown, retry = true): Promise<T> {
  const r = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-App-Password': password() },
    body: JSON.stringify(body),
  })
  if (r.status === 401 && retry) {
    const pw = window.prompt('This demo is password protected. Enter the password:')
    if (pw) {
      try {
        localStorage.setItem(PW_KEY, pw)
      } catch {
        /* private mode: ask again next time */
      }
      return post<T>(path, body, false)
    }
  }
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const b = await r.json()
      msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail ?? b)
    } catch {
      /* not json */
    }
    throw new Error(r.status === 401 ? 'Wrong password.' : msg)
  }
  return r.json()
}

/** Resize in the browser (fixes phone-photo rotation too): keeps requests under the 4.5 MB limit. */
async function toJpeg(file: Blob, maxSide = 2000) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' })
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height))
  const c = document.createElement('canvas')
  c.width = Math.round(bmp.width * k)
  c.height = Math.round(bmp.height * k)
  c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height)
  const dataUrl = c.toDataURL('image/jpeg', 0.9)
  const blob = await (await fetch(dataUrl)).blob()
  return { dataUrl, url: URL.createObjectURL(blob), width: c.width, height: c.height }
}

const statelessApi: typeof serverApi = {
  products: async () => {
    const r = await req<{ products: Product[]; categories: Category[] }>('/products')
    local.products = r.products
    return r
  },
  samples: async () => (await fetch('/samples/index.json')).json(),
  upload: async (file: Blob) => {
    const u = await toJpeg(file)
    if (Math.min(u.width, u.height) < 400) throw new Error('Photo is too small. Use at least 400 px on the short side.')
    const id = 'local_' + Math.random().toString(36).slice(2, 10)
    local.uploads.set(id, u)
    return { id, width: u.width, height: u.height, url: u.url }
  },
  deleteUpload: async (id: string) => {
    const u = local.uploads.get(id)
    if (u) URL.revokeObjectURL(u.url)
    local.uploads.delete(id)
    return { deleted: true }
  },
  autoplace: async (upload_id: string, product_id: string) => {
    const u = local.uploads.get(upload_id)
    const p = local.products.find((x) => x.id === product_id)
    if (!u || !p) throw new Error('Upload or product not found.')
    const { detectLandmarks } = await import('./landmarks')
    const landmarks = await detectLandmarks(u.url, p.category)
    return post<AutoPlacement>('/autoplace', { image: u.dataUrl, product_id, landmarks })
  },
  createTryon: async (body) => {
    const u = local.uploads.get(body.upload_id)
    if (!u) throw new Error('Upload not found.')
    const id = 'job_' + Math.random().toString(36).slice(2, 10)
    const pending: Job = {
      id, upload_id: body.upload_id, product_id: body.product_id, status: 'generating', placement: body.placement,
      provider: body.provider ?? '', seed: 0, settings: {}, error: null, attempts: 0, timings: null,
      assets: {}, tiles: [], debug: {},
    }
    local.jobs.set(id, pending)
    // the whole try-on runs inside this one request; the page polls the local copy meanwhile
    post<Job>('/tryon', { image: u.dataUrl, product_id: body.product_id, placement: body.placement,
                          provider: body.provider, settings: body.settings ?? {} })
      .then((j) => local.jobs.set(id, { ...j, id, upload_id: body.upload_id, product_id: body.product_id }))
      .catch((e) => local.jobs.set(id, { ...pending, status: 'failed', error: (e as Error).message }))
    return pending
  },
  health: () => req<{ hf_token: boolean; fal_key: boolean; provider: string }>('/health'),
  getTryon: async (id: string) => {
    const j = local.jobs.get(id)
    if (!j) throw new Error('Result not found.')
    return j
  },
  deleteTryon: async (id: string) => {
    local.jobs.delete(id)
    return { deleted: true, upload_deleted: false }
  },
}

export const api = STATELESS ? statelessApi : serverApi

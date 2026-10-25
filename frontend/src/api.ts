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

export type Placement = { items: PlacementItem[]; chain?: Chain }

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

export const api = {
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
    settings?: { output_long_side: number }
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

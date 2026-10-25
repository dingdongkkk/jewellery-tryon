// Body landmarks in the browser (MediaPipe Tasks, same models as the Python server). Used on the serverless
// deployment, where MediaPipe can't run in the function. Output matches backend/app/autoplace.py `landmarks`:
// coordinates normalised to the photo.
import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'

type Pt = [number, number, number]
export type Landmarks = {
  face?: { points: Pt[] }
  pose?: { points: Pt[] }
  hands?: [number, number][][]
}

let fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null
const models: { face?: FaceLandmarker; pose?: PoseLandmarker; hand?: HandLandmarker } = {}

async function getFileset() {
  fileset ??= await FilesetResolver.forVisionTasks('/mediapipe-wasm')
  return fileset
}

async function faceModel() {
  models.face ??= await FaceLandmarker.createFromOptions(await getFileset(), {
    baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
    runningMode: 'IMAGE',
    numFaces: 1,
  })
  return models.face
}
async function poseModel() {
  models.pose ??= await PoseLandmarker.createFromOptions(await getFileset(), {
    baseOptions: { modelAssetPath: '/models/pose_landmarker_full.task' },
    runningMode: 'IMAGE',
  })
  return models.pose
}
async function handModel() {
  models.hand ??= await HandLandmarker.createFromOptions(await getFileset(), {
    baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
    runningMode: 'IMAGE',
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
  })
  return models.hand
}

/** The photo, shrunk onto a larger grey canvas: detectors miss faces / hands that fill the frame. */
function padded(img: HTMLImageElement, pad: number) {
  const W = img.naturalWidth
  const H = img.naturalHeight
  const c = document.createElement('canvas')
  c.width = Math.round(W * (1 + 2 * pad))
  c.height = Math.round(H * (1 + 2 * pad))
  const ctx = c.getContext('2d')!
  ctx.fillStyle = 'rgb(128,128,128)'
  ctx.fillRect(0, 0, c.width, c.height)
  const ox = Math.round(W * pad)
  const oy = Math.round(H * pad)
  ctx.drawImage(img, ox, oy)
  // map canvas-normalised x,y back to the photo
  const map = (x: number, y: number): [number, number] => [(x * c.width - ox) / W, (y * c.height - oy) / H]
  return { canvas: c, map, scale: c.width / W }
}

async function detectFace(img: HTMLImageElement) {
  const m = await faceModel()
  let r = m.detect(img)
  if (r.faceLandmarks.length) return { points: r.faceLandmarks[0].map((p) => [p.x, p.y, p.z] as Pt) }
  for (const pad of [0.35, 0.8]) {
    const p = padded(img, pad)
    r = m.detect(p.canvas)
    if (r.faceLandmarks.length)
      return {
        points: r.faceLandmarks[0].map((q) => {
          const [x, y] = p.map(q.x, q.y)
          return [x, y, q.z * p.scale] as Pt
        }),
      }
  }
  return undefined
}

async function detectPose(img: HTMLImageElement) {
  const r = (await poseModel()).detect(img)
  if (!r.landmarks.length) return undefined
  return { points: r.landmarks[0].map((p) => [p.x, p.y, p.visibility ?? 1] as Pt) }
}

async function detectHands(img: HTMLImageElement) {
  const m = await handModel()
  let r = m.detect(img)
  if (r.landmarks.length) return r.landmarks.map((h) => h.map((p) => [p.x, p.y] as [number, number]))
  for (const pad of [0.5, 1.0]) {
    const p = padded(img, pad)
    r = m.detect(p.canvas)
    if (r.landmarks.length) return r.landmarks.map((h) => h.map((q) => p.map(q.x, q.y)))
  }
  return []
}

export async function loadImage(url: string): Promise<HTMLImageElement> {
  // onload rather than img.decode(): decode() never settles in some embedded browsers
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not read the photo.'))
    img.src = url
  })
}

/** Only the detectors a category needs (models are downloaded on first use and then cached). */
export async function detectLandmarks(imageUrl: string, category: string): Promise<Landmarks> {
  const img = await loadImage(imageUrl)
  if (category === 'ring') {
    const hands = await detectHands(img)
    return { hands, pose: hands.length ? undefined : await detectPose(img) }
  }
  if (category === 'pendant') return { face: await detectFace(img), pose: await detectPose(img) }
  return { face: await detectFace(img) }
}

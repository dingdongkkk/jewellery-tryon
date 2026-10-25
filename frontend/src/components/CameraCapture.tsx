import { useEffect, useRef, useState } from 'react'

type Props = { onCapture: (photo: Blob) => void; onClose: () => void }

/** Take a photo with the device camera (front or back). Needs https or localhost, like every browser camera API. */
export function CameraCapture({ onCapture, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [facing, setFacing] = useState<'user' | 'environment'>('user')
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [count, setCount] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    const start = async () => {
      setReady(false)
      setError(null)
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('This browser can’t use the camera here. Open the app over https (or localhost), or upload a photo.')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1440 } },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
          setReady(true)
        }
      } catch (e) {
        const name = (e as DOMException).name
        setError(
          name === 'NotAllowedError'
            ? 'Camera permission was denied. Allow camera access in your browser, or upload a photo instead.'
            : name === 'NotFoundError'
              ? 'No camera was found on this device.'
              : `Couldn’t start the camera (${name || 'unknown error'}).`,
        )
      }
    }
    start()
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [facing])

  const snap = () => {
    const v = videoRef.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    const ctx = c.getContext('2d')!
    // the preview is mirrored like a mirror; the saved selfie is mirrored too, so it matches what you saw
    if (facing === 'user') {
      ctx.translate(c.width, 0)
      ctx.scale(-1, 1)
    }
    ctx.drawImage(v, 0, 0)
    c.toBlob((b) => b && onCapture(b), 'image/jpeg', 0.95)
  }

  // a short countdown so you can pose with both hands (e.g. for a ring photo)
  const snapWithTimer = () => {
    let n = 3
    setCount(n)
    const id = window.setInterval(() => {
      n -= 1
      if (n <= 0) {
        window.clearInterval(id)
        setCount(null)
        snap()
      } else setCount(n)
    }, 1000)
  }

  return (
    <div className="camera-overlay" role="dialog" aria-label="Take a photo">
      <div className="camera-box">
        {error ? (
          <p className="error">{error}</p>
        ) : (
          <div className="camera-view">
            <video ref={videoRef} playsInline muted className={facing === 'user' ? 'mirrored' : ''} />
            {count !== null && <div className="camera-count">{count}</div>}
            <p className="camera-tip">
              Earrings: face the camera with your ears visible. Pendant: show your neck and upper chest. Ring: show the
              back of your hand.
            </p>
          </div>
        )}
        <div className="camera-actions">
          <button className="primary" disabled={!ready || count !== null} onClick={snap}>
            📸 Take photo
          </button>
          <button disabled={!ready || count !== null} onClick={snapWithTimer}>
            3-second timer
          </button>
          <button onClick={() => setFacing((f) => (f === 'user' ? 'environment' : 'user'))}>Switch camera</button>
          <button className="ghost" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

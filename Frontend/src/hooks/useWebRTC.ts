import { useEffect, useRef, useState } from 'react'

// FastAPI server that handles the WebRTC SDP offer/answer handshake.
// Replace with your laptop's IP when on the same network.
const SIGNALING_URL = 'http://localhost:8000/offer'

export function useWebRTC(robotId: number) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let pc: RTCPeerConnection

    async function start() {
      try {
        pc = new RTCPeerConnection()
        pcRef.current = pc

        // We only receive video — no need to send tracks
        pc.addTransceiver('video', { direction: 'recvonly' })

        pc.ontrack = (event) => {
          if (videoRef.current) {
            videoRef.current.srcObject = event.streams[0]
          }
        }

        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        const response = await fetch(`${SIGNALING_URL}?robot_id=${robotId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        })

        if (!response.ok) throw new Error(`Signaling failed: ${response.status}`)

        const answer = await response.json()
        await pc.setRemoteDescription(new RTCSessionDescription(answer))
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Connection failed')
      }
    }

    start()

    return () => {
      pc?.close()
    }
  }, [robotId])

  return { videoRef, error }
}

import { useEffect, useState } from 'react'
import mqtt from 'mqtt'

// Connects to Mosquitto broker over WebSockets (port 8000).
// Replace with your laptop's IP when on the same network.
const BROKER_URL = 'wss://mit.ethanzhao.us/mqtt'

export interface Telemetry {
  trashCapacity: number
  recycleCapacity: number
  batteryPercentage: number
  lat: number
  lng: number
}

export function useMqtt(robotId: number, fallback: Telemetry) {
  const [telemetry, setTelemetry] = useState<Telemetry>(fallback)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    setTelemetry(fallback)
    const client = mqtt.connect(BROKER_URL)

    client.on('connect', () => {
      setConnected(true)
      client.subscribe(`robot/${robotId}/telemetry/capacity`)
      client.subscribe(`robot/${robotId}/telemetry/location`)
    })

    client.on('message', (topic, payload) => {
      try {
        const data = JSON.parse(payload.toString())
        if (topic === `robot/${robotId}/telemetry/capacity`) {
          setTelemetry((prev) => ({
            ...prev,
            trashCapacity: data.trashCapacity ?? prev.trashCapacity,
            recycleCapacity: data.recycleCapacity ?? prev.recycleCapacity,
            batteryPercentage: data.batteryPercentage ?? prev.batteryPercentage,
          }))
        } else if (topic === `robot/${robotId}/telemetry/location`) {
          setTelemetry((prev) => ({
            ...prev,
            lat: data.lat ?? prev.lat,
            lng: data.lng ?? prev.lng,
          }))
        }
      } catch {
        // Malformed message — ignore
      }
    })

    client.on('disconnect', () => setConnected(false))

    return () => {
      client.end()
    }
  }, [robotId])

  return { telemetry, connected }
}

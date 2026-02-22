import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import type { Robot } from '../types'
import { useMqtt } from '../hooks/useMqtt'

interface RobotDetailsProps {
  robot: Robot
}

const MapRecenter = ({ lat, lng }: { lat: number; lng: number }) => {
  const map = useMap()
  useEffect(() => {
    map.setView([lat, lng])
  }, [lat, lng, map])
  return null
}

const RobotDetails = ({ robot }: RobotDetailsProps) => {
  const { name, minutesAway } = robot

  // Live telemetry over MQTT — falls back to static robot data until connected
  const { telemetry } = useMqtt(robot.id, {
    trashCapacity: robot.trashCapacity,
    recycleCapacity: robot.recycleCapacity,
    batteryPercentage: robot.batteryPercentage,
    lat: robot.lat,
    lng: robot.lng,
  })

  const { trashCapacity, recycleCapacity, batteryPercentage, lat, lng } = telemetry

  // Track whether the video feed is alive (a frame arrived within the last 2s)
  const [videoLive, setVideoLive] = useState(false)
  const videoTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleVideoFrame = () => {
    setVideoLive(true)
    if (videoTimeoutRef.current) clearTimeout(videoTimeoutRef.current)
    videoTimeoutRef.current = setTimeout(() => setVideoLive(false), 2000)
  }

  useEffect(() => {
    return () => {
      if (videoTimeoutRef.current) clearTimeout(videoTimeoutRef.current)
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="grid h-full w-full grid-rows-[auto_minmax(90px,_1fr)_120px_auto_auto_auto] gap-4 overflow-hidden rounded-2xl border border-white/70 bg-white/90 p-6 text-[#111729] shadow-[0_10px_24px_rgba(27,43,78,0.1)] backdrop-blur"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="m-0 text-[22px] font-bold">{name}</h2>
          <p className="mt-1 text-sm text-[#7f879e]">
            <span className="font-semibold">{minutesAway}</span> minutes away
          </p>
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
            videoLive ? 'bg-[#D1FAE5] text-[#065F46]' : 'bg-[#F3F4F6] text-[#9BA1B0]'
          }`}
        >
          {videoLive ? 'Online' : 'Offline'}
        </span>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.35 }}
        className="relative flex h-full min-h-0 w-full items-center justify-center overflow-hidden rounded-xl bg-[#111729]"
        style={{ aspectRatio: '16 / 9' }}
      >
        <img
          src={`http://localhost:8000/video-feed/${robot.id}`}
          alt="Live camera feed"
          onLoad={handleVideoFrame}
          className="h-full w-full object-cover"
        />
        {!videoLive && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#111729]">
            <span className="text-sm font-semibold text-[#9BA1B0]">Error: No connection</span>
          </div>
        )}
      </motion.div>

      <div className="h-full overflow-hidden rounded-xl">
        <MapContainer center={[lat, lng]} zoom={18} style={{ height: '100%', width: '100%' }} key={robot.id} zoomControl={false}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <MapRecenter lat={lat} lng={lng} />
          <Marker position={[lat, lng]}>
            <Popup>{name}</Popup>
          </Marker>
        </MapContainer>
      </div>

      <div>
        <p className="mb-1.5 text-[13px] font-semibold">Battery</p>
        <div className="flex items-center gap-2.5">
          {batteryPercentage > 20 ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-[18px] w-[18px] shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-[18px] w-[18px] shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="M11.412 15.655 9.75 21.75l3.745-4.012M9.257 13.5H3.75l2.659-2.849m2.048-2.194L14.25 2.25 12 10.5h8.25l-4.707 5.043M8.457 8.457 3 3m5.457 5.457 7.086 7.086m0 0L21 21" />
            </svg>
          )}
          <div className="h-5 flex-1 overflow-hidden rounded-full bg-[#dbe7ff]">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${batteryPercentage}%` }}
              transition={{ duration: 0.45, ease: 'easeOut' }}
              className={`flex h-full items-center justify-center rounded-full text-[11px] font-semibold text-white ${
                batteryPercentage <= 20 ? 'bg-[#E34F4F]' : 'bg-[#597FF5]'
              }`}
            >
              {batteryPercentage}%
            </motion.div>
          </div>
        </div>
      </div>

      <div>
        <p className="mb-1.5 text-[13px] font-semibold">Trash Capacity</p>
        <div className="flex items-center gap-2.5">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="h-[18px] w-[18px] shrink-0">
            <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
          </svg>
          <div className="h-5 flex-1 overflow-hidden rounded-full bg-[#dbe7ff]">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${trashCapacity}%` }}
              transition={{ duration: 0.45, ease: 'easeOut', delay: 0.05 }}
              className="flex h-full items-center justify-center rounded-full bg-[#597FF5] text-[11px] font-semibold text-white"
            >
              {trashCapacity}%
            </motion.div>
          </div>
        </div>
      </div>

      <div>
        <p className="mb-1.5 text-[13px] font-semibold">Recycle Capacity</p>
        <div className="flex items-center gap-2.5">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="h-[18px] w-[18px] shrink-0">
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
          </svg>
          <div className="h-5 flex-1 overflow-hidden rounded-full bg-[#dbe7ff]">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${recycleCapacity}%` }}
              transition={{ duration: 0.45, ease: 'easeOut', delay: 0.1 }}
              className={`flex h-full items-center justify-center rounded-full text-[11px] font-semibold text-white ${
                recycleCapacity >= 100 ? 'bg-[#E34F4F]' : 'bg-[#597FF5]'
              }`}
            >
              {recycleCapacity}%
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default RobotDetails

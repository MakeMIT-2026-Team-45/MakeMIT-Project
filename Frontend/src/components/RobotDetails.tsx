import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet'
import type { Robot } from '../types'
import { useMqtt } from '../hooks/useMqtt'
import { useWebRTC } from '../hooks/useWebRTC'

const HOME_LAT = 42.3570
const HOME_LNG = -71.0920

interface RobotDetailsProps {
  robot: Robot
}

const RobotDetails = ({ robot }: RobotDetailsProps) => {
  const { name, minutesAway } = robot

  // Live telemetry over MQTT — falls back to static robot data until connected
  const { telemetry, connected } = useMqtt(robot.id, {
    trashCapacity: robot.trashCapacity,
    recycleCapacity: robot.recycleCapacity,
    batteryPercentage: robot.batteryPercentage,
    lat: robot.lat,
    lng: robot.lng,
  })

  const { trashCapacity, recycleCapacity, batteryPercentage, lat, lng } = telemetry

  // Live video over WebRTC
  const { videoRef, error: videoError } = useWebRTC(robot.id)

  const midLat = (lat + HOME_LAT) / 2
  const midLng = (lng + HOME_LNG) / 2

  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '20px',
        padding: '24px',
        boxShadow: '0px 8px 25px rgba(0, 0, 0, 0.08)',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        color: '#111729',
        width: '100%',
        height: '100%',
        overflowY: 'auto',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: '#111729' }}>{name}</h2>
          <p style={{ margin: 0, fontSize: '14px', color: '#9BA1B0', marginTop: '4px' }}>
            <span style={{ fontWeight: 600 }}>{minutesAway}</span> minutes away
          </p>
        </div>
        <span style={{
          fontSize: '11px',
          fontWeight: 600,
          padding: '4px 10px',
          borderRadius: '999px',
          backgroundColor: connected ? '#D1FAE5' : '#F3F4F6',
          color: connected ? '#065F46' : '#9BA1B0',
        }}>
          {connected ? 'Live' : 'Offline'}
        </span>
      </div>

      {/* Live Video Feed */}
      <div style={{
        backgroundColor: '#111729',
        borderRadius: '12px',
        width: '100%',
        aspectRatio: '16 / 9',
        overflow: 'hidden',
        flexShrink: 0,
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: videoError ? 'none' : 'block' }}
        />
        {videoError && (
          <span style={{ color: '#9BA1B0', fontSize: '13px' }}>
            Video unavailable — waiting for WebRTC connection
          </span>
        )}
      </div>

      {/* Map */}
      <div style={{ borderRadius: '12px', overflow: 'hidden', height: '200px', flexShrink: 0 }}>
        <MapContainer center={[midLat, midLng]} zoom={14} style={{ height: '100%', width: '100%' }} key={`${lat}-${lng}`} zoomControl={false}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <Marker position={[lat, lng]}>
            <Popup>{name}</Popup>
          </Marker>
          <Marker position={[HOME_LAT, HOME_LNG]}>
            <Popup>You</Popup>
          </Marker>
          <Polyline positions={[[lat, lng], [HOME_LAT, HOME_LNG]]} color="#597FF5" weight={3} />
        </MapContainer>
      </div>

      {/* Battery */}
      <div>
        <p style={{ margin: '0 0 6px', fontSize: '13px', fontWeight: 600, color: '#111729' }}>Battery</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {batteryPercentage > 20 ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{ width: '18px', height: '18px', flexShrink: 0 }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{ width: '18px', height: '18px', flexShrink: 0 }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M11.412 15.655 9.75 21.75l3.745-4.012M9.257 13.5H3.75l2.659-2.849m2.048-2.194L14.25 2.25 12 10.5h8.25l-4.707 5.043M8.457 8.457 3 3m5.457 5.457 7.086 7.086m0 0L21 21" />
            </svg>
          )}
          <div style={{ flex: 1, backgroundColor: '#C0D9FF', borderRadius: '999px', height: '20px', overflow: 'hidden' }}>
            <div style={{
              width: `${batteryPercentage}%`,
              backgroundColor: batteryPercentage <= 20 ? '#E34F4F' : '#597FF5',
              height: '100%',
              borderRadius: '999px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ color: '#FFFFFF', fontSize: '11px', fontWeight: 600 }}>{batteryPercentage}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Trash Capacity */}
      <div>
        <p style={{ margin: '0 0 6px', fontSize: '13px', fontWeight: 600, color: '#111729' }}>Trash Capacity</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" style={{ width: '18px', height: '18px', flexShrink: 0 }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
          </svg>
          <div style={{ flex: 1, backgroundColor: '#C0D9FF', borderRadius: '999px', height: '20px', overflow: 'hidden' }}>
            <div style={{
              width: `${trashCapacity}%`,
              backgroundColor: '#597FF5',
              height: '100%',
              borderRadius: '999px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ color: '#FFFFFF', fontSize: '11px', fontWeight: 600 }}>{trashCapacity}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recycle Capacity */}
      <div>
        <p style={{ margin: '0 0 6px', fontSize: '13px', fontWeight: 600, color: '#111729' }}>Recycle Capacity</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" style={{ width: '18px', height: '18px', flexShrink: 0 }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
          </svg>
          <div style={{ flex: 1, backgroundColor: '#C0D9FF', borderRadius: '999px', height: '20px', overflow: 'hidden' }}>
            <div style={{
              width: `${recycleCapacity}%`,
              backgroundColor: recycleCapacity >= 100 ? '#E34F4F' : '#597FF5',
              height: '100%',
              borderRadius: '999px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ color: '#FFFFFF', fontSize: '11px', fontWeight: 600 }}>{recycleCapacity}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RobotDetails

export interface Robot {
  id: number
  name: string
  minutesAway: number | string
  trashCapacity: number
  recycleCapacity: number
  batteryPercentage: number
  lat: number
  lng: number
}

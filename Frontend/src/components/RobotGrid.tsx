import type { Robot } from '../types'
import { motion } from 'framer-motion'
import RobotCard from './RobotCard'

const robots: Robot[] = [
  { id: 1, name: 'Robot 1', minutesAway: '3', batteryPercentage: 20, trashCapacity: 20, recycleCapacity: 100, lat: 42.3601, lng: -71.0942 },
  { id: 2, name: 'Robot 2', minutesAway: '5', batteryPercentage: 75, trashCapacity: 50, recycleCapacity: 60, lat: 42.3612, lng: -71.0926 },
  { id: 3, name: 'Robot 3', minutesAway: '8', batteryPercentage: 90, trashCapacity: 10, recycleCapacity: 30, lat: 42.3588, lng: -71.0960 },
  { id: 4, name: 'Robot 4', minutesAway: '2', batteryPercentage: 15, trashCapacity: 80, recycleCapacity: 45, lat: 42.3625, lng: -71.0910 },
  { id: 5, name: 'Robot 5', minutesAway: '12', batteryPercentage: 60, trashCapacity: 35, recycleCapacity: 70, lat: 42.3578, lng: -71.0975 },
  { id: 6, name: 'Robot 6', minutesAway: '1', batteryPercentage: 45, trashCapacity: 95, recycleCapacity: 20, lat: 42.3640, lng: -71.0900 },
]

interface RobotGridProps {
  onSelect: (robot: Robot) => void
}

const RobotGrid = ({ onSelect }: RobotGridProps) => {
  return (
    <motion.div
      initial="hidden"
      animate="show"
      variants={{
        hidden: { opacity: 0 },
        show: {
          opacity: 1,
          transition: {
            staggerChildren: 0.06,
          },
        },
      }}
      className="flex flex-col gap-6"
    >
      {robots.map((robot) => (
        <motion.div
          key={robot.id}
          variants={{
            hidden: { opacity: 0, y: 12 },
            show: { opacity: 1, y: 0, transition: { duration: 0.28, ease: 'easeOut' } },
          }}
        >
          <RobotCard
            name={robot.name}
            minutesAway={robot.minutesAway}
            batteryPercentage={robot.batteryPercentage}
            trashCapacity={robot.trashCapacity}
            recycleCapacity={robot.recycleCapacity}
            onManage={() => onSelect(robot)}
          />
        </motion.div>
      ))}
    </motion.div>
  )
}

export default RobotGrid

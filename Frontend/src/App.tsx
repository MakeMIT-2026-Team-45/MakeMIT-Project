import { useState } from 'react'
import { motion } from 'framer-motion'
import './App.css'
import RobotGrid from './components/RobotGrid'
import RobotDetails from './components/RobotDetails'
import type { Robot } from './types'

function App() {
  const [selectedRobot, setSelectedRobot] = useState<Robot | null>(null)

  return (
    <div className="flex min-h-screen w-screen items-center justify-center bg-gradient-to-br from-[#d9dde6] via-[#e4e8f1] to-[#e8edf7] p-4 text-[#111729] md:p-8 overflow-x-auto">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut', delay: 0.05 }}
        className="grid h-[min(90vh,860px)] w-full max-w-[1450px] grid-cols-1 gap-6 overflow-hidden rounded-2xl border border-white/70 bg-white/60 p-4 shadow-[0_12px_36px_rgba(28,45,80,0.12)] backdrop-blur md:grid-cols-2 md:gap-8 md:p-6 overflow-x-auto"
      >
        <div className="overflow-y-auto pr-1 pb-4 md:pb-8 overflow-x-auto">
            <RobotGrid onSelect={setSelectedRobot} />
        </div>

        <div className="flex items-center justify-center pb-2 pt-1 md:pb-4">
          {selectedRobot ? (
            <RobotDetails robot={selectedRobot} />
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="w-full rounded-2xl border border-dashed border-[#c9d5fb] bg-white/70 p-10 text-center text-[#6c748b]"
            >
              Select a robot to view live telemetry.
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  )
}

export default App

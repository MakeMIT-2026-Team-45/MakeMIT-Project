import { useState } from 'react'
import './App.css'
import RobotGrid from './components/RobotGrid'
import RobotDetails from './components/RobotDetails'
import type { Robot } from './types'

function App() {
  const [selectedRobot, setSelectedRobot] = useState<Robot | null>(null)

  return (
    <div className="grid grid-cols-10 w-screen h-screen pt-16 gap-x-8 pb-8 px-8 overflow-hidden">
      <div className="col-span-10 bg-[#F2F3F7] rounded-xl overflow-hidden">
        <div className="grid grid-cols-2 h-full">
          <div className="overflow-y-auto px-8 pt-8 pb-16">
            <RobotGrid onSelect={setSelectedRobot} />
          </div>
          <div className="flex items-center justify-center p-8 pb-8">
            {selectedRobot ? (
              <RobotDetails robot={selectedRobot} />
            ) : (
              <span className="text-[#9BA1B0]">No robot selected.</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

import './App.css'
import RobotGrid from './components/RobotGrid'

function App() {
  return (
    <div className="grid grid-cols-10 w-screen h-screen pt-16 gap-x-8 pb-8 pr-8 overflow-hidden">
      <div className="col-span-1"></div>
      <div className="col-span-9 bg-[#F2F3F7] rounded-xl overflow-hidden">
        <div className="grid grid-cols-2 h-full">
          <div className="overflow-y-auto px-8 pt-8 pb-16">
            <RobotGrid />
          </div>
          <div className="flex items-center justify-center text-[#9BA1B0]">No robot selected.</div>
        </div>
      </div>
    </div>
  )
}

export default App

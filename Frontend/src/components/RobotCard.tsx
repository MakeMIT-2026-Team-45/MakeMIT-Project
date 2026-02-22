import { motion } from 'framer-motion'

interface RobotCardProps {
  name: string
  minutesAway: number | string
  trashCapacity: number
  recycleCapacity: number
  batteryPercentage: number
  onManage?: () => void
}

const RobotCard = ({
  name,
  minutesAway,
  trashCapacity,
  recycleCapacity,
  batteryPercentage,
  onManage,
}: RobotCardProps) => {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ type: 'spring', stiffness: 270, damping: 24 }}
      className="w-full rounded-2xl border border-white/70 bg-white/90 p-4 text-[#111729] shadow-[10px_10px_24px_rgba(27,43,78,0.1)] backdrop-blur"
    >
      <div>
        <h2 className="m-0 text-lg font-bold">{name}</h2>
        <p className="mt-1 text-sm text-[#7f879e]">
          <span className="font-semibold">{minutesAway}</span> {minutesAway == 1 ? 'minute' : 'minutes'} away
        </p>
      </div>

      <div className="mt-4 flex items-center gap-2.5">
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

      <div className="mt-3 flex flex-col gap-2.5">
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

      <motion.button
        whileTap={{ scale: 0.98 }}
        onClick={onManage}
        className="mt-4 w-full rounded-xl border border-[#597FF5]/40 bg-[#eff3ff] py-2 text-sm font-semibold text-[#3558bf] transition-colors hover:bg-[#e3ebff]"
      >
        Manage Robot
      </motion.button>
    </motion.div>
  )
}

export default RobotCard

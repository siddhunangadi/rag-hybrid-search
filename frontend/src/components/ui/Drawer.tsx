import { ReactNode } from 'react'
import { X } from 'lucide-react'
import clsx from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'

interface DrawerProps {
  isOpen: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
  size?: 'sm' | 'md' | 'lg'
}

const sizeClasses = {
  sm: 'w-96',
  md: 'w-[600px]',
  lg: 'w-[800px]',
}

export function Drawer({ isOpen, onClose, title, children, footer, size = 'md' }: DrawerProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/50"
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={clsx('fixed right-0 top-0 bottom-0 z-50 bg-slate-900 border-l border-slate-800 flex flex-col', sizeClasses[size])}
          >
            <div className="flex items-center justify-between p-6 border-b border-slate-800">
              <h2 className="text-lg font-semibold">{title}</h2>
              <button onClick={onClose} className="text-slate-400 hover:text-slate-200 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">{children}</div>

            {footer && <div className="border-t border-slate-800 p-6 bg-slate-900/50">{footer}</div>}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

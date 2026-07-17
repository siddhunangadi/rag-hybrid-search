import { Link, useLocation } from 'react-router-dom'
import { BarChart3, Upload, MessageCircle, BookOpen, History, Activity, Settings, ChevronRight } from 'lucide-react'
import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import clsx from 'clsx'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const isActive = (path: string) => location.pathname === path

  const navItems = [
    { href: '/', icon: BarChart3, label: 'Dashboard' },
    { href: '/upload', icon: Upload, label: 'Upload' },
    { href: '/chat', icon: MessageCircle, label: 'AI Assistant' },
    { href: '/regulations', icon: BookOpen, label: 'Regulations' },
    { href: '/audit', icon: History, label: 'Audit' },
    { href: '/health', icon: Activity, label: 'Health' },
    { href: '/admin', icon: Settings, label: 'Admin' },
  ]

  return (
    <div className="flex h-screen bg-slate-950">
      <aside className="w-64 border-r border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 flex flex-col">
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="p-6 border-b border-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg" />
            <div>
              <h1 className="font-bold text-white">Compliance AI</h1>
              <p className="text-xs text-slate-400">Enterprise Edition</p>
            </div>
          </div>
        </motion.div>

        <nav className="flex-1 px-3 py-6 space-y-1 overflow-y-auto scrollbar-thin">
          {navItems.map((item, idx) => (
            <motion.div
              key={item.href}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
            >
              <Link
                to={item.href}
                className={clsx(
                  'flex items-center justify-between gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 group',
                  isActive(item.href)
                    ? 'bg-blue-600/90 text-white font-medium shadow-lg'
                    : 'text-slate-300 hover:bg-slate-800/50 hover:text-slate-100'
                )}
              >
                <div className="flex items-center gap-3">
                  <item.icon className={clsx('w-4 h-4', isActive(item.href) ? 'text-white' : 'text-slate-400')} />
                  <span className="text-sm">{item.label}</span>
                </div>
                {isActive(item.href) && <ChevronRight className="w-3 h-3 opacity-70" />}
              </Link>
            </motion.div>
          ))}
        </nav>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="p-4 border-t border-slate-800/50 space-y-2 bg-slate-900/50">
          <p className="text-xs text-slate-400">Build: 1.0.0</p>
          <p className="text-xs text-slate-500">© 2026 Financial Compliance AI</p>
        </motion.div>
      </aside>

      <main className="flex-1 overflow-auto bg-slate-950">
        <div className="max-w-7xl mx-auto px-8 py-8">{children}</div>
      </main>
    </div>
  )
}

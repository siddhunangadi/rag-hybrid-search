import { ReactNode } from 'react'
import clsx from 'clsx'

interface Tab {
  id: string
  label: string
  content: ReactNode
  badge?: number
}

interface TabsProps {
  tabs: Tab[]
  activeId: string
  onTabChange: (id: string) => void
}

export function Tabs({ tabs, activeId, onTabChange }: TabsProps) {
  return (
    <div>
      <div className="flex gap-1 border-b border-slate-800">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={clsx(
              'px-4 py-2 text-sm font-medium transition-all relative',
              activeId === tab.id
                ? 'text-blue-400'
                : 'text-slate-400 hover:text-slate-300'
            )}
          >
            {tab.label}
            {tab.badge && (
              <span className="ml-2 bg-blue-600 text-white text-xs rounded-full px-2 py-0.5">
                {tab.badge}
              </span>
            )}
            {activeId === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
            )}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tabs.find((tab) => tab.id === activeId)?.content}
      </div>
    </div>
  )
}

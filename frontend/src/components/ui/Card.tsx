import { ReactNode } from 'react'
import clsx from 'clsx'

interface CardProps {
  children: ReactNode
  className?: string
  interactive?: boolean
}

export function Card({ children, className, interactive }: CardProps) {
  return (
    <div
      className={clsx(
        'bg-slate-900 border border-slate-800 rounded-lg p-6 shadow-lg',
        interactive && 'hover:border-slate-700 transition-colors cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  )
}

export function MetricCard({ label, value, trend, unit }: {
  label: string
  value: string | number
  trend?: { direction: 'up' | 'down'; percent: number }
  unit?: string
}) {
  return (
    <Card>
      <p className="text-slate-400 text-sm font-medium">{label}</p>
      <div className="flex items-end gap-2 mt-2">
        <p className="text-3xl font-bold">{value}</p>
        {unit && <p className="text-slate-400 text-sm mb-1">{unit}</p>}
      </div>
      {trend && (
        <div className={`mt-2 text-sm font-medium flex items-center gap-1 ${
          trend.direction === 'up' ? 'text-emerald-400' : 'text-red-400'
        }`}>
          <span>{trend.direction === 'up' ? '↑' : '↓'} {trend.percent}%</span>
        </div>
      )}
    </Card>
  )
}

export function StatusCard({ status, label, details }: {
  status: 'healthy' | 'warning' | 'error'
  label: string
  details?: string
}) {
  const statusColors = {
    healthy: 'bg-emerald-500/10 border-emerald-500/30',
    warning: 'bg-amber-500/10 border-amber-500/30',
    error: 'bg-red-500/10 border-red-500/30',
  }

  const statusDots = {
    healthy: 'bg-emerald-500',
    warning: 'bg-amber-500',
    error: 'bg-red-500',
  }

  return (
    <div className={`border rounded-lg p-4 ${statusColors[status]}`}>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${statusDots[status]}`} />
        <span className="font-medium text-sm">{label}</span>
      </div>
      {details && <p className="text-xs text-slate-400 mt-1">{details}</p>}
    </div>
  )
}

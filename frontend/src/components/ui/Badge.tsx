import clsx from 'clsx'
import { ReactNode } from 'react'

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  icon?: ReactNode
}

const variantClasses: Record<BadgeVariant, string> = {
  default: 'bg-slate-700 text-slate-100',
  success: 'bg-emerald-500/20 text-emerald-300',
  warning: 'bg-amber-500/20 text-amber-300',
  error: 'bg-red-500/20 text-red-300',
  info: 'bg-blue-500/20 text-blue-300',
}

export function Badge({ children, variant = 'default', icon }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium',
        variantClasses[variant]
      )}
    >
      {icon && <span className="text-xs">{icon}</span>}
      {children}
    </span>
  )
}

export function StatusBadge({ status }: { status: 'pending' | 'processing' | 'completed' | 'failed' }) {
  const variants: Record<typeof status, BadgeVariant> = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'error',
  }

  const labels = {
    pending: 'Pending',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
  }

  return <Badge variant={variants[status]}>{labels[status]}</Badge>
}

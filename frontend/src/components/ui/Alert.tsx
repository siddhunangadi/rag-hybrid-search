import { ReactNode } from 'react'
import { AlertCircle, CheckCircle, InfoIcon, AlertTriangle, X } from 'lucide-react'
import clsx from 'clsx'

type AlertVariant = 'success' | 'error' | 'warning' | 'info'

interface AlertProps {
  variant: AlertVariant
  title: string
  message?: string
  onClose?: () => void
  action?: { label: string; onClick: () => void }
}

const variantConfig: Record<AlertVariant, { bg: string; border: string; icon: ReactNode }> = {
  success: {
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    icon: <CheckCircle className="w-5 h-5 text-emerald-400" />,
  },
  error: {
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    icon: <AlertCircle className="w-5 h-5 text-red-400" />,
  },
  warning: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
  },
  info: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
    icon: <InfoIcon className="w-5 h-5 text-blue-400" />,
  },
}

export function Alert({ variant, title, message, onClose, action }: AlertProps) {
  const config = variantConfig[variant]

  return (
    <div className={clsx('rounded-lg border p-4 flex gap-3', config.bg, config.border)}>
      <div className="flex-shrink-0">{config.icon}</div>
      <div className="flex-1">
        <h3 className="font-medium text-sm">{title}</h3>
        {message && <p className="text-sm text-slate-300 mt-1">{message}</p>}
        {action && (
          <button onClick={action.onClick} className="text-sm font-medium mt-2 hover:underline">
            {action.label}
          </button>
        )}
      </div>
      {onClose && (
        <button onClick={onClose} className="text-slate-400 hover:text-slate-200">
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  )
}

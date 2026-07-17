import { InputHTMLAttributes, ReactNode } from 'react'
import clsx from 'clsx'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: ReactNode
  hint?: string
}

export function Input({ label, error, icon, hint, className, id, ...props }: InputProps) {
  const inputId = id || `input-${Math.random()}`

  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-slate-200">
          {label}
        </label>
      )}
      <div className="relative">
        {icon && <div className="absolute left-3 top-3 text-slate-400">{icon}</div>}
        <input
          id={inputId}
          className={clsx(
            'w-full px-4 py-2 bg-slate-800 border rounded-lg text-slate-50 placeholder:text-slate-500',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            'transition-all',
            icon && 'pl-10',
            error && 'border-red-500 focus:ring-red-500',
            !error && 'border-slate-700',
            className
          )}
          {...props}
        />
      </div>
      {error && <p className="text-sm text-red-400">{error}</p>}
      {hint && !error && <p className="text-sm text-slate-400">{hint}</p>}
    </div>
  )
}

export function Textarea({ label, error, hint, className, id, ...props }: Omit<InputProps, 'icon'> & React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const inputId = id || `textarea-${Math.random()}`

  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-slate-200">
          {label}
        </label>
      )}
      <textarea
        id={inputId}
        className={clsx(
          'w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-50 placeholder:text-slate-500',
          'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
          'transition-all resize-none',
          error && 'border-red-500 focus:ring-red-500',
          className
        )}
        {...props}
      />
      {error && <p className="text-sm text-red-400">{error}</p>}
      {hint && !error && <p className="text-sm text-slate-400">{hint}</p>}
    </div>
  )
}

import { useState, useRef, useEffect, ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import clsx from 'clsx'

interface DropdownOption {
  value: string
  label: string
  icon?: ReactNode
}

interface DropdownProps {
  options: DropdownOption[]
  value: string
  onChange: (value: string) => void
  label?: string
  placeholder?: string
}

export function Dropdown({ options, value, onChange, label, placeholder }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedOption = options.find((opt) => opt.value === value)

  return (
    <div className="relative w-full" ref={ref}>
      {label && <label className="block text-sm font-medium text-slate-200 mb-1">{label}</label>}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-50 text-left flex items-center justify-between hover:border-slate-600 transition-colors"
      >
        <span>{selectedOption?.label || placeholder || 'Select...'}</span>
        <ChevronDown className={clsx('w-4 h-4 text-slate-400 transition-transform', isOpen && 'rotate-180')} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-slate-900 border border-slate-700 rounded-lg shadow-lg z-50">
          {options.map((option) => (
            <button
              key={option.value}
              onClick={() => {
                onChange(option.value)
                setIsOpen(false)
              }}
              className={clsx(
                'w-full px-4 py-2 text-left text-sm flex items-center gap-2 transition-colors',
                value === option.value
                  ? 'bg-blue-600/20 text-blue-300'
                  : 'hover:bg-slate-800 text-slate-300'
              )}
            >
              {option.icon && <span>{option.icon}</span>}
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

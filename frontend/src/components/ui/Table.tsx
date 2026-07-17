import { ReactNode } from 'react'
import clsx from 'clsx'
import { ChevronUp, ChevronDown } from 'lucide-react'

interface Column<T> {
  key: keyof T
  label: string
  sortable?: boolean
  render?: (value: T[keyof T], row: T) => ReactNode
  width?: string
}

interface TableProps<T> {
  columns: Column<T>[]
  data: T[]
  onSort?: (key: string, direction: 'asc' | 'desc') => void
  sortBy?: string
  sortDir?: 'asc' | 'desc'
  onRowClick?: (row: T) => void
  loading?: boolean
}

export function Table<T extends { id?: string }>({
  columns,
  data,
  onSort,
  sortBy,
  sortDir,
  onRowClick,
  loading,
}: TableProps<T>) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-slate-800/50 border-b border-slate-700">
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={clsx('px-6 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider', col.width)}
              >
                <button
                  onClick={() => col.sortable && onSort?.(String(col.key), sortDir === 'asc' ? 'desc' : 'asc')}
                  className={clsx('flex items-center gap-2', col.sortable && 'hover:text-slate-100 cursor-pointer')}
                >
                  {col.label}
                  {col.sortable && sortBy === String(col.key) && (
                    <span className="text-blue-400">
                      {sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </span>
                  )}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} className="bg-slate-900/50">
                {columns.map((col) => (
                  <td key={String(col.key)} className="px-6 py-4">
                    <div className="h-4 bg-slate-700 rounded animate-pulse w-full" />
                  </td>
                ))}
              </tr>
            ))
          ) : data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-6 py-12 text-center text-slate-400">
                No data available
              </td>
            </tr>
          ) : (
            data.map((row, idx) => (
              <tr
                key={row.id || idx}
                onClick={() => onRowClick?.(row)}
                className={clsx(
                  'bg-slate-900/20 hover:bg-slate-800/50 transition-colors',
                  onRowClick && 'cursor-pointer'
                )}
              >
                {columns.map((col) => (
                  <td key={String(col.key)} className="px-6 py-4 text-sm text-slate-300">
                    {col.render ? col.render(row[col.key], row) : String(row[col.key] ?? '—')}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

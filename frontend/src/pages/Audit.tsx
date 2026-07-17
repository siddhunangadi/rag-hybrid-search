import { useState, useEffect } from 'react'
import { Download, Loader } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiClient } from '@/lib/api'
import { AuditEvent } from '@/lib/types'
import { Card, Button, Badge } from '@/components/ui'

export default function Audit() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'upload' | 'query' | 'failure'>('all')

  useEffect(() => {
    apiClient.getAuditEvents(100, 0).then((res) => {
      setEvents(res.events)
      setLoading(false)
    })
  }, [])

  const filtered = events.filter((e) => {
    if (filter === 'all') return true
    if (filter === 'upload') return e.action.includes('upload')
    if (filter === 'query') return e.action.includes('query')
    if (filter === 'failure') return e.status === 'failure'
    return true
  })

  const exportCSV = () => {
    const csv = [
      ['Timestamp', 'Event Type', 'Action', 'Status', 'User', 'Endpoint'],
      ...filtered.map((e) => [e.timestamp, e.event_type, e.action, e.status, e.key_id, e.endpoint]),
    ]
      .map((row) => row.map((cell) => `"${cell}"`).join(','))
      .join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'audit-log.csv'
    a.click()
  }

  return (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Audit Center</h1>
        <p className="text-slate-400 mt-1">Track compliance events and system activities</p>
      </div>

      <Card>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex flex-wrap gap-2">
            {(['all', 'upload', 'query', 'failure'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded text-sm font-medium transition-all ${
                  filter === f ? 'bg-blue-600 text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                }`}
              >
                {f === 'all' && 'All Events'}
                {f === 'upload' && 'Uploads'}
                {f === 'query' && 'Queries'}
                {f === 'failure' && 'Failures'}
              </button>
            ))}
          </div>
          <Button onClick={exportCSV} variant="secondary">
            <Download className="w-4 h-4" />
            Export CSV
          </Button>
        </div>
      </Card>

      {loading ? (
        <Card className="text-center py-12 flex items-center justify-center gap-2 text-slate-400">
          <Loader className="w-4 h-4 animate-spin" />
          Loading audit log...
        </Card>
      ) : (
        <Card>
          <div>
            <p className="text-sm text-slate-400 mb-4">{filtered.length} events</p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-slate-700">
                  <tr>
                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Timestamp</th>
                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Action</th>
                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Status</th>
                    <th className="text-left py-3 px-4 text-slate-400 font-medium">User</th>
                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Endpoint</th>
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence>
                    {filtered.map((event, idx) => (
                      <motion.tr
                        key={event.event_id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: idx * 0.02 }}
                        className="border-b border-slate-800 hover:bg-slate-800/30"
                      >
                        <td className="py-3 px-4 text-slate-300 text-sm">{new Date(event.timestamp).toLocaleString()}</td>
                        <td className="py-3 px-4 text-slate-300">{event.action}</td>
                        <td className="py-3 px-4">
                          <Badge>{event.status}</Badge>
                        </td>
                        <td className="py-3 px-4 text-slate-400 text-sm">{event.key_id.substring(0, 8)}</td>
                        <td className="py-3 px-4 text-slate-400 text-sm">{event.endpoint}</td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}
    </motion.div>
  )
}

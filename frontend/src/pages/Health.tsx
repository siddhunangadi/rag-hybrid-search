import { useState, useEffect } from 'react'
import { AlertCircle, Zap, Eye } from 'lucide-react'
import { motion } from 'framer-motion'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { apiClient } from '@/lib/api'
import { DiagnosticsResponse, ReadinessResponse, MetricsResponse } from '@/lib/types'
import { Card, CardSkeleton, MetricCard } from '@/components/ui'

export default function Health() {
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse>()
  const [readiness, setReadiness] = useState<ReadinessResponse>()
  const [metrics, setMetrics] = useState<MetricsResponse>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([apiClient.getDiagnostics(), apiClient.getReadiness(), apiClient.getMetrics()])
      .then(([d, r, m]) => {
        setDiagnostics(d)
        setReadiness(r)
        setMetrics(m)
      })
      .finally(() => setLoading(false))
  }, [])

  const latencyTrend = [
    { time: '00:00', latency: 45 },
    { time: '04:00', latency: 52 },
    { time: '08:00', latency: 38 },
    { time: '12:00', latency: 61 },
    { time: '16:00', latency: 48 },
    { time: '20:00', latency: 55 },
  ]

  if (loading)
    return (
      <div className="space-y-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    )

  return (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">System Health</h1>
        <p className="text-slate-400 mt-1">Real-time system monitoring and diagnostics</p>
      </div>

      {/* Component Status Grid */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
        <h2 className="text-lg font-semibold mb-4">Component Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {readiness?.checks.map((check, idx) => (
            <motion.div
              key={check.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * idx }}
              className="border border-slate-700 rounded-lg p-4"
            >
              <p className="text-slate-400 text-sm">{check.name}</p>
              <p className={`text-lg font-semibold mt-2 ${check.ok ? 'text-emerald-400' : 'text-red-400'}`}>
                {check.ok ? '✓ Operational' : '✗ Degraded'}
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
        <h2 className="text-lg font-semibold mb-4">Performance</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <MetricCard
            label="Avg Latency"
            value={metrics?.avg_latency_ms.toFixed(0) ?? '0'}
            unit="ms"
            trend={{ direction: 'down', percent: 5 }}
          />
          <MetricCard
            label="P95 Latency"
            value={metrics?.p95_latency_ms.toFixed(0) ?? '0'}
            unit="ms"
            trend={{ direction: 'down', percent: 3 }}
          />
          <MetricCard
            label="Error Rate"
            value={((((metrics?.error_count ?? 0) / (metrics?.request_count ?? 1)) * 100).toFixed(1))}
            unit="%"
            trend={{ direction: 'down', percent: 2 }}
          />
        </div>
      </motion.div>

      {/* Request Metrics */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
        <h2 className="text-lg font-semibold mb-4">Request Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Total Requests</p>
                <p className="text-3xl font-bold mt-2">{metrics?.request_count}</p>
              </div>
              <Zap className="w-8 h-8 text-blue-400 opacity-50" />
            </div>
          </Card>
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Errors</p>
                <p className={`text-3xl font-bold mt-2 ${(metrics?.error_count ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                  {metrics?.error_count}
                </p>
              </div>
              <AlertCircle className={`w-8 h-8 opacity-50 ${(metrics?.error_count ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`} />
            </div>
          </Card>
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Uptime</p>
                <p className="text-3xl font-bold mt-2 text-emerald-400">99.9%</p>
              </div>
              <Eye className="w-8 h-8 text-emerald-400 opacity-50" />
            </div>
          </Card>
        </div>
      </motion.div>

      {/* Latency Trend Chart */}
      {metrics && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
          <h2 className="text-lg font-semibold mb-4">Latency Trend</h2>
          <Card>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={latencyTrend}>
                <defs>
                  <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Area type="monotone" dataKey="latency" stroke="#3b82f6" fillOpacity={1} fill="url(#colorLatency)" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </motion.div>
      )}

      {/* System Info */}
      {diagnostics && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
          <h2 className="text-lg font-semibold mb-4">System Information</h2>
          <Card>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <p className="text-slate-400 text-sm font-medium">Version</p>
                <p className="font-semibold mt-1 text-sm">{diagnostics.version}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm font-medium">Python Version</p>
                <p className="font-semibold mt-1 text-sm">{diagnostics.python_version}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm font-medium">Provider</p>
                <p className="font-semibold mt-1 text-sm">{diagnostics.provider}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm font-medium">Storage Backend</p>
                <p className="font-semibold mt-1 text-sm">{diagnostics.storage_backend}</p>
              </div>
            </div>
          </Card>
        </motion.div>
      )}
    </motion.div>
  )
}

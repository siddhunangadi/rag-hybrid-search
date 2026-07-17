import { useEffect, useState } from 'react'
import { Lock } from 'lucide-react'
import { motion } from 'framer-motion'
import { apiClient } from '@/lib/api'
import { DiagnosticsResponse } from '@/lib/types'
import { Card, CardSkeleton } from '@/components/ui'

export default function Admin() {
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    apiClient
      .getDiagnostics()
      .then((d) => setDiagnostics(d))
      .finally(() => setLoading(false))
  }, [])

  if (loading)
    return (
      <div className="space-y-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    )

  return (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div className="flex items-center gap-2">
        <Lock className="w-6 h-6 text-amber-500" />
        <div>
          <h1 className="text-3xl font-bold">Administration</h1>
          <p className="text-slate-400 mt-1">System configuration and diagnostics (read-only)</p>
        </div>
      </div>

      {/* Configuration */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">Version</p>
            <p className="font-semibold text-sm">{diagnostics?.version}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">Python Version</p>
            <p className="font-semibold text-sm">{diagnostics?.python_version}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">LLM Provider</p>
            <p className="font-semibold text-sm">{diagnostics?.provider}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">Rerank Backend</p>
            <p className="font-semibold text-sm">{diagnostics?.rerank_backend}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">Storage Backend</p>
            <p className="font-semibold text-sm">{diagnostics?.storage_backend}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg">
            <p className="text-xs text-slate-400 mb-1">Status</p>
            <p className="font-semibold text-sm text-emerald-400">Operational</p>
          </div>
        </div>
      </Card>

      {/* Provider Status */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Provider Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { name: 'Pinecone', status: diagnostics?.pinecone_ready },
            { name: 'BM25 Index', status: diagnostics?.bm25_ready },
            { name: 'Embedding Provider', status: diagnostics?.embedding_provider_ready },
            { name: 'Audit Log', status: diagnostics?.audit_ready },
          ].map((item) => (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="border border-slate-700 rounded-lg p-4"
            >
              <p className="text-slate-400 text-sm">{item.name}</p>
              <p className={`text-lg font-semibold mt-2 ${item.status ? 'text-emerald-400' : 'text-red-400'}`}>
                {item.status ? '✓ Ready' : '✗ Offline'}
              </p>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Statistics */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-slate-400 mb-2">Total Requests</p>
            <p className="text-3xl font-bold">{diagnostics?.request_count}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400 mb-2">Error Count</p>
            <p className="text-3xl font-bold text-red-400">{diagnostics?.error_count}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400 mb-2">Avg Response Latency</p>
            <p className="text-3xl font-bold">{diagnostics?.avg_response_latency_ms.toFixed(0)}ms</p>
          </div>
        </div>
      </Card>
    </motion.div>
  )
}

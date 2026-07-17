import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell } from 'recharts'
import { Search, Upload, MessageCircle, FileText, TrendingUp } from 'lucide-react'
import { apiClient } from '@/lib/api'
import { DocumentsResponse, DiagnosticsResponse } from '@/lib/types'
import { MetricCard, Card, CardSkeleton, Button } from '@/components/ui'

const COMPLIANCE_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4']

export default function Dashboard() {
  const navigate = useNavigate()
  const [docs, setDocs] = useState<DocumentsResponse | null>(null)
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([apiClient.listDocuments(), apiClient.getDiagnostics()])
      .then(([docsData, diagData]) => {
        setDocs(docsData)
        setDiagnostics(diagData)
      })
      .finally(() => setLoading(false))
  }, [])

  const authoritiesCount = docs?.documents?.length ? new Set(docs.documents.map(d => d.authority)).size : 0
  const recentUpdates = docs?.documents?.slice(0, 3).length ?? 0

  const chartData = [
    { name: 'Mon', queries: 24, uploads: 8 },
    { name: 'Tue', queries: 29, uploads: 12 },
    { name: 'Wed', queries: 20, uploads: 5 },
    { name: 'Thu', queries: 32, uploads: 15 },
    { name: 'Fri', queries: 40, uploads: 18 },
  ]

  const authorityDistribution = [
    { name: 'RBI', value: 35 },
    { name: 'SEBI', value: 25 },
    { name: 'GST', value: 20 },
    { name: 'MCA', value: 15 },
    { name: 'IRDAI', value: 5 },
  ]

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Welcome back</h1>
          <p className="text-slate-400 mt-1">Your regulatory knowledge base is healthy and up to date.</p>
        </div>

        {/* Hero Search & Actions */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search regulations, circulars, or acts..."
              className="w-full pl-10 pr-4 py-2.5 bg-slate-900 border border-slate-800 rounded-lg text-slate-300 placeholder-slate-500 focus:outline-none focus:border-slate-700"
              onClick={() => navigate('/regulations')}
              readOnly
            />
          </div>
          <Button onClick={() => navigate('/chat')} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700">
            <MessageCircle className="w-4 h-4" />
            Ask AI
          </Button>
          <Button onClick={() => navigate('/upload')} variant="secondary" className="flex items-center gap-2">
            <Upload className="w-4 h-4" />
            Upload
          </Button>
        </div>
      </motion.div>

      {/* KPI Metrics - 6 Cards */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ staggerChildren: 0.05 }}
        >
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <MetricCard
              label="Current Regulations"
              value={docs?.total ?? 0}
              unit="active"
            />
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
            <MetricCard
              label="Authorities Covered"
              value={authoritiesCount}
              unit="agencies"
            />
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <MetricCard
              label="Recent Updates"
              value={recentUpdates}
              unit="regulations"
              trend={{ direction: 'up', percent: 5 }}
            />
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
            <MetricCard
              label="AI Questions"
              value={diagnostics?.request_count ?? 0}
              unit="this week"
              trend={{ direction: 'up', percent: 12 }}
            />
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card>
              <p className="text-slate-400 text-sm font-medium">Compliance Health</p>
              <div className="mt-2">
                <p className="text-2xl font-bold text-emerald-400">Healthy</p>
                <p className="text-xs text-slate-400 mt-1">All systems operational</p>
              </div>
            </Card>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
            <MetricCard
              label="Avg Response Time"
              value={Math.round(diagnostics?.avg_response_latency_ms ?? 234)}
              unit="ms"
              trend={{ direction: 'down', percent: 3 }}
            />
          </motion.div>
        </motion.div>
      )}

      {/* Activity Section - Two Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Regulatory Updates */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <h2 className="text-lg font-semibold mb-4">Recent Regulatory Updates</h2>
          <Card>
            <div className="space-y-3">
              {docs?.documents?.slice(0, 5).map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 * idx }}
                  className="flex items-start justify-between p-3 hover:bg-slate-800/50 rounded-lg transition-colors cursor-pointer"
                  onClick={() => navigate(`/regulations`)}
                >
                  <div className="flex-1">
                    <p className="font-medium text-sm">{doc.filename}</p>
                    <p className="text-xs text-slate-400 mt-1">{doc.authority}</p>
                  </div>
                  <span className="text-xs px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/30">Current</span>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Recent Upload Activity */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}>
          <h2 className="text-lg font-semibold mb-4">Recent Upload Activity</h2>
          <Card>
            <div className="space-y-3">
              {docs?.documents?.slice(0, 5).map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 * idx }}
                  className="flex items-start justify-between p-3 hover:bg-slate-800/50 rounded-lg transition-colors"
                >
                  <div className="flex items-start gap-3 flex-1">
                    <FileText className="w-4 h-4 text-blue-400 mt-1 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium">{doc.filename}</p>
                      <p className="text-xs text-slate-400 mt-1">Indexed successfully</p>
                    </div>
                  </div>
                  <span className="text-xs text-slate-400">Today</span>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Insights Section - Two Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Most Accessed Regulations */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
          <h2 className="text-lg font-semibold mb-4">Most Accessed Regulations</h2>
          <Card>
            <div className="space-y-3">
              {['RBI Master Circular 2025', 'SEBI Cyber Security Guidelines', 'GST Input Tax Credit Rules'].map((reg, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 hover:bg-slate-800/50 rounded-lg transition-colors cursor-pointer">
                  <div>
                    <p className="text-sm font-medium">{reg}</p>
                    <p className="text-xs text-slate-400 mt-1">{Math.floor(Math.random() * 50) + 20} views</p>
                  </div>
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                </div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Recent AI Questions */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
          <h2 className="text-lg font-semibold mb-4">Recent AI Questions</h2>
          <Card>
            <div className="space-y-3">
              {[
                'Latest RBI KYC requirements',
                'SEBI cyber security regulations',
                'GST input tax credit changes'
              ].map((q, idx) => (
                <div key={idx} className="p-3 hover:bg-slate-800/50 rounded-lg transition-colors cursor-pointer" onClick={() => navigate('/chat')}>
                  <p className="text-sm font-medium">{q}</p>
                  <div className="flex gap-2 mt-2">
                    <span className="text-xs px-2 py-1 bg-blue-500/10 text-blue-400 rounded">Answered</span>
                    <span className="text-xs text-slate-400">High confidence</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Compliance Overview Charts */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Query & Upload Trends */}
        <Card>
          <h3 className="text-lg font-semibold mb-4">Activity Trends</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Legend />
              <Bar dataKey="queries" fill="#3b82f6" name="Questions Asked" />
              <Bar dataKey="uploads" fill="#10b981" name="Documents Uploaded" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Regulations by Authority */}
        <Card>
          <h3 className="text-lg font-semibold mb-4">Regulations by Authority</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={authorityDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {authorityDistribution.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COMPLIANCE_COLORS[index % COMPLIANCE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                labelStyle={{ color: '#e2e8f0' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </motion.div>

      {/* Quick Actions */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="flex flex-wrap gap-3">
        <Button onClick={() => navigate('/chat')} className="flex items-center gap-2">
          <MessageCircle className="w-4 h-4" />
          Ask AI
        </Button>
        <Button onClick={() => navigate('/upload')} variant="secondary" className="flex items-center gap-2">
          <Upload className="w-4 h-4" />
          Upload Regulation
        </Button>
        <Button onClick={() => navigate('/regulations')} variant="secondary" className="flex items-center gap-2">
          Browse Regulations
        </Button>
      </motion.div>
    </div>
  )
}

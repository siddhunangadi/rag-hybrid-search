import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Download, ChevronDown } from 'lucide-react'
import { apiClient } from '@/lib/api'
import { DocumentSummary } from '@/lib/types'
import { Table, Drawer, Search, Pagination, Card, Button, Badge, Tabs } from '@/components/ui'

export default function Regulations() {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [filteredDocs, setFilteredDocs] = useState<DocumentSummary[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [mode, setMode] = useState<'search' | 'browse'>('search')
  const [page, setPage] = useState(1)
  const [selectedDoc, setSelectedDoc] = useState<DocumentSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<string>('filename')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [filters, setFilters] = useState({ authority: '', country: '', riskCategory: '' })
  const [expandedAuthority, setExpandedAuthority] = useState<string | null>(null)

  const pageSize = 20

  useEffect(() => {
    apiClient.listDocuments().then((res) => {
      setDocs(res.documents)
      setFilteredDocs(res.documents)
      setLoading(false)
    })
  }, [])

  useEffect(() => {
    const result = docs.filter((doc) => {
      const matchesSearch = !searchTerm || doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) || doc.authority?.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesAuthority = !filters.authority || doc.authority === filters.authority
      const matchesCountry = !filters.country || doc.country === filters.country
      const matchesRisk = !filters.riskCategory || doc.risk_category === filters.riskCategory
      return matchesSearch && matchesAuthority && matchesCountry && matchesRisk
    })

    result.sort((a, b) => {
      const aVal = a[sortBy as keyof DocumentSummary] ?? ''
      const bVal = b[sortBy as keyof DocumentSummary] ?? ''
      return sortDir === 'asc' ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal))
    })

    setFilteredDocs(result)
    setPage(1)
  }, [searchTerm, docs, sortBy, sortDir, filters])

  const paginatedDocs = filteredDocs.slice((page - 1) * pageSize, page * pageSize)
  const authorities = [...new Set(docs.map(d => d.authority))]
  const countries = [...new Set(docs.map(d => d.country))]
  const riskCategories = [...new Set(docs.map(d => d.risk_category))]

  const columns = [
    {
      key: 'filename' as const,
      label: 'Regulation',
      sortable: true,
      render: (val: string | number | undefined) => <span className="font-medium">{val || '—'}</span>,
    },
    {
      key: 'authority' as const,
      label: 'Authority',
      sortable: true,
    },
    {
      key: 'country' as const,
      label: 'Country',
      sortable: true,
      render: (val: string | number | undefined) => <Badge>{val || 'N/A'}</Badge>,
    },
    {
      key: 'risk_category' as const,
      label: 'Risk',
      sortable: true,
      render: (val: string | number | undefined) => <span className="text-sm">{val || '—'}</span>,
    },
    {
      key: 'effective_date' as const,
      label: 'Effective Date',
      sortable: true,
      render: (val: string | number | undefined) => <span className="text-sm text-slate-400">{val || '—'}</span>,
    },
  ]

  const tabsData = [
    {
      id: 'search',
      label: 'Search',
      content: (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Filters Sidebar */}
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 }}>
            <Card className="h-fit">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-semibold text-slate-300 block mb-2">Authority</label>
                  <select
                    value={filters.authority}
                    onChange={(e) => setFilters({ ...filters, authority: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300"
                  >
                    <option value="">All Authorities</option>
                    {authorities.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm font-semibold text-slate-300 block mb-2">Country</label>
                  <select
                    value={filters.country}
                    onChange={(e) => setFilters({ ...filters, country: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300"
                  >
                    <option value="">All Countries</option>
                    {countries.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm font-semibold text-slate-300 block mb-2">Risk Category</label>
                  <select
                    value={filters.riskCategory}
                    onChange={(e) => setFilters({ ...filters, riskCategory: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300"
                  >
                    <option value="">All Risks</option>
                    {riskCategories.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Search & Results */}
          <div className="lg:col-span-3 space-y-4">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
              <Search value={searchTerm} onChange={setSearchTerm} placeholder="Search regulations, circulars, acts..." />
            </motion.div>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
              <Card>
                {loading ? (
                  <div className="space-y-3">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="h-12 bg-slate-800 rounded animate-pulse" />
                    ))}
                  </div>
                ) : (
                  <>
                    <Table
                      columns={columns}
                      data={paginatedDocs}
                      onSort={(key, dir) => {
                        setSortBy(key)
                        setSortDir(dir)
                      }}
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onRowClick={(row) => setSelectedDoc(row)}
                    />
                    <div className="mt-6 pt-6 border-t border-slate-800">
                      <Pagination page={page} pageSize={pageSize} total={filteredDocs.length} onPageChange={setPage} />
                    </div>
                  </>
                )}
              </Card>
            </motion.div>
          </div>
        </div>
      ),
    },
    {
      id: 'browse',
      label: 'Browse',
      content: (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          <Card>
            <div className="space-y-2">
              {authorities.map((auth) => (
                <div key={auth}>
                  <button
                    onClick={() => setExpandedAuthority(expandedAuthority === auth ? null : (auth || null))}
                    className="w-full flex items-center gap-2 p-3 hover:bg-slate-800/50 rounded-lg text-left transition-colors"
                  >
                    <ChevronDown className={`w-4 h-4 transition-transform ${expandedAuthority === auth ? '' : '-rotate-90'}`} />
                    <span className="font-medium">{auth}</span>
                  </button>
                  {expandedAuthority === auth && (
                    <div className="pl-6 space-y-1 pb-2">
                      {docs
                        .filter((d) => d.authority === auth)
                        .map((d) => (
                          <button
                            key={d.id}
                            onClick={() => setSelectedDoc(d)}
                            className="w-full text-left p-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded transition-colors"
                          >
                            {d.filename}
                          </button>
                        ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      ),
    },
  ]

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold">Regulations</h1>
        <p className="text-slate-400 mt-1">Search and browse active regulations and compliance documents</p>
      </motion.div>

      {/* Tabs */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
        <Tabs tabs={tabsData} activeId={mode} onTabChange={(id) => setMode(id as 'search' | 'browse')} />
      </motion.div>

      {/* Detail Drawer */}
      <Drawer isOpen={!!selectedDoc} onClose={() => setSelectedDoc(null)} title="Regulation Details" size="lg">
        {selectedDoc && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">{selectedDoc.filename}</h2>
              <p className="text-slate-400">{selectedDoc.regulation}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Authority</p>
                <p className="font-semibold">{selectedDoc.authority || '—'}</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Country</p>
                <p className="font-semibold">{selectedDoc.country || '—'}</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Risk Category</p>
                <p className="font-semibold">{selectedDoc.risk_category || '—'}</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Status</p>
                <p className="font-semibold text-emerald-400">Current</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-slate-400 mb-2">Effective Date</p>
                <p>{selectedDoc.effective_date || '—'}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 mb-2">Version</p>
                <p>{selectedDoc.version || '—'}</p>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800 flex gap-2">
              <Button variant="secondary">
                <Download className="w-4 h-4" />
                Export as PDF
              </Button>
            </div>
          </div>
        )}
      </Drawer>
    </div>
  )
}

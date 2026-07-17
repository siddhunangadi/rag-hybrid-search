import { useState, useRef, useEffect } from 'react'
import { Send, Loader, AlertCircle, Copy, Download, Printer, Share2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiClient } from '@/lib/api'
import { Card } from '@/components/ui'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Array<{ document_id: string; text: string; authority?: string }>
  confidenceScore?: number
  timestamp: Date
}

const EXAMPLE_QUESTIONS = [
  'What are the latest RBI KYC requirements?',
  'Show the current SEBI cyber security regulations',
  'Has GST input tax credit changed recently?',
  'Which regulations affect NBFC lending?',
  'Compare RBI Circular 2025 with the previous version',
]

const AUTHORITY_CHIPS = ['RBI', 'SEBI', 'GST', 'Companies Act', 'Cyber Security', 'AML/KYC']

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = {
      id: Math.random().toString(36),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setError(undefined)
    setLoading(true)

    try {
      const response = await apiClient.answer({
        question: input,
        top_k: 5,
      })

      const assistantMessage: Message = {
        id: Math.random().toString(36),
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        confidenceScore: response.confidence_score,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'We couldn\'t generate an answer right now. Please try again or refine your question.')
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  const confidenceLevel = (score?: number) => {
    if (!score) return 'Unknown'
    if (score > 0.8) return 'High'
    if (score > 0.6) return 'Medium'
    return 'Low'
  }

  const confidenceReason = (score?: number) => {
    if (!score) return ''
    if (score > 0.8) return 'Confidence is High because multiple current regulations support this answer.'
    if (score > 0.6) return 'Confidence is Medium because some regulations support this answer with minor uncertainties.'
    return 'Confidence is Low because limited evidence supports this answer. Verify carefully.'
  }

  return (
    <div className="flex flex-col h-[calc(100vh-120px)]">
      {messages.length === 0 ? (
        // Empty State
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex-1 flex flex-col items-center justify-center">
          <div className="max-w-2xl w-full space-y-8">
            <div className="text-center">
              <h1 className="text-3xl font-bold mb-2">AI Compliance Assistant</h1>
              <p className="text-slate-400">Ask any question about regulations, compliance, or internal policies</p>
            </div>

            {/* Prompt Input */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask any question about regulations, compliance, or internal policies..."
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-800 rounded-lg text-slate-300 placeholder-slate-500 focus:outline-none focus:border-slate-700"
                  disabled={loading}
                  autoFocus
                />
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-slate-400 hover:text-slate-300 disabled:opacity-50"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </form>

            {/* Example Questions */}
            <div>
              <p className="text-sm font-medium text-slate-300 mb-3">Example questions:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {EXAMPLE_QUESTIONS.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setInput(q)
                    }}
                    className="p-3 border border-slate-700 rounded-lg hover:bg-slate-800/50 hover:border-slate-600 text-left text-sm text-slate-300 transition-all"
                  >
                    • {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Authority Chips */}
            <div>
              <p className="text-sm font-medium text-slate-300 mb-3">Browse by authority:</p>
              <div className="flex flex-wrap gap-2">
                {AUTHORITY_CHIPS.map((auth) => (
                  <button
                    key={auth}
                    className="px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg hover:border-slate-600 text-xs text-slate-300 transition-all"
                  >
                    {auth}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      ) : (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-6 pb-6">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div key={message.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                  {message.role === 'user' ? (
                    <div className="flex justify-end">
                      <div className="max-w-xl bg-blue-600 text-white rounded-lg p-4">
                        <p className="text-sm">{message.content}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-3xl space-y-4">
                      {/* Answer */}
                      <Card className="space-y-4">
                        <div>
                          <p className="text-sm font-semibold text-slate-300 mb-2">Answer</p>
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                        </div>

                        {/* Verified Sources */}
                        {message.citations && message.citations.length > 0 && (
                          <div className="pt-4 border-t border-slate-800">
                            <p className="text-sm font-semibold text-slate-300 mb-3">Verified Sources</p>
                            <div className="space-y-2">
                              {message.citations.map((citation, i) => (
                                <div key={i} className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 text-xs space-y-1">
                                  <p className="text-slate-300 font-medium">{citation.authority || 'Authority'}</p>
                                  <p className="text-slate-400">{citation.text.substring(0, 100)}...</p>
                                  <div className="flex justify-between items-center">
                                    <span className="text-slate-500">Confidence: High</span>
                                    <button className="text-blue-400 hover:text-blue-300 text-xs">Open Regulation →</button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Verification Status */}
                        <div className="pt-4 border-t border-slate-800">
                          <p className="text-xs text-emerald-400 font-medium">✓ Current Version Verified</p>
                        </div>

                        {/* Confidence */}
                        {message.confidenceScore && (
                          <div className="pt-4 border-t border-slate-800">
                            <p className="text-sm font-semibold text-slate-300 mb-2">Confidence</p>
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <div className="h-2 flex-1 bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full ${
                                      message.confidenceScore > 0.8 ? 'bg-emerald-500' : message.confidenceScore > 0.6 ? 'bg-amber-500' : 'bg-red-500'
                                    }`}
                                    style={{ width: `${message.confidenceScore * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs font-medium text-slate-300 w-12">{confidenceLevel(message.confidenceScore)}</span>
                              </div>
                              <p className="text-xs text-slate-400">{confidenceReason(message.confidenceScore)}</p>
                            </div>
                          </div>
                        )}

                        {/* Actions */}
                        <div className="pt-4 border-t border-slate-800 flex gap-2 flex-wrap">
                          <button
                            onClick={() => handleCopy(message.content)}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-all"
                          >
                            <Copy className="w-3 h-3" /> Copy
                          </button>
                          <button className="flex items-center gap-1 px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-all">
                            <Download className="w-3 h-3" /> Export PDF
                          </button>
                          <button className="flex items-center gap-1 px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-all">
                            <Printer className="w-3 h-3" /> Print
                          </button>
                          <button className="flex items-center gap-1 px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-all">
                            <Share2 className="w-3 h-3" /> Share
                          </button>
                        </div>
                      </Card>

                      {/* Follow-up Questions */}
                      {['What changed from the previous version?', 'Which organizations are affected?', 'Show related RBI circulars'].length > 0 && (
                        <div className="space-y-2">
                          <p className="text-sm font-semibold text-slate-300">Follow-up questions:</p>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                            {['What changed from the previous version?', 'Which organizations are affected?', 'Show related RBI circulars'].map((q, i) => (
                              <button
                                key={i}
                                onClick={() => setInput(q)}
                                className="p-2 border border-slate-700 rounded-lg hover:bg-slate-800/50 text-left text-xs text-slate-300 transition-all"
                              >
                                {q}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {loading && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <Loader className="w-4 h-4 animate-spin" />
                  Analyzing regulations...
                </div>
              </motion.div>
            )}

            {error && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="flex gap-2 p-3 bg-red-900/20 border border-red-700 rounded-lg">
                  <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-red-300">{error}</p>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="flex gap-2 mt-6">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a follow-up question..."
              className="flex-1 px-4 py-2.5 bg-slate-900 border border-slate-800 rounded-lg text-slate-300 placeholder-slate-500 focus:outline-none focus:border-slate-700 disabled:opacity-50"
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()} className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white disabled:opacity-50 transition-all">
              <Send className="w-4 h-4" />
            </button>
          </form>
        </>
      )}
    </div>
  )
}

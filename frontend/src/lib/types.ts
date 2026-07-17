// API Response Types
export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  timestamp: string
}

export interface DocumentSummary {
  id: string
  filename: string
  uploaded_at: string
  chunk_count: number
  size_bytes: number
  authority?: string
  regulation?: string
  version?: string
  effective_date?: string
  country?: string
  risk_category?: string
}

export interface DocumentsResponse {
  documents: DocumentSummary[]
  total: number
}

export interface AnswerRequest {
  question: string
  top_k?: number
  metadata_filters?: Record<string, unknown>
}

export interface Citation {
  chunk_id: string
  document_id: string
  text: string
  metadata?: Record<string, unknown>
}

export interface RagAnswer {
  answer: string
  citations: Citation[]
  confidence_score: number
  retrieval_latency_ms: number
  generation_latency_ms: number
  retrieved_chunks: ContextChunk[]
}

export interface ContextChunk {
  chunk_id: string
  document_id: string
  text: string
  score: number
  metadata: Record<string, unknown>
}

export interface AuditEvent {
  event_id: string
  event_type: string
  timestamp: string
  request_id: string
  key_id: string
  role: string
  endpoint: string
  action: string
  status: 'success' | 'failure'
  metadata?: Record<string, unknown>
}

export interface AuditEventsResponse {
  events: AuditEvent[]
  total: number
}

export interface JobStatusResponse {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  result?: Record<string, unknown>
  error?: string
}

export interface DiagnosticsResponse {
  version: string
  python_version: string
  provider: string
  rerank_backend: string
  storage_backend: string
  pinecone_ready: boolean
  bm25_ready: boolean
  audit_ready: boolean
  embedding_provider_ready: boolean
  request_count: number
  error_count: number
  avg_response_latency_ms: number
  config_summary: Record<string, unknown>
}

export interface LivenessResponse {
  status: 'ok'
  timestamp: string
}

export interface ReadinessResponse {
  ready: boolean
  checks: Array<{
    name: string
    ok: boolean
    detail?: string
  }>
}

export interface MetricsResponse {
  request_count: number
  error_count: number
  avg_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
}

/// <reference types="vite/client" />
import axios, { AxiosInstance } from 'axios'
import * as Types from './types'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      headers: {
        'Content-Type': 'application/json',
        'X-Debug-Token': localStorage.getItem('debugToken') || '',
      },
    })
  }

  // Health
  async getHealth(): Promise<Types.HealthResponse> {
    const { data } = await this.client.get('/health')
    return data
  }

  async getLiveness(): Promise<Types.LivenessResponse> {
    const { data } = await this.client.get('/health/live')
    return data
  }

  async getReadiness(): Promise<Types.ReadinessResponse> {
    const { data } = await this.client.get('/health/ready')
    return data
  }

  // Documents
  async listDocuments(): Promise<Types.DocumentsResponse> {
    const { data } = await this.client.get('/documents')
    return data
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/documents/${documentId}`)
  }

  async uploadDocuments(files: File[], metadata: Record<string, unknown>): Promise<Types.JobStatusResponse> {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    Object.entries(metadata).forEach(([key, value]) => {
      formData.append(key, String(value))
    })

    const { data } = await this.client.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  }

  async uploadDocumentsAsync(files: File[], metadata: Record<string, unknown>): Promise<Types.JobStatusResponse> {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    Object.entries(metadata).forEach(([key, value]) => {
      formData.append(key, String(value))
    })

    const { data } = await this.client.post('/upload/async', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  }

  // Chat
  async answer(request: Types.AnswerRequest): Promise<Types.RagAnswer> {
    const { data } = await this.client.post('/answer', request)
    return data
  }

  async *answerStream(request: Types.AnswerRequest): AsyncGenerator<string> {
    const response = await this.client.post('/answer/stream', request, {
      responseType: 'stream',
    })
    const reader = response.data.getReader()
    const decoder = new TextDecoder()

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        yield decoder.decode(value, { stream: true })
      }
    } finally {
      reader.releaseLock()
    }
  }

  // Jobs
  async getJobStatus(jobId: string): Promise<Types.JobStatusResponse> {
    const { data } = await this.client.get(`/jobs/${jobId}`)
    return data
  }

  // Audit
  async getAuditEvents(limit = 100, offset = 0): Promise<Types.AuditEventsResponse> {
    const { data } = await this.client.get('/audit/events', {
      params: { limit, offset },
    })
    return data
  }

  // Diagnostics
  async getDiagnostics(): Promise<Types.DiagnosticsResponse> {
    const { data } = await this.client.get('/diagnostics')
    return data
  }

  // Metrics
  async getMetrics(): Promise<Types.MetricsResponse> {
    const { data } = await this.client.get('/metrics')
    return data
  }

  // Version
  async getVersion(): Promise<{ version: string }> {
    const { data } = await this.client.get('/version')
    return data
  }
}

export const apiClient = new ApiClient()

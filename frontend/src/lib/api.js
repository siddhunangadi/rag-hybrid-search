/// <reference types="vite/client" />
import axios from 'axios';
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
class ApiClient {
    constructor() {
        Object.defineProperty(this, "client", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.client = axios.create({
            baseURL: BASE_URL,
            headers: {
                'Content-Type': 'application/json',
                'X-Debug-Token': localStorage.getItem('debugToken') || '',
            },
        });
    }
    // Health
    async getHealth() {
        const { data } = await this.client.get('/health');
        return data;
    }
    async getLiveness() {
        const { data } = await this.client.get('/health/live');
        return data;
    }
    async getReadiness() {
        const { data } = await this.client.get('/health/ready');
        return data;
    }
    // Documents
    async listDocuments() {
        const { data } = await this.client.get('/documents');
        return data;
    }
    async deleteDocument(documentId) {
        await this.client.delete(`/documents/${documentId}`);
    }
    async uploadDocuments(files, metadata) {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        Object.entries(metadata).forEach(([key, value]) => {
            formData.append(key, String(value));
        });
        const { data } = await this.client.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return data;
    }
    async uploadDocumentsAsync(files, metadata) {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        Object.entries(metadata).forEach(([key, value]) => {
            formData.append(key, String(value));
        });
        const { data } = await this.client.post('/upload/async', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return data;
    }
    // Chat
    async answer(request) {
        const { data } = await this.client.post('/answer', request);
        return data;
    }
    async *answerStream(request) {
        const response = await this.client.post('/answer/stream', request, {
            responseType: 'stream',
        });
        const reader = response.data.getReader();
        const decoder = new TextDecoder();
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done)
                    break;
                yield decoder.decode(value, { stream: true });
            }
        }
        finally {
            reader.releaseLock();
        }
    }
    // Jobs
    async getJobStatus(jobId) {
        const { data } = await this.client.get(`/jobs/${jobId}`);
        return data;
    }
    // Audit
    async getAuditEvents(limit = 100, offset = 0) {
        const { data } = await this.client.get('/audit/events', {
            params: { limit, offset },
        });
        return data;
    }
    // Diagnostics
    async getDiagnostics() {
        const { data } = await this.client.get('/diagnostics');
        return data;
    }
    // Metrics
    async getMetrics() {
        const { data } = await this.client.get('/metrics');
        return data;
    }
    // Version
    async getVersion() {
        const { data } = await this.client.get('/version');
        return data;
    }
}
export const apiClient = new ApiClient();

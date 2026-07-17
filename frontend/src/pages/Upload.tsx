import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { Upload as UploadIcon, X, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiClient } from '@/lib/api'
import { Card, Button, Input } from '@/components/ui'

const metadataSchema = z.object({
  authority: z.string().min(1, 'Authority required'),
  regulation: z.string().min(1, 'Regulation required'),
  version: z.string().min(1, 'Version required'),
  effective_date: z.string().min(1, 'Effective date required'),
  country: z.string().min(1, 'Country required'),
  risk_category: z.string().min(1, 'Risk category required'),
  document_type: z.string().min(1, 'Document type required'),
})

type Metadata = z.infer<typeof metadataSchema>

interface UploadItem {
  id: string
  file: File
  progress: number
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

export default function Upload() {
  const [files, setFiles] = useState<UploadItem[]>([])
  const [dragActive, setDragActive] = useState(false)
  const { register, handleSubmit, formState: { errors } } = useForm<Metadata>({
    resolver: zodResolver(metadataSchema),
    defaultValues: {
      authority: '',
      regulation: '',
      version: '1.0',
      effective_date: new Date().toISOString().split('T')[0],
      country: 'IN',
      risk_category: 'COMPLIANCE',
      document_type: 'REGULATION',
    },
  })

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files) {
      addFiles(Array.from(e.dataTransfer.files))
    }
  }

  const addFiles = (newFiles: File[]) => {
    const items: UploadItem[] = newFiles.map((file) => ({
      id: Math.random().toString(36),
      file,
      progress: 0,
      status: 'pending',
    }))
    setFiles((prev) => [...prev, ...items])
  }

  const onSubmit = async (metadata: Metadata) => {
    if (files.length === 0) return

    const pendingFiles = files.filter((f) => f.status === 'pending')
    if (pendingFiles.length === 0) return

    for (const item of pendingFiles) {
      const maxSizeMB = 50
      if (item.file.size > maxSizeMB * 1024 * 1024) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === item.id
              ? { ...f, status: 'error', error: `File exceeds ${maxSizeMB}MB limit` }
              : f
          )
        )
        continue
      }

      const validTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/csv',
        'text/markdown',
        'text/plain',
        'text/html',
      ]
      if (!validTypes.includes(item.file.type) && !item.file.name.match(/\.(pdf|docx|xlsx|csv|md|txt|html)$/i)) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === item.id ? { ...f, status: 'error', error: 'Invalid file type' } : f
          )
        )
        continue
      }
    }

    const uploadPromises = pendingFiles
      .filter((item) => files.find((f) => f.id === item.id)?.status === 'pending')
      .map(async (item) => {
        try {
          setFiles((prev) =>
            prev.map((f) => (f.id === item.id ? { ...f, status: 'uploading', progress: 25 } : f))
          )

          await apiClient.uploadDocuments([item.file], metadata)

          setFiles((prev) =>
            prev.map((f) =>
              f.id === item.id ? { ...f, status: 'success', progress: 100 } : f
            )
          )
        } catch (err) {
          setFiles((prev) =>
            prev.map((f) =>
              f.id === item.id
                ? { ...f, status: 'error', error: err instanceof Error ? err.message : 'Upload failed' }
                : f
            )
          )
        }
      })

    await Promise.all(uploadPromises)
  }

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }

  return (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Upload Center</h1>
        <p className="text-slate-400 mt-1">Upload regulatory documents with metadata</p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
        {/* Drag & Drop */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
            dragActive ? 'border-blue-500 bg-blue-50/5' : 'border-slate-700 hover:border-slate-600'
          }`}
        >
          <UploadIcon className="w-12 h-12 mx-auto mb-4 text-slate-400" />
          <h3 className="text-lg font-semibold mb-2">Drag documents here</h3>
          <p className="text-slate-400 mb-4">or select files to upload (PDF, DOCX, XLSX, CSV, MD, TXT, HTML)</p>
          <label className="inline-block">
            <input
              type="file"
              multiple
              className="hidden"
              onChange={(e) => e.target.files && addFiles(Array.from(e.target.files))}
              accept=".pdf,.docx,.xlsx,.csv,.md,.txt,.html"
            />
            <Button>Select Files</Button>
          </label>
        </div>

        {/* Metadata Form */}
        <Card>
          <h3 className="text-lg font-semibold mb-4">Document Metadata</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Authority *</label>
              <Input {...register('authority')} placeholder="RBI / SEBI / Income Tax" />
              {errors.authority && <p className="text-red-400 text-xs mt-1">{errors.authority.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Regulation *</label>
              <Input {...register('regulation')} placeholder="Circular 2024" />
              {errors.regulation && <p className="text-red-400 text-xs mt-1">{errors.regulation.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Version *</label>
              <Input {...register('version')} placeholder="1.0" />
              {errors.version && <p className="text-red-400 text-xs mt-1">{errors.version.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Effective Date *</label>
              <Input {...register('effective_date')} type="date" />
              {errors.effective_date && <p className="text-red-400 text-xs mt-1">{errors.effective_date.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Country *</label>
              <select {...register('country')} className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300 focus:outline-none focus:border-slate-600">
                <option value="IN">India</option>
                <option value="US">United States</option>
                <option value="GB">United Kingdom</option>
              </select>
              {errors.country && <p className="text-red-400 text-xs mt-1">{errors.country.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Risk Category *</label>
              <select {...register('risk_category')} className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300 focus:outline-none focus:border-slate-600">
                <option value="COMPLIANCE">Compliance</option>
                <option value="OPERATIONAL">Operational</option>
                <option value="FINANCIAL">Financial</option>
                <option value="REPUTATIONAL">Reputational</option>
              </select>
              {errors.risk_category && <p className="text-red-400 text-xs mt-1">{errors.risk_category.message}</p>}
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300 block mb-2">Document Type *</label>
              <select {...register('document_type')} className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-300 focus:outline-none focus:border-slate-600">
                <option value="REGULATION">Regulation</option>
                <option value="CIRCULAR">Circular</option>
                <option value="GUIDELINE">Guideline</option>
                <option value="POLICY">Policy</option>
              </select>
              {errors.document_type && <p className="text-red-400 text-xs mt-1">{errors.document_type.message}</p>}
            </div>
          </div>
        </Card>

        {/* Upload Queue */}
        {files.length > 0 && (
          <Card>
            <h3 className="text-lg font-semibold mb-4">Upload Queue ({files.length})</h3>
            <div className="space-y-3">
              <AnimatePresence>
                {files.map((file) => (
                  <motion.div
                    key={file.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="border border-slate-700 rounded p-4 space-y-2"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="font-medium text-sm">{file.file.name}</p>
                        <p className="text-xs text-slate-400">{(file.file.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                      {file.status === 'success' && <CheckCircle className="w-5 h-5 text-emerald-500" />}
                      {file.status === 'error' && <AlertCircle className="w-5 h-5 text-red-500" />}
                      {file.status === 'uploading' && <Loader className="w-5 h-5 text-blue-500 animate-spin" />}
                      <button
                        type="button"
                        onClick={() => removeFile(file.id)}
                        className="ml-2 text-slate-400 hover:text-slate-300"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>

                    {file.status === 'uploading' && (
                      <div className="bg-slate-800 rounded h-2 overflow-hidden">
                        <div className="bg-blue-500 h-full transition-all" style={{ width: `${file.progress}%` }} />
                      </div>
                    )}

                    {file.error && <p className="text-red-400 text-sm">{file.error}</p>}

                    {file.status === 'success' && (
                      <p className="text-emerald-400 text-sm">Upload complete</p>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </Card>
        )}

        {/* Submit */}
        <Button disabled={files.length === 0}>
          Upload {files.length > 0 ? `(${files.length})` : ''} Document{files.length !== 1 ? 's' : ''}
        </Button>
      </form>
    </motion.div>
  )
}

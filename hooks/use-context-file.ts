"use client"

import { useState, useCallback } from 'react'
import { upload } from '@vercel/blob/client'
import { validateContextFile } from '@/backend/validation'
import { buildAuthHeaders } from '@/lib/api-utils'

export interface ContextFileInfo {
  name: string
  type: string
  size: number
}

export function useContextFile(authToken?: string | null) {
  const [contextFile, setContextFile] = useState<ContextFileInfo | null>(null)
  const [contextBlobUrl, setContextBlobUrl] = useState<string | null>(null)
  const [extractedText, setExtractedText] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const uploadContextFile = useCallback(async (file: File) => {
    setError(null)

    const validation = validateContextFile({
      name: file.name,
      type: file.type,
      size: file.size,
    })

    if (!validation.valid) {
      setError(validation.error)
      return
    }

    setIsExtracting(true)
    setContextFile({ name: file.name, type: file.type, size: file.size })

    try {
      // Upload to Vercel Blob
      const blob = await upload(file.name, file, {
        access: 'public',
        handleUploadUrl: '/api/upload',
      })
      setContextBlobUrl(blob.url)

      // Call extract-context API
      const response = await fetch('/api/extract-context', {
        method: 'POST',
        headers: buildAuthHeaders(authToken),
        body: JSON.stringify({
          blobUrl: blob.url,
          fileName: file.name,
        }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || `Extraction failed with status ${response.status}`)
      }

      const { text } = await response.json()
      setExtractedText(text)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to extract text from file'
      setError(message)
      setContextFile(null)
      setExtractedText(null)
    } finally {
      setIsExtracting(false)
    }
  }, [authToken])

  const removeContextFile = useCallback(() => {
    setContextFile(null)
    setContextBlobUrl(null)
    setExtractedText(null)
    setError(null)
    setIsExtracting(false)
  }, [])

  return {
    contextFile,
    contextBlobUrl,
    extractedText,
    isExtracting,
    error,
    uploadContextFile,
    removeContextFile,
  }
}

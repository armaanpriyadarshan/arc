"use client"

import { useState, useRef, useCallback } from 'react'
import { upload } from '@vercel/blob/client'
import { validateFile } from '@/backend/validation'
import { shouldExtractClientSide, extractAudioClientSide } from '@/lib/client-audio'
import { buildAuthHeaders } from '@/lib/api-utils'
import type { Message, Attachment } from '@/hooks/use-message-context'
import { generateId } from '@/hooks/use-message-context'
import type { DeliveryAnalytics } from '@/lib/delivery-analytics'

interface UseTranscriptionDeps {
  messagesRef: React.RefObject<Message[]>
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  setError: (error: string | null) => void
  authTokenRef: React.RefObject<string | null>
}

export function useTranscription(deps: UseTranscriptionDeps) {
  const { messagesRef, setMessages, setError, authTokenRef } = deps
  const [isCompressing, setIsCompressing] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [transcript, setTranscript] = useState<string | null>(null)
  const transcriptRef = useRef<string | null>(null)
  transcriptRef.current = transcript

  const [deliveryAnalytics, setDeliveryAnalytics] = useState<DeliveryAnalytics | null>(null)
  const deliveryAnalyticsRef = useRef<DeliveryAnalytics | null>(null)
  deliveryAnalyticsRef.current = deliveryAnalytics

  const abortControllerRef = useRef<AbortController | null>(null)

  function abortInFlight() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
  }

  const uploadFile = useCallback(
    async (
      file: File,
      callbacks: {
        onTranscriptReady: (messages: Message[], transcript: string, analytics: DeliveryAnalytics | null) => Promise<void>
      }
    ) => {
      const validation = validateFile({
        name: file.name,
        type: file.type,
        size: file.size,
      })

      if (!validation.valid) {
        setError(validation.error)
        return
      }

      abortInFlight()
      setIsTranscribing(true)
      setError(null)

      const uploadMessage: Message = {
        id: generateId(),
        role: 'user',
        content: 'Uploaded a recording for review',
        attachment: {
          name: file.name,
          type: file.type,
          size: file.size,
        } as Attachment,
      }

      const updatedMessages = [...messagesRef.current, uploadMessage]
      messagesRef.current = updatedMessages
      setMessages(updatedMessages)

      const controller = new AbortController()
      abortControllerRef.current = controller

      try {
        let fileToUpload: File = file
        if (shouldExtractClientSide(file)) {
          setIsCompressing(true)
          const compressed = await extractAudioClientSide(file)
          setIsCompressing(false)
          if (compressed) {
            fileToUpload = compressed
          }
        }

        let blob: { url: string }
        try {
          blob = await upload(fileToUpload.name, fileToUpload, {
            access: 'public',
            handleUploadUrl: '/api/upload',
          })
        } catch {
          await new Promise((r) => setTimeout(r, 1000))
          blob = await upload(fileToUpload.name, fileToUpload, {
            access: 'public',
            handleUploadUrl: '/api/upload',
          })
        }

        const response = await fetch('/api/transcribe', {
          method: 'POST',
          headers: buildAuthHeaders(authTokenRef.current),
          body: JSON.stringify({ blobUrl: blob.url, fileName: fileToUpload.name }),
          signal: controller.signal,
        })

        if (!response.ok) {
          const err = await response.json().catch(() => ({}))
          throw new Error(
            err.error || `Transcription failed with status ${response.status}`
          )
        }

        const data = await response.json()
        const newTranscript = data.transcript as string
        const newAnalytics = (data.analytics as DeliveryAnalytics) ?? null

        // Store delivery analytics (latest transcription wins)
        deliveryAnalyticsRef.current = newAnalytics
        setDeliveryAnalytics(newAnalytics)

        // Append to existing transcript (multiple recordings in presentation mode)
        const combined = transcriptRef.current
          ? `${transcriptRef.current}\n\n${newTranscript}`
          : newTranscript
        transcriptRef.current = combined
        setTranscript(combined)
        setIsTranscribing(false)

        const updatedWithTranscript = updatedMessages.map(m =>
          m.id === uploadMessage.id
            ? { ...m, content: `[Presentation transcript]\n${newTranscript}` }
            : m
        )
        messagesRef.current = updatedWithTranscript
        setMessages(updatedWithTranscript)

        await callbacks.onTranscriptReady(updatedWithTranscript, combined, newAnalytics)
      } catch (err: unknown) {
        setIsCompressing(false)
        if (
          err instanceof Error &&
          err.name === 'AbortError' &&
          controller.signal.aborted
        ) {
          setIsTranscribing(false)
          return
        }
        setError(
          err instanceof Error ? err.message : 'Failed to transcribe file'
        )
        setIsTranscribing(false)
      }
    },
    [messagesRef, setMessages, setError, authTokenRef]
  )

  const resetTranscription = useCallback(() => {
    abortInFlight()
    setTranscript(null)
    setDeliveryAnalytics(null)
    setIsCompressing(false)
    setIsTranscribing(false)
  }, [])

  return {
    transcript,
    transcriptRef,
    deliveryAnalytics,
    deliveryAnalyticsRef,
    isCompressing,
    isTranscribing,
    uploadFile,
    abortInFlight,
    resetTranscription,
  }
}

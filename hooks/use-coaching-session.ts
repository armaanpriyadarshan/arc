"use client"

import { useState, useRef, useCallback } from 'react'
import { buildAuthHeaders } from '@/lib/api-utils'
import { parseSSEStream, createRAFBatcher } from '@/lib/sse-utils'
import { useMessageContext, generateId } from '@/hooks/use-message-context'
import { useTranscription } from '@/hooks/use-transcription'
import { useResearchPipeline } from '@/hooks/use-research-pipeline'
import { usePresentationStage } from '@/hooks/use-presentation-stage'
import type { CoachingStage } from '@/lib/coaching-stages'

// Re-export types so existing imports keep working
export type { Attachment, Message } from '@/hooks/use-message-context'
export type { ResearchMeta } from '@/hooks/use-research-pipeline'

export function useCoachingSession(authToken?: string | null) {
  const [slideContext, setSlideContext] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const slideContextRef = useRef<string | null>(null)
  const authTokenRef = useRef<string | null>(null)
  slideContextRef.current = slideContext
  authTokenRef.current = authToken ?? null

  // Sub-hooks
  const {
    messages, setMessages, messagesRef, addMessage, resetMessages,
  } = useMessageContext()

  const stageHook = usePresentationStage()
  const {
    stage, stageRef, setupContext, setupContextRef,
    audiencePulseHistory, appendPulseLabels,
    startPresentation, finishPresentation, resetStage,
  } = stageHook

  const researchHook = useResearchPipeline({ authTokenRef, setupContextRef })
  const {
    researchContext, researchContextRef, researchMeta, researchSearchTerms,
    isResearching, startResearchEarly, resetResearch,
  } = researchHook

  const setErrorSafe = useCallback((err: string | null) => setError(err), [])

  const transcriptionHook = useTranscription({
    messagesRef, setMessages, setError: setErrorSafe, authTokenRef,
  })
  const {
    transcript, transcriptRef, deliveryAnalytics, isCompressing, isTranscribing,
    abortInFlight, resetTranscription,
  } = transcriptionHook

  // ── Streaming ──

  const streamChatResponse = useCallback(
    async (
      currentMessages: ReturnType<typeof useMessageContext>['messages'],
      currentTranscript: string | null,
      overrides?: { stage?: CoachingStage }
    ) => {
      setIsStreaming(true)

      const assistantMessageId = generateId()
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: 'assistant' as const, content: '' },
      ])

      const updateMessageContent = (content: string) =>
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId ? { ...m, content } : m
          )
        )

      const batcher = createRAFBatcher(updateMessageContent)

      try {
        const effectiveStage = overrides?.stage ?? stageRef.current

        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: buildAuthHeaders(authTokenRef.current),
          body: JSON.stringify({
            messages: currentMessages
              .filter((m) => m.content !== '')
              .map((m) => ({ role: m.role, content: m.content })),
            transcript: currentTranscript ?? undefined,
            researchContext: researchContextRef.current ?? undefined,
            slideContext: slideContextRef.current ?? undefined,
            stage: effectiveStage,
            setupContext: setupContextRef.current ?? undefined,
          }),
        })

        if (!response.ok) {
          const err = await response.json().catch(() => ({}))
          throw new Error(
            err.error || `Request failed with status ${response.status}`
          )
        }

        const reader = response.body!.getReader()

        await parseSSEStream<{ content?: string }>(reader, {
          onEvent: (parsed) => {
            if (parsed.content) {
              batcher.append(parsed.content)
            }
          },
        })

        batcher.flush()
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
          batcher.cancel()
          if (batcher.accumulated) {
            updateMessageContent(batcher.accumulated)
          } else {
            setMessages((prev) =>
              prev.filter((m) => m.id !== assistantMessageId)
            )
          }
          return
        }
        batcher.cancel()
        setMessages((prev) =>
          prev.filter((m) => m.id !== assistantMessageId)
        )
        setError(
          err instanceof Error ? err.message : 'Something went wrong'
        )
      } finally {
        setIsStreaming(false)
      }
    },
    [setMessages, stageRef, authTokenRef, researchContextRef, slideContextRef, setupContextRef]
  )

  // ── Send message ──

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim()
      if (!trimmed) return

      abortInFlight()

      const userMessage = {
        id: generateId(),
        role: 'user' as const,
        content: trimmed,
      }

      const updatedMessages = [...messagesRef.current, userMessage]
      messagesRef.current = updatedMessages
      setMessages(updatedMessages)

      await streamChatResponse(updatedMessages, transcriptRef.current)
    },
    [abortInFlight, messagesRef, setMessages, streamChatResponse, transcriptRef]
  )

  // ── Upload file ──

  const uploadFile = useCallback(
    async (file: File) => {
      await transcriptionHook.uploadFile(file, {
        onTranscriptReady: async (updatedMessages, newTranscript, _analytics) => {
          if (stageRef.current === 'present') {
            await streamChatResponse(updatedMessages, newTranscript)
          } else {
            await finishPresentation()
          }
        },
      })
    },
    [transcriptionHook, stageRef, streamChatResponse, finishPresentation]
  )

  // ── Reset ──

  const clearError = useCallback(() => setError(null), [])

  const resetConversation = useCallback(() => {
    abortInFlight()
    resetMessages()
    resetTranscription()
    resetResearch()
    resetStage()
    setSlideContext(null)
    slideContextRef.current = null
    setIsStreaming(false)
    setError(null)
  }, [abortInFlight, resetMessages, resetTranscription, resetResearch, resetStage])

  return {
    messages,
    transcript,
    deliveryAnalytics,
    researchContext,
    researchMeta,
    researchSearchTerms,
    slideContext,
    isCompressing,
    isTranscribing,
    isResearching,
    isStreaming,
    error,
    stage,
    setupContext,
    audiencePulseHistory,
    appendPulseLabels,
    sendMessage,
    uploadFile,
    addMessage,
    setSlideContext,
    clearError,
    resetConversation,
    startPresentation,
    finishPresentation,
    startResearchEarly,
  }
}

"use client"

import { useState, useRef, useCallback } from "react"
import { buildAuthHeaders } from "@/lib/api-utils"
import { parseSSEStream, createRAFBatcher } from "@/lib/sse-utils"
import type { SessionScoresV2 } from "@/lib/sessions"

interface FeedbackStreamState {
  letterText: string
  scores: SessionScoresV2 | null
  isStreaming: boolean
  error: string | null
}

interface StartStreamOptions {
  authToken: string
  sessionId: string
  transcript?: string
  setup: {
    topic: string
    audience: string
    goal: string
    additionalContext?: string
    fileContext?: string
  }
  messages: { role: string; content: string }[]
  researchContext?: string
  slideContext?: string
  deliveryAnalyticsSummary?: string
}

export function useFeedbackStream() {
  const [state, setState] = useState<FeedbackStreamState>({
    letterText: "",
    scores: null,
    isStreaming: false,
    error: null,
  })

  const abortRef = useRef<AbortController | null>(null)

  const startStream = useCallback(async (options: StartStreamOptions) => {
    // Abort any existing stream
    abortRef.current?.abort()

    const controller = new AbortController()
    abortRef.current = controller

    setState({ letterText: "", scores: null, isStreaming: true, error: null })

    try {
      const response = await fetch("/api/feedback-score", {
        method: "POST",
        headers: buildAuthHeaders(options.authToken),
        body: JSON.stringify({
          sessionId: options.sessionId,
          transcript: options.transcript,
          setup: options.setup,
          messages: options.messages,
          researchContext: options.researchContext,
          slideContext: options.slideContext,
          deliveryAnalyticsSummary: options.deliveryAnalyticsSummary,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const reader = response.body!.getReader()
      const batcher = createRAFBatcher((content) =>
        setState((prev) => ({ ...prev, letterText: content }))
      )

      await parseSSEStream<{ type: string; content?: string; scores?: SessionScoresV2; error?: string }>(
        reader,
        {
          onEvent: (parsed) => {
            if (parsed.type === "letter_chunk" && parsed.content) {
              batcher.append(parsed.content)
            } else if (parsed.type === "scores" && parsed.scores) {
              batcher.flush()
              setState((prev) => ({
                ...prev,
                scores: parsed.scores!,
                isStreaming: false,
              }))
            } else if (parsed.type === "error" && parsed.error) {
              setState((prev) => ({
                ...prev,
                error: parsed.error!,
                isStreaming: false,
              }))
            }
          },
          onDone: () => {
            batcher.flush()
          },
        }
      )

      batcher.flush()
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : "Stream failed",
        isStreaming: false,
      }))
    }
  }, [])

  return {
    letterText: state.letterText,
    scores: state.scores,
    isStreaming: state.isStreaming,
    error: state.error,
    startStream,
  }
}

"use client"

import { useState, useCallback, useRef, useEffect } from 'react'
import { upload } from '@vercel/blob/client'
import { validateSlideFile } from '@/backend/validation'
import { buildAuthHeaders } from '@/lib/api-utils'
import { parseSSEStream } from '@/lib/sse-utils'
import type { SlideFeedback, DeckFeedback } from '@/backend/slides'

export type { SlideFeedback, DeckFeedback }

export interface ReviewSnapshot {
  slideFeedbacks: SlideFeedback[]
  deckSummary: DeckFeedback | null
  thumbnails: Record<number, string>
  blobUrl: string
  fileName: string
}

export type AnalysisStep =
  | 'idle'
  | 'uploading'
  | 'downloading'
  | 'rendering'
  | 'analyzing'
  | 'done'
  | 'error'

export interface AnalysisProgress {
  step: AnalysisStep
  slidesCompleted: number
  slidesTotal: number
}

export interface UseSlideReviewReturn {
  slideFeedbacks: SlideFeedback[]
  deckSummary: DeckFeedback | null
  progress: AnalysisProgress
  error: string | null
  panelOpen: boolean
  thumbnails: Record<number, string>
  reviews: Record<string, ReviewSnapshot>
  activeReviewKey: string | null
  displayedKey: string | null
  isAnalyzing: boolean
  blobUrl: string | null
  fileName: string | null
  uploadAndAnalyze: (file: File, audienceContext?: string, reviewKey?: string, existingBlobUrl?: string) => Promise<void>
  reanalyze: (audienceContext: string) => Promise<void>
  openPanel: () => void
  closePanel: () => void
  openReview: (key: string) => void
  markBlobPersisted: (url: string) => void
  reset: () => void
}

const INITIAL_PROGRESS: AnalysisProgress = {
  step: 'idle',
  slidesCompleted: 0,
  slidesTotal: 0,
}

export function useSlideReview(authToken?: string | null): UseSlideReviewReturn {
  const [slideFeedbacks, setSlideFeedbacks] = useState<SlideFeedback[]>([])
  const [deckSummary, setDeckSummary] = useState<DeckFeedback | null>(null)
  const [progress, setProgress] = useState<AnalysisProgress>(INITIAL_PROGRESS)
  const [error, setError] = useState<string | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [thumbnails, setThumbnails] = useState<Record<number, string>>({})
  const [reviews, setReviews] = useState<Record<string, ReviewSnapshot>>({})
  const [activeReviewKey, setActiveReviewKey] = useState<string | null>(null)
  const [displayedKey, setDisplayedKey] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  const storedBlobUrlRef = useRef<string | null>(null)
  const storedFileNameRef = useRef<string | null>(null)
  const activeReviewKeyRef = useRef<string | null>(null)
  // Tracks which review is currently shown — gates display state updates in
  // runAnalysis so switching to a past review doesn't get trampled.
  const displayedKeyRef = useRef<string | null>(null)
  // Always-current progress + thumbnails for the running analysis, regardless
  // of whether it's displayed. Used to restore state when switching back.
  const runningProgressRef = useRef<AnalysisProgress>(INITIAL_PROGRESS)
  const runningThumbnailsRef = useRef<Record<number, string>>({})
  const sessionBlobUrlsRef = useRef<Set<string>>(new Set())
  const abortControllerRef = useRef<AbortController | null>(null)

  const openPanel = useCallback(() => setPanelOpen(true), [])
  const closePanel = useCallback(() => setPanelOpen(false), [])

  const openReview = useCallback((key: string) => {
    setReviews((prev) => {
      const snapshot = prev[key]
      if (!snapshot) return prev

      const isRunning = activeReviewKeyRef.current === key

      setSlideFeedbacks(snapshot.slideFeedbacks)
      setDeckSummary(snapshot.deckSummary)
      displayedKeyRef.current = key
      setDisplayedKey(key)

      if (isRunning) {
        // Restore the live progress tracked by runAnalysis (never 'done')
        setProgress(runningProgressRef.current)
        // Use the rendered thumbnails directly; snapshot may still have {} if
        // the post-analysis patch hasn't run yet
        setThumbnails(
          Object.keys(runningThumbnailsRef.current).length > 0
            ? runningThumbnailsRef.current
            : snapshot.thumbnails
        )
      } else {
        setProgress({
          step: 'done',
          slidesCompleted: snapshot.slideFeedbacks.length,
          slidesTotal: snapshot.slideFeedbacks.length,
        })
        setThumbnails(snapshot.thumbnails)
      }

      storedBlobUrlRef.current = snapshot.blobUrl
      storedFileNameRef.current = snapshot.fileName
      setBlobUrl(snapshot.blobUrl)
      setFileName(snapshot.fileName)
      setError(null)
      setPanelOpen(true)
      return prev
    })
  }, [])

  const reset = useCallback(() => {
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    setSlideFeedbacks([])
    setDeckSummary(null)
    setProgress(INITIAL_PROGRESS)
    setError(null)
    setThumbnails({})
    setReviews({})
    storedBlobUrlRef.current = null
    storedFileNameRef.current = null
    setBlobUrl(null)
    setFileName(null)
    activeReviewKeyRef.current = null
    displayedKeyRef.current = null
    runningProgressRef.current = INITIAL_PROGRESS
    runningThumbnailsRef.current = {}
    setActiveReviewKey(null)
    setDisplayedKey(null)
    setIsAnalyzing(false)
    setPanelOpen(false)
  }, [])

  const runAnalysis = useCallback(
    async (blobUrl: string, fileName: string, audienceContext?: string, reviewKey?: string) => {
      // Abort any in-flight analysis before starting a new one
      abortControllerRef.current?.abort()
      const controller = new AbortController()
      abortControllerRef.current = controller

      const initialProgress: AnalysisProgress = { step: 'downloading', slidesCompleted: 0, slidesTotal: 0 }
      setSlideFeedbacks([])
      setDeckSummary(null)
      setError(null)
      setProgress(initialProgress)
      setIsAnalyzing(true)
      runningProgressRef.current = initialProgress

      if (reviewKey) {
        activeReviewKeyRef.current = reviewKey
        displayedKeyRef.current = reviewKey
        setActiveReviewKey(reviewKey)
        setDisplayedKey(reviewKey)
        setReviews((prev) => ({
          ...prev,
          [reviewKey]: {
            ...(prev[reviewKey] ?? { blobUrl, fileName, thumbnails: {} }),
            slideFeedbacks: [],
            deckSummary: null,
          },
        }))
      }

      const response = await fetch('/api/slides/analyze', {
        method: 'POST',
        headers: buildAuthHeaders(authToken),
        signal: controller.signal,
        body: JSON.stringify({ blobUrl, fileName, audienceContext }),
      })

      if (!response.ok) {
        const json = await response.json().catch(() => ({}))
        throw new Error(json.error ?? `Server error ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response stream available')

      const localFeedbacks: SlideFeedback[] = []
      let localDeckSummary: DeckFeedback | null = null
      let currentProgress: AnalysisProgress = initialProgress
      let streamError: Error | null = null

      await parseSSEStream<{ type: string; data: unknown }>(reader, {
        onEvent: (event) => {
          if (event.type === 'status') {
            const d = event.data as { step: string; total?: number; completed?: number }
            currentProgress = {
              step: d.step as AnalysisStep,
              slidesTotal: d.total ?? 0,
              slidesCompleted: d.completed ?? 0,
            }
            runningProgressRef.current = currentProgress
            if (!reviewKey || displayedKeyRef.current === reviewKey) {
              setProgress(currentProgress)
            }
          } else if (event.type === 'slide_feedback') {
            const feedback = event.data as SlideFeedback
            localFeedbacks.push(feedback)
            currentProgress = { ...currentProgress, slidesCompleted: currentProgress.slidesCompleted + 1 }
            runningProgressRef.current = currentProgress

            if (reviewKey) {
              setReviews((prev) => {
                const existing = prev[reviewKey]
                if (!existing) return prev
                return { ...prev, [reviewKey]: { ...existing, slideFeedbacks: [...localFeedbacks] } }
              })
            }

            if (!reviewKey || displayedKeyRef.current === reviewKey) {
              setSlideFeedbacks((prev) => [...prev, feedback])
              setProgress(currentProgress)
            }
          } else if (event.type === 'deck_summary') {
            localDeckSummary = event.data as DeckFeedback

            if (reviewKey) {
              setReviews((prev) => {
                const existing = prev[reviewKey]
                if (!existing) return prev
                return { ...prev, [reviewKey]: { ...existing, deckSummary: localDeckSummary } }
              })
            }

            if (!reviewKey || displayedKeyRef.current === reviewKey) {
              setDeckSummary(localDeckSummary)
            }
          } else if (event.type === 'error') {
            const d = event.data as { message: string }
            streamError = new Error(d.message)
          }
        },
        onDone: () => {
          currentProgress = { ...currentProgress, step: 'done' }
          runningProgressRef.current = currentProgress
          if (!reviewKey || displayedKeyRef.current === reviewKey) {
            setProgress(currentProgress)
          }
        },
      }, '\n')

      if (streamError) throw streamError

      if (reviewKey) {
        setReviews((prev) => ({
          ...prev,
          [reviewKey]: {
            slideFeedbacks: localFeedbacks,
            deckSummary: localDeckSummary,
            thumbnails: prev[reviewKey]?.thumbnails ?? {},
            blobUrl,
            fileName,
          },
        }))
      }

      setIsAnalyzing(false)
    },
    [authToken]
  )

  const uploadAndAnalyze = useCallback(
    async (file: File, audienceContext?: string, reviewKey?: string, existingBlobUrl?: string) => {
      const validation = validateSlideFile({
        name: file.name,
        type: file.type,
        size: file.size,
      })
      if (!validation.valid) {
        setError(validation.error)
        return
      }

      setSlideFeedbacks([])
      setDeckSummary(null)
      setError(null)
      storedBlobUrlRef.current = null
      storedFileNameRef.current = null
      setBlobUrl(null)
      setFileName(null)
      runningThumbnailsRef.current = {}
      if (reviewKey) {
        activeReviewKeyRef.current = reviewKey
        displayedKeyRef.current = reviewKey
        setActiveReviewKey(reviewKey)
        setDisplayedKey(reviewKey)
      }
      setPanelOpen(true)

      try {
        let resolvedBlobUrl: string

        if (existingBlobUrl) {
          // Reuse an already-uploaded blob (e.g. same file attached as context)
          resolvedBlobUrl = existingBlobUrl
          setProgress({ step: 'downloading', slidesCompleted: 0, slidesTotal: 0 })
        } else {
          setProgress({ step: 'uploading', slidesCompleted: 0, slidesTotal: 0 })
          const blob = await upload(file.name, file, {
            access: 'public',
            handleUploadUrl: '/api/upload',
          })
          resolvedBlobUrl = blob.url
          sessionBlobUrlsRef.current.add(resolvedBlobUrl)
        }

        storedBlobUrlRef.current = resolvedBlobUrl
        storedFileNameRef.current = file.name
        setBlobUrl(resolvedBlobUrl)
        setFileName(file.name)

        const thumbnailPromise = import('@/lib/pdf-thumbnails')
          .then(({ renderPdfThumbnails }) => renderPdfThumbnails(resolvedBlobUrl))
          .catch(() => ({} as Record<number, string>))

        thumbnailPromise.then((t) => {
          // Always store for the running analysis so openReview can access them
          runningThumbnailsRef.current = t
          // Only update live display state if this analysis is currently shown
          if (!reviewKey || displayedKeyRef.current === reviewKey) {
            setThumbnails(t)
          }
          // Patch snapshot immediately if it already exists
          if (reviewKey) {
            setReviews((prev) => {
              const existing = prev[reviewKey]
              if (!existing) return prev
              return { ...prev, [reviewKey]: { ...existing, thumbnails: t } }
            })
          }
        })

        await runAnalysis(resolvedBlobUrl, file.name, audienceContext, reviewKey)

        // Ensure thumbnails are in the snapshot (handles the race where
        // thumbnails resolved before the snapshot was created)
        if (reviewKey) {
          const t = await thumbnailPromise
          setReviews((prev) => {
            const existing = prev[reviewKey]
            if (!existing) return prev
            if (Object.keys(existing.thumbnails).length > 0) return prev
            return { ...prev, [reviewKey]: { ...existing, thumbnails: t } }
          })
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return
        const message =
          err instanceof Error ? err.message : 'An unexpected error occurred'
        setError(message)
        setProgress((p) => ({ ...p, step: 'error' }))
        setIsAnalyzing(false)
      }
    },
    [runAnalysis]
  )

  const reanalyze = useCallback(
    async (audienceContext: string) => {
      const blobUrl = storedBlobUrlRef.current
      const fileName = storedFileNameRef.current
      if (!blobUrl || !fileName) {
        setError('No slide deck loaded. Please upload a file first.')
        return
      }

      try {
        await runAnalysis(blobUrl, fileName, audienceContext, displayedKeyRef.current ?? undefined)
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return
        const message =
          err instanceof Error ? err.message : 'An unexpected error occurred'
        setError(message)
        setProgress((p) => ({ ...p, step: 'error' }))
        setIsAnalyzing(false)
      }
    },
    [runAnalysis]
  )

  const markBlobPersisted = useCallback((url: string) => {
    sessionBlobUrlsRef.current.delete(url)
  }, [])

  const deleteBlobUrls = useCallback((urls: string[]) => {
    if (urls.length === 0) return
    const payload = JSON.stringify({ urls })
    const sent = navigator.sendBeacon(
      '/api/blob/delete',
      new Blob([payload], { type: 'application/json' })
    )
    if (!sent) {
      fetch('/api/blob/delete', {
        method: 'POST',
        headers: buildAuthHeaders(),
        body: payload,
        keepalive: true,
      }).catch(() => {})
    }
  }, [])

  useEffect(() => {
    const onPageHide = () => {
      deleteBlobUrls([...sessionBlobUrlsRef.current])
    }

    window.addEventListener('pagehide', onPageHide)
    return () => {
      window.removeEventListener('pagehide', onPageHide)
      deleteBlobUrls([...sessionBlobUrlsRef.current])
    }
  }, [deleteBlobUrls])

  return {
    slideFeedbacks,
    deckSummary,
    progress,
    error,
    panelOpen,
    thumbnails,
    reviews,
    activeReviewKey,
    displayedKey,
    isAnalyzing,
    blobUrl,
    fileName,
    uploadAndAnalyze,
    reanalyze,
    openPanel,
    closePanel,
    openReview,
    markBlobPersisted,
    reset,
  }
}

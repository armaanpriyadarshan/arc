"use client"

import { useEffect, useRef, useState } from "react"
import { ArrowLeft, Loader2, ChevronDown, FileText, Download } from "lucide-react"
import { motion } from "framer-motion"
import { useFeedbackStream } from "@/hooks/use-feedback-stream"
import { isV2Scores, isStructuredSlideReview, type SessionScoresV2, type SlideReviewData } from "@/lib/sessions"
import { FeedbackLetter } from "@/components/feedback/feedback-letter"
import { ActionItems } from "@/components/feedback/action-items"
import { RubricDetail } from "@/components/feedback/rubric-detail"
import { DeliveryFeedback } from "@/components/feedback/delivery-feedback"
import { SlideReviewSection } from "@/components/feedback/slide-review-section"

export interface TrialFeedbackData {
  setup: { topic: string; audience: string; goal: string; additionalContext?: string; fileContext?: string }
  transcript: string | null
  messages: { role: string; content: string }[]
  researchContext: string | null
  slideContext: string | null
  slideReview: unknown
  deliveryAnalyticsSummary: string | null
}

export function TrialFeedbackView({ data }: { data: TrialFeedbackData }) {
  const feedbackStream = useFeedbackStream()
  const streamStarted = useRef(false)

  useEffect(() => {
    if (streamStarted.current) return
    streamStarted.current = true
    feedbackStream.startStream({
      sessionId: "trial",
      transcript: data.transcript ?? undefined,
      setup: data.setup,
      messages: data.messages
        .filter((m) => m.content?.trim())
        .map((m) => ({ role: m.role, content: m.content })),
      researchContext: data.researchContext ?? undefined,
      slideContext: data.slideContext ?? undefined,
      deliveryAnalyticsSummary: data.deliveryAnalyticsSummary ?? undefined,
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const v2Scores =
    feedbackStream.scores && isV2Scores(feedbackStream.scores)
      ? (feedbackStream.scores as SessionScoresV2)
      : null
  const isStreamingLetter = feedbackStream.isStreaming && feedbackStream.letterText.length > 0

  const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase())
  const headerTitle = v2Scores?.refinedTitle ?? titleCase(data.setup.topic)
  const headerAudience = v2Scores?.refinedAudience ?? titleCase(data.setup.audience)
  const headerGoal = v2Scores?.refinedGoal ?? titleCase(data.setup.goal)
  const date = new Date()

  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false)
  async function handleDownloadPdf() {
    if (!v2Scores) return
    setIsGeneratingPdf(true)
    try {
      const { generatePdfReport } = await import("@/components/feedback/pdf/generate-pdf")
      await generatePdfReport({
        scores: v2Scores,
        setup: data.setup,
        transcript: data.transcript,
        date,
      })
    } catch (err) {
      console.error("[feedback] PDF generation failed:", err)
    } finally {
      setIsGeneratingPdf(false)
    }
  }

  return (
    <div className="relative min-h-screen bg-background pb-24">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden" aria-hidden="true">
        <div
          className="absolute -left-60 -top-60 h-[600px] w-[600px] rounded-full opacity-[0.07] blur-[100px]"
          style={{ background: "radial-gradient(circle, hsl(36 72% 50%), transparent 70%)" }}
        />
        <div
          className="absolute -right-40 top-[40%] h-[500px] w-[500px] rounded-full opacity-[0.04] blur-[100px]"
          style={{ background: "radial-gradient(circle, hsl(142 71% 45%), transparent 70%)" }}
        />
      </div>

      <div className="mx-auto max-w-3xl px-4 pt-12 pb-6 sm:px-6">
        {/* Back */}
        <motion.div
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8"
        >
          <a
            href="/chat"
            className="inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Back to chat"
          >
            <ArrowLeft className="h-4 w-4" />
          </a>
        </motion.div>

        {/* Header */}
        <header>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            {headerTitle ? (
              <div className="flex items-center gap-2">
                <h1 className="font-display text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
                  {headerTitle}
                </h1>
                {v2Scores && (
                  <button
                    type="button"
                    onClick={handleDownloadPdf}
                    disabled={isGeneratingPdf}
                    className="inline-flex items-center justify-center rounded-md border border-border/60 bg-muted/40 p-1 text-muted-foreground transition-colors hover:border-primary/30 hover:text-primary disabled:pointer-events-none disabled:opacity-50"
                    aria-label="Download report"
                  >
                    {isGeneratingPdf ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Download className="h-3.5 w-3.5" />
                    )}
                  </button>
                )}
              </div>
            ) : (
              <div className="h-8 w-2/3 animate-pulse rounded-lg bg-muted/40" />
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                {headerAudience}
              </span>
              <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                {headerGoal}
              </span>
              <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                {date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
              </span>
            </div>
          </motion.div>
        </header>

        {/* Content */}
        <div className="mt-12 space-y-12">
          {v2Scores ? (
            <>
              <FeedbackLetter letter={v2Scores.feedbackLetter} />
              <motion.section
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              >
                <ActionItems items={v2Scores.actionItems ?? []} />
                <div className="mt-4">
                  <RubricDetail
                    rubric={v2Scores.rubric}
                    strongestMoment={v2Scores.strongestMoment}
                    areaToImprove={v2Scores.areaToImprove}
                  />
                </div>
              </motion.section>
              {v2Scores.deliveryFeedback && v2Scores.deliveryFeedback.length > 0 && (
                <DeliveryFeedback observations={v2Scores.deliveryFeedback} />
              )}
            </>
          ) : isStreamingLetter ? (
            <>
              <FeedbackLetter letter={feedbackStream.letterText} />
              <div className="space-y-4">
                <div>
                  <div className="mb-3 h-3.5 w-24 animate-pulse rounded bg-muted/40" />
                  <div className="rounded-xl border border-border/60 bg-card">
                    {[1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className={`flex gap-4 px-5 py-4 ${i > 1 ? "border-t border-border/40" : ""}`}
                      >
                        <div className="h-6 w-6 shrink-0 animate-pulse rounded-full bg-muted/40" />
                        <div className="flex-1 space-y-2">
                          <div className="h-3.5 w-2/5 animate-pulse rounded bg-muted/40" />
                          <div className="h-3 w-4/5 animate-pulse rounded bg-muted/40" />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                {[1, 2, 3].map((i) => (
                  <div key={i} className="rounded-xl border border-border/60 bg-card px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="h-3.5 flex-1 animate-pulse rounded bg-muted/40" />
                      <div className="h-5 w-16 animate-pulse rounded-full bg-muted/40" />
                      <div className="h-3.5 w-6 animate-pulse rounded bg-muted/40" />
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="mt-16">
              <div className="flex flex-col items-center gap-4 py-16">
                <div className="relative">
                  <div className="h-12 w-12 animate-spin rounded-full border-[3px] border-muted border-t-primary" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground/80">Analyzing your presentation</p>
                  <p className="mt-1 text-xs text-muted-foreground">Preparing your feedback...</p>
                </div>
              </div>
              <div className="mt-8 space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-20 animate-pulse rounded-xl bg-muted/40" />
                ))}
              </div>
            </div>
          )}

          {/* Slide deck review */}
          {isStructuredSlideReview(data.slideReview as SlideReviewData | null) && (
            <SlideReviewSection slideReview={data.slideReview as SlideReviewData} />
          )}

          {/* Transcript */}
          {data.transcript && <TranscriptSection transcript={data.transcript} />}
        </div>

        {/* Footer */}
        <div className="mt-10 pb-2 text-center">
          <a
            href="/chat"
            className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-muted/40 px-5 py-2.5 text-sm text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
          >
            Start a new session
            <ArrowLeft className="h-3.5 w-3.5 rotate-180" />
          </a>
        </div>
      </div>
    </div>
  )
}

/* ── Collapsible transcript ── */
function TranscriptSection({ transcript }: { transcript: string }) {
  const [open, setOpen] = useState(false)

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 rounded-xl border border-border/60 bg-card px-5 py-4 text-left transition-colors hover:bg-muted/30"
      >
        <FileText className="h-4 w-4 text-muted-foreground/50" />
        <span className="flex-1 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
          Transcript
        </span>
        <ChevronDown
          className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <div className="mt-2 max-h-96 overflow-y-auto rounded-xl border border-border/60 bg-card px-5 py-4">
          <div className="space-y-3 text-sm leading-relaxed text-foreground/60">
            {transcript.split('\n\n').map((block, i) => {
              const labelMatch = block.match(/^\[(.+?)\]:\s*/)
              if (labelMatch) {
                return (
                  <p key={i}>
                    <span className="font-semibold text-foreground/80">{labelMatch[1]}</span>
                    <span className="text-foreground/40">{': '}</span>
                    {block.slice(labelMatch[0].length)}
                  </p>
                )
              }
              return <p key={i}>{block}</p>
            })}
          </div>
        </div>
      )}
    </motion.section>
  )
}

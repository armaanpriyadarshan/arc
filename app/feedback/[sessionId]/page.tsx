"use client"

import { useEffect, useState, useRef, use } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Loader2, ChevronDown, FileText, Download } from "lucide-react"
import { motion } from "framer-motion"
import ReactMarkdown from "react-markdown"
import { useAuth } from "@/contexts/auth-context"
import {
  getSession,
  isV2Scores,
  isStructuredSlideReview,
  type SessionDocument,
  type SessionScoresV2,
} from "@/lib/sessions"
import { FeedbackLetter } from "@/components/feedback/feedback-letter"
import { RubricRadar } from "@/components/feedback/rubric-radar"
import { RubricDetail } from "@/components/feedback/rubric-detail"
import { FollowUpChat } from "@/components/feedback/follow-up-chat"
import { useFollowUpChat } from "@/hooks/use-follow-up-chat"
import { SlideReviewSection } from "@/components/feedback/slide-review-section"
import { DeliveryAnalyticsSection } from "@/components/feedback/delivery-analytics-section"

const SCORE_POLL_INTERVAL = 3000
const SCORE_POLL_MAX_ATTEMPTS = 20

export default function FeedbackPage({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = use(params)
  const router = useRouter()
  const { user, loading: authLoading } = useAuth()
  const [session, setSession] = useState<(SessionDocument & { id: string }) | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [authToken, setAuthToken] = useState<string | null>(null)
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false)
  const pollCount = useRef(0)

  // Get auth token for follow-up chat
  useEffect(() => {
    if (!user) return
    user.getIdToken().then(setAuthToken).catch(() => {})
  }, [user])

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login")
    }
  }, [authLoading, user, router])

  // Load session data from Firestore
  useEffect(() => {
    if (!user) return

    async function load() {
      try {
        const data = await getSession(sessionId, user!.uid)
        if (!data) {
          setLoadError("Session not found")
          return
        }
        setSession(data)
      } catch (err) {
        console.error("[feedback] Failed to load session:", err)
        setLoadError("Failed to load session. Check that Firestore is enabled.")
      } finally {
        setIsLoading(false)
      }
    }

    load()
  }, [sessionId, user])

  // Poll for scores if missing
  useEffect(() => {
    if (!session || session.scores) return
    if (!user) return

    const interval = setInterval(async () => {
      pollCount.current++
      if (pollCount.current > SCORE_POLL_MAX_ATTEMPTS) {
        clearInterval(interval)
        return
      }

      try {
        const updated = await getSession(sessionId, user.uid)
        if (updated?.scores) {
          setSession(updated)
          clearInterval(interval)
        }
      } catch {
        // silently retry
      }
    }, SCORE_POLL_INTERVAL)

    return () => clearInterval(interval)
  }, [session, sessionId, user])

  // Follow-up chat hook
  const followUp = useFollowUpChat({
    authToken: authToken ?? "",
    transcript: session?.transcript ?? null,
    researchContext: session?.researchContext ?? null,
    slideContext: session?.slideReview && "raw" in session.slideReview
      ? (session.slideReview as { raw: string }).raw
      : null,
    setupContext: session?.setup ?? null,
  })

  // Derive V2 scores early so the PDF handler can reference them.
  // Safe even when session is null (returns null).
  const v2ScoresForPdf = session?.scores && isV2Scores(session.scores)
    ? (session.scores as SessionScoresV2)
    : null

  async function handleDownloadPdf() {
    if (!session || !v2ScoresForPdf) return
    setIsGeneratingPdf(true)
    try {
      const { generatePdfReport } = await import("@/components/feedback/pdf/generate-pdf")
      await generatePdfReport({
        scores: v2ScoresForPdf,
        setup: session.setup,
        transcript: session.transcript,
        date: session.createdAt?.toDate?.() ?? new Date(),
        analytics: session.deliveryAnalytics ?? null,
      })
    } catch (err) {
      console.error("[feedback] PDF generation failed:", err)
    } finally {
      setIsGeneratingPdf(false)
    }
  }

  if (authLoading || isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary/60" />
          <p className="text-sm text-muted-foreground">Loading your report...</p>
        </div>
      </div>
    )
  }

  if (loadError || !session) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background">
        <p className="text-foreground">{loadError ?? "Session not found"}</p>
        <a href="/chat" className="text-sm text-primary hover:underline">
          Back to chat
        </a>
      </div>
    )
  }

  const scores = session.scores
  const isV2 = scores && isV2Scores(scores)
  const v2Scores = isV2 ? (scores as SessionScoresV2) : null
  const date = session.createdAt?.toDate?.() ?? new Date()

  // Show refined AI values when available, raw values for old sessions that have
  // scores but lack refined fields, and null (skeleton) while scores are still loading.
  const hasScores = !!scores
  const headerTitle = v2Scores?.refinedTitle ?? (hasScores ? session.setup.topic : null)
  const headerAudience = v2Scores?.refinedAudience ?? (hasScores ? session.setup.audience : null)
  const headerGoal = v2Scores?.refinedGoal ?? (hasScores ? session.setup.goal : null)

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
        {/* ── Top bar: back + download ── */}
        <motion.div
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8 flex items-center justify-between"
        >
          <a
            href="/chat"
            className="inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Back to chat"
          >
            <ArrowLeft className="h-4 w-4" />
          </a>
        </motion.div>

        {/* ── Header ── */}
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
                {v2ScoresForPdf && (
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
              {headerAudience ? (
                <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                  {headerAudience}
                </span>
              ) : (
                <span className="inline-block h-6 w-28 animate-pulse rounded-full bg-muted/40" />
              )}
              {headerGoal ? (
                <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                  {headerGoal}
                </span>
              ) : (
                <span className="inline-block h-6 w-24 animate-pulse rounded-full bg-muted/40" />
              )}
              <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                {date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
              </span>
            </div>
          </motion.div>
        </header>

        {/* ── Content ── */}
        <div className="mt-12 space-y-12">
          {isV2 && v2Scores ? (
            <>
              {/* Feedback Letter */}
              <FeedbackLetter letter={v2Scores.feedbackLetter} />

              {/* Rubric Section */}
              <motion.section
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="rounded-xl border border-border/60 bg-card p-5">
                  <RubricRadar rubric={v2Scores.rubric} />
                </div>
                <div className="mt-4">
                  <RubricDetail
                    rubric={v2Scores.rubric}
                    strongestMoment={v2Scores.strongestMoment}
                    areaToImprove={v2Scores.areaToImprove}
                  />
                </div>
              </motion.section>
            </>
          ) : scores ? (
            /* V1 backward compat: render last assistant message as markdown */
            <V1Fallback session={session} />
          ) : (
            /* Loading state while scores generate */
            <ScoresLoadingState />
          )}

          {/* Delivery analytics (collapsible) */}
          {session.deliveryAnalytics && (
            <DeliveryAnalyticsSection analytics={session.deliveryAnalytics} />
          )}

          {/* Slide deck review (collapsible) */}
          {isStructuredSlideReview(session.slideReview) && (
            <SlideReviewSection slideReview={session.slideReview} />
          )}

          {/* Transcript (collapsible) */}
          {session.transcript && (
            <TranscriptSection transcript={session.transcript} />
          )}

          {/* Follow-up chat */}
          {authToken && (
            <motion.section
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            >
              <FollowUpChat
                messages={followUp.messages}
                isStreaming={followUp.isStreaming}
                onSend={followUp.sendMessage}
              />
            </motion.section>
          )}
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

/* ── V1 backward compat ── */
function V1Fallback({ session }: { session: SessionDocument }) {
  // Find the last assistant message (the feedback) to render
  const feedbackMessage = [...session.messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.content.length > 100)

  if (!feedbackMessage) {
    return (
      <p className="text-sm text-muted-foreground">
        No feedback available for this session.
      </p>
    )
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="prose prose-sm max-w-none text-[0.9375rem] leading-[1.7] text-foreground/90 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:uppercase [&_h3]:tracking-wide [&_strong]:text-foreground [&_blockquote]:border-primary/20 [&_blockquote]:text-foreground/70">
        <ReactMarkdown>{feedbackMessage.content}</ReactMarkdown>
      </div>
    </motion.section>
  )
}

/* ── Scores loading state ── */
function ScoresLoadingState() {
  return (
    <div className="mt-16">
      <div className="flex flex-col items-center gap-4 py-16">
        <div className="relative">
          <div className="h-12 w-12 animate-spin rounded-full border-[3px] border-muted border-t-primary" />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-foreground/80">Analyzing your presentation</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Preparing your feedback...
          </p>
        </div>
      </div>

      {/* Skeleton placeholders */}
      <div className="mt-8 space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 animate-pulse rounded-xl bg-muted/40" />
        ))}
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
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/60">
            {transcript}
          </p>
        </div>
      )}
    </motion.section>
  )
}

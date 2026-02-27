import {
  collection,
  doc,
  setDoc,
  getDoc,
  getDocs,
  updateDoc,
  query,
  where,
  orderBy,
  limit,
  serverTimestamp,
  type Timestamp,
} from "firebase/firestore"
import { db } from "@/lib/firebase"
import { buildAuthHeaders } from "@/lib/api-utils"
import type { SlideFeedback, DeckFeedback } from "@/backend/slides"

/* ── Slide review data (structured) ── */

export interface SlideReviewData {
  raw: string                       // formatted text for scoring API (backward compat)
  deckSummary: DeckFeedback | null
  slideFeedbacks: SlideFeedback[]
  blobUrl?: string                  // Vercel Blob URL of PDF (for re-rendering thumbnails)
  fileName?: string
}

/** Type guard: structured slide review has a `slideFeedbacks` array. */
export function isStructuredSlideReview(
  review: SlideReviewData | { raw: string } | null | undefined
): review is SlideReviewData {
  return !!review && "slideFeedbacks" in review && Array.isArray((review as SlideReviewData).slideFeedbacks)
}

/* ── Score types (V1 — legacy, kept for backward compat) ── */

export interface CategoryScore {
  score: number // 0-100
  summary: string // 2-3 sentences
  evidence: string[] // transcript quotes
  suggestion: string // specific improvement
}

export interface SessionScores {
  overall: number // 0-100
  categories: {
    clarity: CategoryScore
    structure: CategoryScore
    engagement: CategoryScore
    persuasiveness: CategoryScore
    audienceAlignment: CategoryScore
    delivery: CategoryScore
  }
  keyMoments: {
    strongest: { quote: string; why: string }
    weakest: { quote: string; why: string }
  }
  actionItems: {
    priority: number // 1, 2, 3
    title: string
    description: string
    impact: "high" | "medium"
  }[]
}

/* ── Score types (V2 — letter + dynamic rubric) ── */

export interface RubricCriterion {
  name: string          // e.g. "Traction Evidence" or "ROI Clarity"
  score: number         // 0-100
  summary: string       // 2-3 sentences
  evidence: string[]    // transcript quotes
  descriptors?: {       // what each scoring tier means for this criterion
    exceptional: string
    proficient: string
    developing: string
    needsWork: string
  }
}

export interface SessionScoresV2 {
  feedbackLetter: string      // full markdown letter from Vera
  rubric: RubricCriterion[]   // 4-6 audience-specific criteria
  strongestMoment: { quote: string; why: string }
  areaToImprove: { issue: string; suggestion: string }
  refinedTitle?: string       // AI-polished presentation title
  refinedAudience?: string    // AI-polished audience description
  refinedGoal?: string        // AI-polished goal description
}

/** Type guard: V2 scores have a `feedbackLetter` property. */
export function isV2Scores(scores: SessionScores | SessionScoresV2): scores is SessionScoresV2 {
  return "feedbackLetter" in scores
}

/* ── Session document ── */

export interface SessionDocument {
  userId: string
  createdAt: Timestamp
  setup: {
    topic: string
    audience: string
    goal: string
    additionalContext?: string
  }
  transcript: string | null
  messages: { role: string; content: string }[]
  audiencePulse: { text: string; emotion: string }[]
  slideReview: SlideReviewData | { raw: string } | null
  researchContext: string | null
  scores: SessionScores | SessionScoresV2 | null
  deliveryAnalyticsSummary?: string | null
}

export interface SessionSummary {
  id: string
  topic: string
  audience: string
  goal: string
  date: Date
  overallScore: number | null
}

/* ── CRUD ── */

const SESSIONS_COLLECTION = "sessions"

export async function saveSession(
  data: Omit<SessionDocument, "createdAt">
): Promise<string> {
  const ref = doc(collection(db, SESSIONS_COLLECTION))
  await setDoc(ref, {
    ...data,
    createdAt: serverTimestamp(),
  })
  return ref.id
}

export async function getSession(
  sessionId: string,
  userId: string
): Promise<(SessionDocument & { id: string }) | null> {
  const ref = doc(db, SESSIONS_COLLECTION, sessionId)
  const snap = await getDoc(ref)
  if (!snap.exists()) return null
  const data = snap.data() as SessionDocument
  if (data.userId !== userId) return null
  return { ...data, id: snap.id }
}

export async function updateSessionScores(
  sessionId: string,
  scores: SessionScores | SessionScoresV2
): Promise<void> {
  const ref = doc(db, SESSIONS_COLLECTION, sessionId)
  await updateDoc(ref, { scores })
}

export async function deleteSession(
  sessionId: string,
  authToken: string
): Promise<void> {
  const res = await fetch("/api/sessions/delete", {
    method: "POST",
    headers: buildAuthHeaders(authToken),
    body: JSON.stringify({ sessionId }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    throw new Error(data?.error ?? "Failed to delete session")
  }
}

export async function listSessions(
  userId: string,
  maxResults = 20
): Promise<SessionSummary[]> {
  const q = query(
    collection(db, SESSIONS_COLLECTION),
    where("userId", "==", userId),
    orderBy("createdAt", "desc"),
    limit(maxResults)
  )
  const snap = await getDocs(q)
  return snap.docs.map((d) => {
    const data = d.data() as SessionDocument

    let overallScore: number | null = null
    if (data.scores) {
      if ("overall" in data.scores) {
        overallScore = data.scores.overall
      } else if (
        "rubric" in data.scores &&
        Array.isArray(data.scores.rubric) &&
        data.scores.rubric.length > 0
      ) {
        const sum = data.scores.rubric.reduce((acc, c) => acc + c.score, 0)
        overallScore = Math.round(sum / data.scores.rubric.length)
      }
    }

    const v2 = data.scores && isV2Scores(data.scores) ? data.scores : null

    return {
      id: d.id,
      topic: v2?.refinedTitle ?? data.setup?.topic ?? "",
      audience: v2?.refinedAudience ?? data.setup?.audience ?? "",
      goal: v2?.refinedGoal ?? data.setup?.goal ?? "",
      date: data.createdAt?.toDate() ?? new Date(),
      overallScore,
    }
  })
}

import type {
  TimestampedWord,
  FillerInstance,
  FillerSummary,
  PauseInstance,
  PaceWindow,
  ContentSegment,
  DeliveryAnalytics,
} from '@/lib/delivery-analytics'

// ── Constants ──

const SINGLE_WORD_FILLERS = new Set([
  'um', 'uh', 'like', 'basically', 'so', 'right', 'actually', 'well', 'literally',
])

const MULTI_WORD_FILLERS = [
  ['you', 'know'],
  ['i', 'mean'],
  ['kind', 'of'],
  ['sort', 'of'],
]

const PAUSE_THRESHOLD_SECONDS = 1.5
const MULTI_WORD_MAX_GAP_SECONDS = 0.5
const PACE_WINDOW_SECONDS = 30
const CONTENT_SEGMENT_SECONDS = 120
const PRECEDING_CONTEXT_WORDS = 5

// ── Utilities ──

function cleanWord(w: string): string {
  return w.replace(/[^a-zA-Z']/g, '').toLowerCase()
}

function stddev(values: number[]): number {
  if (values.length < 2) return 0
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const sqDiffs = values.map((v) => (v - mean) ** 2)
  return Math.sqrt(sqDiffs.reduce((a, b) => a + b, 0) / values.length)
}

export function emptyAnalytics(): DeliveryAnalytics {
  return {
    words: [],
    totalDurationSeconds: 0,
    averageWpm: 0,
    paceWindows: [],
    paceVariation: 0,
    fillerInstances: [],
    fillerSummary: [],
    totalFillerCount: 0,
    fillersPerMinute: 0,
    pauses: [],
    totalPauseCount: 0,
    averagePauseDuration: 0,
    longestPause: null,
    contentSegments: [],
  }
}

// ── Filler Detection ──

export function detectFillers(words: TimestampedWord[]): FillerInstance[] {
  const results: FillerInstance[] = []
  const consumed = new Set<number>()

  // Pass 1: multi-word fillers (greedy)
  for (let i = 0; i < words.length; i++) {
    if (consumed.has(i)) continue

    for (const pattern of MULTI_WORD_FILLERS) {
      if (i + pattern.length > words.length) continue

      let match = true
      let temporallyClose = true

      for (let j = 0; j < pattern.length; j++) {
        if (consumed.has(i + j)) { match = false; break }
        if (cleanWord(words[i + j].word) !== pattern[j]) { match = false; break }
        if (j > 0) {
          const gap = words[i + j].start - words[i + j - 1].end
          if (gap > MULTI_WORD_MAX_GAP_SECONDS) { temporallyClose = false; break }
        }
      }

      if (match && temporallyClose) {
        const indices = Array.from({ length: pattern.length }, (_, j) => i + j)
        indices.forEach((idx) => consumed.add(idx))
        results.push({
          phrase: pattern.join(' '),
          timestamp: words[i].start,
          wordIndices: indices,
        })
        break
      }
    }
  }

  // Pass 2: single-word fillers (skip consumed)
  for (let i = 0; i < words.length; i++) {
    if (consumed.has(i)) continue
    const cleaned = cleanWord(words[i].word)
    if (SINGLE_WORD_FILLERS.has(cleaned)) {
      consumed.add(i)
      results.push({
        phrase: cleaned,
        timestamp: words[i].start,
        wordIndices: [i],
      })
    }
  }

  return results.sort((a, b) => a.timestamp - b.timestamp)
}

// ── Pause Detection ──

export function detectPauses(words: TimestampedWord[]): PauseInstance[] {
  const pauses: PauseInstance[] = []
  for (let i = 1; i < words.length; i++) {
    const gap = words[i].start - words[i - 1].end
    if (gap >= PAUSE_THRESHOLD_SECONDS) {
      const contextStart = Math.max(0, i - PRECEDING_CONTEXT_WORDS)
      const contextWords = words.slice(contextStart, i).map((w) => w.word)
      pauses.push({
        start: words[i - 1].end,
        end: words[i].start,
        duration: gap,
        precedingWord: words[i - 1].word,
        followingWord: words[i].word,
        precedingContext: contextWords.join(' '),
      })
    }
  }
  return pauses
}

// ── Pace Windows ──

export function computePaceWindows(words: TimestampedWord[], totalDuration: number): PaceWindow[] {
  if (words.length === 0 || totalDuration <= 0) return []

  const windows: PaceWindow[] = []
  const numWindows = Math.max(1, Math.ceil(totalDuration / PACE_WINDOW_SECONDS))
  const windowDuration = totalDuration / numWindows

  for (let i = 0; i < numWindows; i++) {
    const startTime = i * windowDuration
    const endTime = (i + 1) * windowDuration
    const wordsInWindow = words.filter((w) => {
      const mid = (w.start + w.end) / 2
      return mid >= startTime && mid < endTime
    })
    const durationMinutes = windowDuration / 60
    windows.push({
      startTime,
      endTime,
      wpm: durationMinutes > 0 ? Math.round(wordsInWindow.length / durationMinutes) : 0,
      wordCount: wordsInWindow.length,
    })
  }

  return windows
}

// ── Content Segments ──

export function computeContentSegments(words: TimestampedWord[], totalDuration: number): ContentSegment[] {
  if (words.length === 0 || totalDuration <= 0) return []

  const numSegments = Math.max(1, Math.ceil(totalDuration / CONTENT_SEGMENT_SECONDS))
  const segmentDuration = totalDuration / numSegments
  const segments: ContentSegment[] = []

  for (let i = 0; i < numSegments; i++) {
    const startTime = i * segmentDuration
    const endTime = (i + 1) * segmentDuration
    const segWords = words.filter((w) => {
      const mid = (w.start + w.end) / 2
      return mid >= startTime && mid < endTime
    })
    const text = segWords.map((w) => w.word).join(' ')
    const durationMinutes = segmentDuration / 60
    const wpm = durationMinutes > 0 ? Math.round(segWords.length / durationMinutes) : 0
    const topicLabel = segWords.slice(0, 8).map((w) => w.word).join(' ')
      + (segWords.length > 8 ? '...' : '')
    segments.push({
      startTime,
      endTime,
      text,
      wpm,
      wordCount: segWords.length,
      topicLabel: topicLabel || '(empty)',
    })
  }

  return segments
}

// ── Main Entry Point ──

export function computeDeliveryAnalytics(words: TimestampedWord[]): DeliveryAnalytics {
  if (words.length === 0) return emptyAnalytics()

  const totalDurationSeconds = words[words.length - 1].end - words[0].start
  if (totalDurationSeconds <= 0) return { ...emptyAnalytics(), words }

  const totalMinutes = totalDurationSeconds / 60
  const averageWpm = Math.round(words.length / totalMinutes)

  const paceWindows = computePaceWindows(words, totalDurationSeconds)
  const paceVariation = stddev(paceWindows.map((w) => w.wpm))

  const fillerInstances = detectFillers(words)
  const totalFillerCount = fillerInstances.length
  const fillersPerMinute = totalMinutes > 0
    ? Math.round((totalFillerCount / totalMinutes) * 10) / 10
    : 0

  // Build filler summary
  const fillerCounts = new Map<string, number>()
  for (const f of fillerInstances) {
    fillerCounts.set(f.phrase, (fillerCounts.get(f.phrase) ?? 0) + 1)
  }
  const fillerSummary: FillerSummary[] = Array.from(fillerCounts.entries())
    .map(([phrase, count]) => ({
      phrase,
      count,
      perMinute: totalMinutes > 0 ? Math.round((count / totalMinutes) * 10) / 10 : 0,
    }))
    .sort((a, b) => b.count - a.count)

  const pauses = detectPauses(words)
  const totalPauseCount = pauses.length
  const averagePauseDuration = pauses.length > 0
    ? Math.round((pauses.reduce((s, p) => s + p.duration, 0) / pauses.length) * 10) / 10
    : 0
  const longestPause = pauses.length > 0
    ? pauses.reduce((max, p) => (p.duration > max.duration ? p : max), pauses[0])
    : null

  const contentSegments = computeContentSegments(words, totalDurationSeconds)

  return {
    words,
    totalDurationSeconds,
    averageWpm,
    paceWindows,
    paceVariation,
    fillerInstances,
    fillerSummary,
    totalFillerCount,
    fillersPerMinute,
    pauses,
    totalPauseCount,
    averagePauseDuration,
    longestPause,
    contentSegments,
  }
}

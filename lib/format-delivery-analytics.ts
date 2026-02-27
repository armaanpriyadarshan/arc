import type { DeliveryAnalytics } from '@/lib/delivery-analytics'

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function describePace(wpm: number): string {
  if (wpm < 100) return 'very slow'
  if (wpm < 120) return 'measured, deliberate'
  if (wpm < 150) return 'conversational, good range'
  if (wpm < 170) return 'brisk but clear'
  if (wpm < 190) return 'fast'
  return 'very fast, may lose audience'
}

function describePaceVariation(stddev: number): string {
  if (stddev < 8) return 'very uniform — could benefit from more variation'
  if (stddev < 20) return 'natural variation'
  if (stddev < 35) return 'notable variation — may indicate rushing or hesitation'
  return 'high variation — inconsistent pacing'
}

export function formatAnalyticsSummary(analytics: DeliveryAnalytics): string {
  if (analytics.totalDurationSeconds <= 0 || analytics.words.length === 0) {
    return 'No delivery data available.'
  }

  const lines: string[] = []

  // Pace
  lines.push(`Speaking pace: ${analytics.averageWpm} WPM average (${describePace(analytics.averageWpm)})`)
  lines.push(`Pace variation: ${analytics.paceVariation.toFixed(1)} WPM std dev (${describePaceVariation(analytics.paceVariation)})`)

  if (analytics.paceWindows.length > 1) {
    const fastest = analytics.paceWindows.reduce((max, w) => w.wpm > max.wpm ? w : max, analytics.paceWindows[0])
    const slowest = analytics.paceWindows.reduce((min, w) => w.wpm < min.wpm ? w : min, analytics.paceWindows[0])
    lines.push(`Fastest section: ${fastest.wpm} WPM at ${formatTime(fastest.startTime)}`)
    lines.push(`Slowest section: ${slowest.wpm} WPM at ${formatTime(slowest.startTime)}`)
  }

  lines.push('')

  // Fillers
  lines.push(`Filler words: ${analytics.totalFillerCount} total (${analytics.fillersPerMinute}/min)`)
  if (analytics.fillerSummary.length > 0) {
    const top = analytics.fillerSummary.slice(0, 3)
      .map((f) => `"${f.phrase}" (${f.count}x)`)
      .join(', ')
    lines.push(`Most frequent: ${top}`)
  }

  lines.push('')

  // Pauses
  lines.push(`Significant pauses (>1.5s): ${analytics.totalPauseCount}`)
  if (analytics.longestPause) {
    lines.push(`Longest pause: ${analytics.longestPause.duration.toFixed(1)}s after "${analytics.longestPause.precedingContext}"`)
  }

  // Content segments
  if (analytics.contentSegments.length > 1) {
    lines.push('')
    lines.push('Pace by content section:')
    for (const seg of analytics.contentSegments) {
      lines.push(`  ${formatTime(seg.startTime)}-${formatTime(seg.endTime)}: ${seg.wpm} WPM — "${seg.topicLabel}"`)
    }
  }

  return lines.join('\n')
}

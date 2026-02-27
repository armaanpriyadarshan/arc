import { describe, it, expect } from 'vitest'
import { formatAnalyticsSummary } from '@/lib/format-delivery-analytics'
import type { DeliveryAnalytics } from '@/lib/delivery-analytics'
import { emptyAnalytics } from '@/backend/delivery-analytics'

function makeAnalytics(overrides: Partial<DeliveryAnalytics> = {}): DeliveryAnalytics {
  return {
    ...emptyAnalytics(),
    words: [{ word: 'hello', start: 0, end: 0.5 }],
    totalDurationSeconds: 300,
    averageWpm: 145,
    paceVariation: 18.3,
    paceWindows: [
      { startTime: 0, endTime: 30, wpm: 155, wordCount: 78 },
      { startTime: 30, endTime: 60, wpm: 112, wordCount: 56 },
      { startTime: 60, endTime: 90, wpm: 178, wordCount: 89 },
      { startTime: 90, endTime: 120, wpm: 135, wordCount: 68 },
    ],
    fillerInstances: [
      { phrase: 'um', timestamp: 5, wordIndices: [2] },
      { phrase: 'um', timestamp: 30, wordIndices: [15] },
      { phrase: 'like', timestamp: 60, wordIndices: [30] },
    ],
    fillerSummary: [
      { phrase: 'um', count: 2, perMinute: 0.4 },
      { phrase: 'like', count: 1, perMinute: 0.2 },
    ],
    totalFillerCount: 3,
    fillersPerMinute: 0.6,
    pauses: [
      {
        start: 45,
        end: 47.5,
        duration: 2.5,
        precedingWord: 'believe',
        followingWord: 'and',
        precedingContext: 'and that is why we believe',
      },
    ],
    totalPauseCount: 1,
    averagePauseDuration: 2.5,
    longestPause: {
      start: 45,
      end: 47.5,
      duration: 2.5,
      precedingWord: 'believe',
      followingWord: 'and',
      precedingContext: 'and that is why we believe',
    },
    contentSegments: [
      { startTime: 0, endTime: 120, text: 'Thank you...', wpm: 155, wordCount: 310, topicLabel: 'Thank you for being here today...' },
      { startTime: 120, endTime: 240, text: 'Our market...', wpm: 178, wordCount: 356, topicLabel: 'Our market opportunity is enormous...' },
      { startTime: 240, endTime: 300, text: 'Let me...', wpm: 125, wordCount: 125, topicLabel: 'Let me walk you through the...' },
    ],
    ...overrides,
  }
}

describe('formatAnalyticsSummary', () => {
  it('produces expected summary for known input', () => {
    const summary = formatAnalyticsSummary(makeAnalytics())

    expect(summary).toContain('145 WPM average')
    expect(summary).toContain('conversational, good range')
    expect(summary).toContain('18.3 WPM std dev')
    expect(summary).toContain('Fastest section: 178 WPM')
    expect(summary).toContain('Slowest section: 112 WPM')
    expect(summary).toContain('Filler words: 3 total')
    expect(summary).toContain('"um" (2x)')
    expect(summary).toContain('"like" (1x)')
    expect(summary).toContain('Significant pauses (>1.5s): 1')
    expect(summary).toContain('2.5s after')
    expect(summary).toContain('Pace by content section:')
    expect(summary).toContain('155 WPM')
    expect(summary).toContain('178 WPM')
    expect(summary).toContain('125 WPM')
  })

  it('handles zero-duration analytics', () => {
    const result = formatAnalyticsSummary(emptyAnalytics())
    expect(result).toBe('No delivery data available.')
  })

  it('handles empty words with positive duration', () => {
    const result = formatAnalyticsSummary({
      ...emptyAnalytics(),
      totalDurationSeconds: 60,
    })
    expect(result).toBe('No delivery data available.')
  })

  it('skips fastest/slowest when only one pace window', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      paceWindows: [{ startTime: 0, endTime: 30, wpm: 140, wordCount: 70 }],
    }))
    expect(summary).not.toContain('Fastest section')
    expect(summary).not.toContain('Slowest section')
  })

  it('skips content sections line when only one segment', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      contentSegments: [{ startTime: 0, endTime: 60, text: 'hello', wpm: 140, wordCount: 140, topicLabel: 'hello...' }],
    }))
    expect(summary).not.toContain('Pace by content section')
  })

  it('handles no fillers', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      fillerInstances: [],
      fillerSummary: [],
      totalFillerCount: 0,
      fillersPerMinute: 0,
    }))
    expect(summary).toContain('Filler words: 0 total')
    expect(summary).not.toContain('Most frequent')
  })

  it('handles no pauses', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      pauses: [],
      totalPauseCount: 0,
      longestPause: null,
    }))
    expect(summary).toContain('Significant pauses (>1.5s): 0')
    expect(summary).not.toContain('Longest pause')
  })
})

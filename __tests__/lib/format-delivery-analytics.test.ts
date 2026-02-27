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

  it('includes energy/volume section when energy windows present', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      energyWindows: [
        { startTime: 0, endTime: 30, rmsDb: -20 },
        { startTime: 30, endTime: 60, rmsDb: -15 },
      ],
      averageEnergyDb: -17.5,
      energyVariation: 3.5,
      peakEnergyDb: -15,
    }))
    expect(summary).toContain('Volume:')
    expect(summary).toContain('-17.5 dB RMS average')
    expect(summary).toContain('Volume variation:')
    expect(summary).toContain('3.5 dB std dev')
    expect(summary).toContain('Peak volume: -15 dB')
    expect(summary).toContain('Loudest section:')
    expect(summary).toContain('Quietest section:')
  })

  it('omits energy section when no energy windows', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      energyWindows: [],
      averageEnergyDb: 0,
      energyVariation: 0,
      peakEnergyDb: 0,
    }))
    expect(summary).not.toContain('Volume:')
  })

  it('includes delivery observations for high filler density', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      fillersPerMinute: 5,
      totalFillerCount: 25,
    }))
    expect(summary).toContain('Delivery observations:')
    expect(summary).toContain('filler word density')
  })

  it('includes delivery observation for flat energy', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      energyWindows: [
        { startTime: 0, endTime: 30, rmsDb: -20 },
        { startTime: 30, endTime: 60, rmsDb: -20.5 },
      ],
      energyVariation: 1.0,
    }))
    expect(summary).toContain('Volume is very flat')
  })

  it('includes delivery observation for no fillers', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      fillerInstances: [],
      fillerSummary: [],
      totalFillerCount: 0,
      fillersPerMinute: 0,
    }))
    expect(summary).toContain('No filler words detected')
  })

  it('includes delivery observation for uniform pace', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      paceVariation: 5,
    }))
    expect(summary).toContain('very uniform')
  })

  it('includes pitch section when pitch windows are present', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      pitchWindows: [
        { startTime: 0, endTime: 30, medianF0Hz: 180, medianF0Semitones: 89.2, f0RangeSemitones: 3.5, f0StddevSemitones: 1.2, voicedFrameRatio: 0.85 },
        { startTime: 30, endTime: 60, medianF0Hz: 220, medianF0Semitones: 92.0, f0RangeSemitones: 4.0, f0StddevSemitones: 1.5, voicedFrameRatio: 0.9 },
      ],
      averagePitchHz: 200,
      averagePitchSemitones: 90.6,
      pitchRangeSemitones: 2.8,
      pitchVariationSemitones: 1.4,
      overallVoicedRatio: 0.88,
    }))
    expect(summary).toContain('Pitch: 200 Hz average')
    expect(summary).toContain('Pitch range:')
    expect(summary).toContain('Pitch variation:')
    expect(summary).toContain('Highest pitch section: 220 Hz')
    expect(summary).toContain('Lowest pitch section: 180 Hz')
  })

  it('omits pitch section when no pitch data', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      pitchWindows: [],
      averagePitchHz: 0,
      averagePitchSemitones: 0,
      pitchRangeSemitones: 0,
      pitchVariationSemitones: 0,
      overallVoicedRatio: 0,
    }))
    expect(summary).not.toContain('Pitch:')
    expect(summary).not.toContain('Pitch range:')
  })

  it('includes monotone observation for flat pitch', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      pitchWindows: [
        { startTime: 0, endTime: 30, medianF0Hz: 180, medianF0Semitones: 89.2, f0RangeSemitones: 1.0, f0StddevSemitones: 0.5, voicedFrameRatio: 0.85 },
        { startTime: 30, endTime: 60, medianF0Hz: 181, medianF0Semitones: 89.3, f0RangeSemitones: 0.8, f0StddevSemitones: 0.4, voicedFrameRatio: 0.9 },
      ],
      averagePitchHz: 180.5,
      averagePitchSemitones: 89.25,
      pitchRangeSemitones: 0.1,
      pitchVariationSemitones: 0.05,
      overallVoicedRatio: 0.88,
    }))
    expect(summary).toContain('monotone')
  })

  it('includes upspeak observation for rising pitch', () => {
    const summary = formatAnalyticsSummary(makeAnalytics({
      pitchWindows: [
        { startTime: 0, endTime: 30, medianF0Hz: 150, medianF0Semitones: 86.0, f0RangeSemitones: 3.0, f0StddevSemitones: 1.0, voicedFrameRatio: 0.85 },
        { startTime: 30, endTime: 60, medianF0Hz: 160, medianF0Semitones: 87.0, f0RangeSemitones: 3.0, f0StddevSemitones: 1.0, voicedFrameRatio: 0.85 },
        { startTime: 60, endTime: 90, medianF0Hz: 200, medianF0Semitones: 90.5, f0RangeSemitones: 3.0, f0StddevSemitones: 1.0, voicedFrameRatio: 0.85 },
      ],
      averagePitchHz: 170,
      averagePitchSemitones: 87.8,
      pitchRangeSemitones: 4.5,
      pitchVariationSemitones: 1.9,
      overallVoicedRatio: 0.85,
    }))
    expect(summary).toContain('Pitch rises toward the end')
  })
})

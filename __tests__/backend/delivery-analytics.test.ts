import { describe, it, expect } from 'vitest'
import {
  computeDeliveryAnalytics,
  detectFillers,
  detectPauses,
  computePaceWindows,
  computeContentSegments,
  emptyAnalytics,
} from '@/backend/delivery-analytics'
import type { TimestampedWord } from '@/lib/delivery-analytics'

function word(w: string, start: number, end: number): TimestampedWord {
  return { word: w, start, end }
}

describe('emptyAnalytics', () => {
  it('returns zeroed analytics', () => {
    const a = emptyAnalytics()
    expect(a.words).toEqual([])
    expect(a.totalDurationSeconds).toBe(0)
    expect(a.averageWpm).toBe(0)
    expect(a.longestPause).toBeNull()
    expect(a.energyWindows).toEqual([])
    expect(a.averageEnergyDb).toBe(0)
    expect(a.energyVariation).toBe(0)
    expect(a.peakEnergyDb).toBe(0)
    expect(a.pitchWindows).toEqual([])
    expect(a.averagePitchHz).toBe(0)
    expect(a.averagePitchSemitones).toBe(0)
    expect(a.pitchRangeSemitones).toBe(0)
    expect(a.pitchVariationSemitones).toBe(0)
    expect(a.overallVoicedRatio).toBe(0)
  })
})

describe('detectFillers', () => {
  it('detects single-word fillers: um, uh', () => {
    const words = [
      word('um', 0, 0.3),
      word('I', 0.4, 0.5),
      word('think', 0.6, 0.9),
      word('uh', 1.0, 1.3),
      word('yes', 1.4, 1.7),
    ]
    const fillers = detectFillers(words)
    expect(fillers).toHaveLength(2)
    expect(fillers[0].phrase).toBe('um')
    expect(fillers[1].phrase).toBe('uh')
  })

  it('detects multi-word filler "you know"', () => {
    const words = [
      word('you', 0, 0.2),
      word('know', 0.3, 0.5),
      word('the', 0.6, 0.8),
      word('thing', 0.9, 1.1),
    ]
    const fillers = detectFillers(words)
    expect(fillers).toHaveLength(1)
    expect(fillers[0].phrase).toBe('you know')
    expect(fillers[0].wordIndices).toEqual([0, 1])
  })

  it('does not detect multi-word filler when gap > 0.5s', () => {
    const words = [
      word('you', 0, 0.2),
      word('know', 1.0, 1.2), // gap = 0.8s > 0.5s
      word('the', 1.3, 1.5),
    ]
    const fillers = detectFillers(words)
    // "you" and "know" individually are not single-word fillers, so nothing detected
    expect(fillers.filter((f) => f.phrase === 'you know')).toHaveLength(0)
  })

  it('does not double-count word consumed by multi-word filler', () => {
    // "sort of" is a multi-word filler; "so" in "sort" should not trigger single-word "so"
    const words = [
      word('sort', 0, 0.3),
      word('of', 0.35, 0.5),
      word('interesting', 0.6, 1.0),
    ]
    const fillers = detectFillers(words)
    expect(fillers).toHaveLength(1)
    expect(fillers[0].phrase).toBe('sort of')
  })

  it('detects "like" as single-word filler', () => {
    const words = [
      word('it', 0, 0.2),
      word('was', 0.3, 0.5),
      word('like', 0.6, 0.8),
      word('amazing', 0.9, 1.3),
    ]
    const fillers = detectFillers(words)
    expect(fillers).toHaveLength(1)
    expect(fillers[0].phrase).toBe('like')
  })

  it('returns empty array for no fillers', () => {
    const words = [
      word('The', 0, 0.2),
      word('market', 0.3, 0.6),
      word('is', 0.7, 0.8),
      word('large', 0.9, 1.2),
    ]
    expect(detectFillers(words)).toEqual([])
  })

  it('sorts results by timestamp', () => {
    const words = [
      word('like', 0, 0.3),
      word('I', 0.4, 0.5),
      word('um', 0.6, 0.8),
      word('think', 0.9, 1.1),
      word('you', 1.2, 1.3),
      word('know', 1.35, 1.5),
    ]
    const fillers = detectFillers(words)
    expect(fillers).toHaveLength(3)
    expect(fillers[0].timestamp).toBeLessThanOrEqual(fillers[1].timestamp)
    expect(fillers[1].timestamp).toBeLessThanOrEqual(fillers[2].timestamp)
  })
})

describe('detectPauses', () => {
  it('detects pause > 1.5s between words', () => {
    const words = [
      word('hello', 0, 0.5),
      word('world', 2.5, 3.0), // gap = 2.0s
    ]
    const pauses = detectPauses(words)
    expect(pauses).toHaveLength(1)
    expect(pauses[0].duration).toBeCloseTo(2.0)
    expect(pauses[0].precedingWord).toBe('hello')
    expect(pauses[0].followingWord).toBe('world')
  })

  it('does not detect pause < 1.5s', () => {
    const words = [
      word('hello', 0, 0.5),
      word('world', 1.0, 1.5), // gap = 0.5s
    ]
    expect(detectPauses(words)).toEqual([])
  })

  it('includes preceding context (up to 5 words)', () => {
    const words = [
      word('one', 0, 0.2),
      word('two', 0.3, 0.5),
      word('three', 0.6, 0.8),
      word('four', 0.9, 1.1),
      word('five', 1.2, 1.4),
      word('six', 1.5, 1.7),
      word('after', 3.5, 3.8), // gap = 1.8s
    ]
    const pauses = detectPauses(words)
    expect(pauses).toHaveLength(1)
    expect(pauses[0].precedingContext).toBe('two three four five six')
  })

  it('returns empty for single word', () => {
    expect(detectPauses([word('hello', 0, 0.5)])).toEqual([])
  })
})

describe('computePaceWindows', () => {
  it('computes correct WPM for even spacing', () => {
    // 60 words over 60 seconds = 60 WPM overall
    // With 30s windows: 2 windows, each should get ~30 words = 60 WPM
    const words: TimestampedWord[] = []
    for (let i = 0; i < 60; i++) {
      words.push(word(`word${i}`, i, i + 0.5))
    }
    const windows = computePaceWindows(words, 60)
    expect(windows).toHaveLength(2)
    expect(windows[0].wpm).toBe(60)
    expect(windows[1].wpm).toBe(60)
  })

  it('returns empty for no words', () => {
    expect(computePaceWindows([], 60)).toEqual([])
  })

  it('returns empty for zero duration', () => {
    expect(computePaceWindows([word('hi', 0, 0.3)], 0)).toEqual([])
  })

  it('creates single window for short duration', () => {
    const words = [word('hi', 0, 0.3), word('there', 0.5, 0.8)]
    const windows = computePaceWindows(words, 10)
    expect(windows).toHaveLength(1)
    expect(windows[0].wordCount).toBe(2)
  })
})

describe('computeContentSegments', () => {
  it('splits into ~2-minute segments', () => {
    // 300 seconds = 5 minutes → ceil(300/120) = 3 segments
    const words: TimestampedWord[] = []
    for (let i = 0; i < 300; i++) {
      words.push(word(`w${i}`, i, i + 0.5))
    }
    const segments = computeContentSegments(words, 300)
    expect(segments).toHaveLength(3)
    for (const seg of segments) {
      expect(seg.wpm).toBeGreaterThan(0)
      expect(seg.topicLabel).toBeTruthy()
    }
  })

  it('returns empty for no words', () => {
    expect(computeContentSegments([], 120)).toEqual([])
  })

  it('topic label uses first ~8 words with ellipsis', () => {
    const words: TimestampedWord[] = []
    for (let i = 0; i < 20; i++) {
      words.push(word(`word${i}`, i, i + 0.5))
    }
    const segments = computeContentSegments(words, 20)
    expect(segments[0].topicLabel).toContain('...')
  })
})

describe('computeDeliveryAnalytics', () => {
  it('returns empty analytics for empty words', () => {
    const result = computeDeliveryAnalytics([])
    expect(result).toEqual(emptyAnalytics())
  })

  it('handles very short input (2-3 words)', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const result = computeDeliveryAnalytics(words)
    expect(result.words).toHaveLength(2)
    expect(result.totalDurationSeconds).toBeCloseTo(1.0)
    expect(result.averageWpm).toBeGreaterThan(0)
  })

  it('computes correct averageWpm', () => {
    // 120 words over 60 seconds = 120 WPM
    const words: TimestampedWord[] = []
    for (let i = 0; i < 120; i++) {
      words.push(word(`w${i}`, i * 0.5, i * 0.5 + 0.3))
    }
    const result = computeDeliveryAnalytics(words)
    // totalDuration = 59.8s, so 120 / (59.8/60) ≈ 120
    expect(result.averageWpm).toBeGreaterThanOrEqual(118)
    expect(result.averageWpm).toBeLessThanOrEqual(122)
  })

  it('detects fillers, pauses, and computes segments together', () => {
    const words = [
      word('hello', 0, 0.5),
      word('um', 0.6, 0.9),
      word('I', 1.0, 1.1),
      word('think', 1.2, 1.5),
      word('that', 1.6, 1.8),
      // 2.5s pause
      word('uh', 4.3, 4.6),
      word('the', 4.7, 4.9),
      word('market', 5.0, 5.4),
      word('is', 5.5, 5.7),
      word('big', 5.8, 6.1),
    ]
    const result = computeDeliveryAnalytics(words)
    expect(result.totalFillerCount).toBe(2) // "um" and "uh"
    expect(result.pauses.length).toBe(1) // gap after "that"
    expect(result.pauses[0].duration).toBeCloseTo(2.5)
    expect(result.averageWpm).toBeGreaterThan(0)
  })

  it('handles all-filler input', () => {
    const words = [
      word('um', 0, 0.3),
      word('uh', 0.4, 0.7),
      word('like', 0.8, 1.0),
      word('um', 1.1, 1.4),
    ]
    const result = computeDeliveryAnalytics(words)
    expect(result.totalFillerCount).toBe(4)
    expect(result.fillersPerMinute).toBeGreaterThan(0)
  })

  it('handles single long pause', () => {
    const words = [
      word('hello', 0, 0.5),
      word('world', 10.5, 11.0), // 10s pause
    ]
    const result = computeDeliveryAnalytics(words)
    expect(result.totalPauseCount).toBe(1)
    expect(result.longestPause!.duration).toBeCloseTo(10.0)
  })

  it('returns words with zero duration as empty analytics (preserves words)', () => {
    const words = [word('hi', 5, 5)]
    const result = computeDeliveryAnalytics(words)
    expect(result.averageWpm).toBe(0)
    expect(result.words).toEqual(words)
  })

  it('integrates energy windows when provided', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const energy = [
      { startTime: 0, endTime: 0.5, rmsDb: -20 },
      { startTime: 0.5, endTime: 1.0, rmsDb: -15 },
    ]
    const result = computeDeliveryAnalytics(words, energy)
    expect(result.energyWindows).toHaveLength(2)
    expect(result.averageEnergyDb).toBeCloseTo(-17.5, 0)
    expect(result.peakEnergyDb).toBe(-15)
    expect(result.energyVariation).toBeGreaterThan(0)
  })

  it('handles empty energy windows gracefully', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const result = computeDeliveryAnalytics(words, [])
    expect(result.energyWindows).toEqual([])
    expect(result.averageEnergyDb).toBe(0)
    expect(result.peakEnergyDb).toBe(0)
  })

  it('integrates pitch windows when provided', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const pitch = [
      { startTime: 0, endTime: 0.5, medianF0Hz: 180, medianF0Semitones: 89.2, f0RangeSemitones: 3.5, f0StddevSemitones: 1.2, voicedFrameRatio: 0.85 },
      { startTime: 0.5, endTime: 1.0, medianF0Hz: 220, medianF0Semitones: 92.0, f0RangeSemitones: 4.0, f0StddevSemitones: 1.5, voicedFrameRatio: 0.9 },
    ]
    const result = computeDeliveryAnalytics(words, [], pitch)
    expect(result.pitchWindows).toHaveLength(2)
    expect(result.averagePitchHz).toBe(200)
    expect(result.averagePitchSemitones).toBe(90.6)
    expect(result.pitchRangeSemitones).toBe(2.8)
    expect(result.pitchVariationSemitones).toBeGreaterThan(0)
    expect(result.overallVoicedRatio).toBeGreaterThan(0)
  })

  it('handles empty pitch windows gracefully', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const result = computeDeliveryAnalytics(words, [], [])
    expect(result.pitchWindows).toEqual([])
    expect(result.averagePitchHz).toBe(0)
    expect(result.pitchRangeSemitones).toBe(0)
  })

  it('excludes unvoiced windows from pitch averaging', () => {
    const words = [word('hello', 0, 0.5), word('world', 0.6, 1.0)]
    const pitch = [
      { startTime: 0, endTime: 0.5, medianF0Hz: 200, medianF0Semitones: 91.0, f0RangeSemitones: 3.0, f0StddevSemitones: 1.0, voicedFrameRatio: 0.8 },
      { startTime: 0.5, endTime: 1.0, medianF0Hz: 0, medianF0Semitones: 0, f0RangeSemitones: 0, f0StddevSemitones: 0, voicedFrameRatio: 0 },
    ]
    const result = computeDeliveryAnalytics(words, [], pitch)
    // Only the voiced window should be used for averages
    expect(result.averagePitchHz).toBe(200)
    expect(result.averagePitchSemitones).toBe(91.0)
    // Range is 0 because there's only one voiced window
    expect(result.pitchRangeSemitones).toBe(0)
    // Overall voiced ratio = (0.8 + 0) / 2 = 0.4
    expect(result.overallVoicedRatio).toBe(0.4)
  })
})

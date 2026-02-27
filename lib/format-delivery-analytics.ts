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

function describeEnergyLevel(db: number): string {
  if (db < -35) return 'very quiet'
  if (db < -25) return 'soft-spoken'
  if (db < -18) return 'moderate volume'
  if (db < -12) return 'strong, projected voice'
  return 'very loud'
}

function describeEnergyVariation(stddev: number): string {
  if (stddev < 1.5) return 'very flat — monotone energy, little dynamic range'
  if (stddev < 3) return 'some variation — moderate dynamic range'
  if (stddev < 5) return 'good dynamic range — natural emphasis shifts'
  return 'highly dynamic — large volume swings'
}

function describePitchRange(semitones: number): string {
  if (semitones < 2) return 'very narrow — monotone pitch'
  if (semitones < 4) return 'narrow — limited intonation'
  if (semitones < 8) return 'natural range — conversational intonation'
  if (semitones < 12) return 'expressive — animated delivery'
  return 'very wide — highly dramatic intonation'
}

function describePitchVariation(stddevSt: number): string {
  if (stddevSt < 1) return 'very flat — pitch barely changes'
  if (stddevSt < 2) return 'limited variation — could be more expressive'
  if (stddevSt < 4) return 'natural variation — good expressiveness'
  return 'highly variable — very animated pitch'
}

function describeDeliveryCharacter(analytics: DeliveryAnalytics): string[] {
  const observations: string[] = []

  // Pace character
  if (analytics.paceVariation < 8) {
    observations.push('Speaking pace is very uniform, which may come across as monotone or rehearsed.')
  } else if (analytics.paceVariation > 30) {
    observations.push('Pace swings significantly — some sections feel rushed while others drag.')
  }

  // Filler density
  if (analytics.fillersPerMinute > 4) {
    observations.push(`High filler word density (${analytics.fillersPerMinute}/min) suggests lack of preparation or nervousness.`)
  } else if (analytics.fillersPerMinute > 2) {
    observations.push(`Moderate filler word use (${analytics.fillersPerMinute}/min) — noticeable but not distracting.`)
  } else if (analytics.totalFillerCount === 0) {
    observations.push('No filler words detected — speech is clean and rehearsed.')
  }

  // Pause patterns
  if (analytics.totalPauseCount === 0 && analytics.totalDurationSeconds > 60) {
    observations.push('No significant pauses — speaker may benefit from deliberate pauses for emphasis.')
  } else if (analytics.longestPause && analytics.longestPause.duration > 5) {
    observations.push(`A ${analytics.longestPause.duration.toFixed(1)}s pause suggests the speaker lost their place or was thinking.`)
  }

  // Energy character
  if (analytics.energyWindows.length > 1) {
    const firstHalf = analytics.energyWindows.slice(0, Math.ceil(analytics.energyWindows.length / 2))
    const secondHalf = analytics.energyWindows.slice(Math.ceil(analytics.energyWindows.length / 2))
    const firstAvg = firstHalf.reduce((s, w) => s + w.rmsDb, 0) / firstHalf.length
    const secondAvg = secondHalf.reduce((s, w) => s + w.rmsDb, 0) / secondHalf.length
    const diff = secondAvg - firstAvg

    if (diff > 3) {
      observations.push('Energy builds through the presentation — started quiet and gained confidence/volume.')
    } else if (diff < -3) {
      observations.push('Energy drops off toward the end — speaker may be losing steam or rushing to finish.')
    }

    if (analytics.energyVariation < 1.5) {
      observations.push('Volume is very flat throughout — delivery lacks vocal emphasis and dynamic range.')
    }
  }

  // Pitch character
  const voicedPitchWindows = analytics.pitchWindows.filter((w) => w.voicedFrameRatio > 0)
  if (voicedPitchWindows.length >= 2) {
    if (analytics.pitchRangeSemitones < 2 && analytics.pitchVariationSemitones < 1) {
      observations.push('Pitch is monotone — very little intonation variation throughout the presentation.')
    }

    const thirdLen = Math.max(1, Math.floor(voicedPitchWindows.length / 3))
    const firstThird = voicedPitchWindows.slice(0, thirdLen)
    const lastThird = voicedPitchWindows.slice(-thirdLen)
    const firstMedian = firstThird.reduce((s, w) => s + w.medianF0Semitones, 0) / firstThird.length
    const lastMedian = lastThird.reduce((s, w) => s + w.medianF0Semitones, 0) / lastThird.length

    if (lastMedian - firstMedian > 2) {
      observations.push('Pitch rises toward the end — may indicate upspeak or increasing uncertainty.')
    } else if (firstMedian - lastMedian > 2) {
      observations.push('Pitch drops toward the end — may indicate fading energy or confidence.')
    }

    const halfLen = Math.ceil(voicedPitchWindows.length / 2)
    const firstHalfStddevs = voicedPitchWindows.slice(0, halfLen).map((w) => w.f0StddevSemitones)
    const secondHalfStddevs = voicedPitchWindows.slice(halfLen).map((w) => w.f0StddevSemitones)
    const firstHalfAvgStddev = firstHalfStddevs.reduce((a, b) => a + b, 0) / firstHalfStddevs.length
    const secondHalfAvgStddev = secondHalfStddevs.reduce((a, b) => a + b, 0) / secondHalfStddevs.length

    if (secondHalfStddevs.length > 0 && secondHalfAvgStddev > firstHalfAvgStddev * 1.5 && firstHalfAvgStddev > 0) {
      observations.push('Pitch becomes more expressive in the second half — speaker warms up over time.')
    } else if (secondHalfStddevs.length > 0 && firstHalfAvgStddev > secondHalfAvgStddev * 1.5 && secondHalfAvgStddev > 0) {
      observations.push('Pitch expressiveness decreases in the second half — delivery becomes more flat.')
    }
  }

  return observations
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

  // Energy / Volume
  if (analytics.energyWindows.length > 0) {
    lines.push(`Volume: ${analytics.averageEnergyDb} dB RMS average (${describeEnergyLevel(analytics.averageEnergyDb)})`)
    lines.push(`Volume variation: ${analytics.energyVariation} dB std dev (${describeEnergyVariation(analytics.energyVariation)})`)
    lines.push(`Peak volume: ${analytics.peakEnergyDb} dB`)

    if (analytics.energyWindows.length > 1) {
      const loudest = analytics.energyWindows.reduce((max, w) => w.rmsDb > max.rmsDb ? w : max, analytics.energyWindows[0])
      const quietest = analytics.energyWindows.reduce((min, w) => w.rmsDb < min.rmsDb ? w : min, analytics.energyWindows[0])
      lines.push(`Loudest section: ${loudest.rmsDb} dB at ${formatTime(loudest.startTime)}`)
      lines.push(`Quietest section: ${quietest.rmsDb} dB at ${formatTime(quietest.startTime)}`)
    }

    lines.push('')
  }

  // Pitch / Intonation
  const voicedPitchWindows = analytics.pitchWindows.filter((w) => w.voicedFrameRatio > 0)
  if (voicedPitchWindows.length > 0) {
    lines.push(`Pitch: ${analytics.averagePitchHz} Hz average (${analytics.averagePitchSemitones.toFixed(1)} semitones)`)
    lines.push(`Pitch range: ${analytics.pitchRangeSemitones.toFixed(1)} semitones across sections (${describePitchRange(analytics.pitchRangeSemitones)})`)
    lines.push(`Pitch variation: ${analytics.pitchVariationSemitones.toFixed(1)} semitones std dev (${describePitchVariation(analytics.pitchVariationSemitones)})`)

    if (voicedPitchWindows.length > 1) {
      const highest = voicedPitchWindows.reduce((max, w) => w.medianF0Hz > max.medianF0Hz ? w : max, voicedPitchWindows[0])
      const lowest = voicedPitchWindows.reduce((min, w) => w.medianF0Hz < min.medianF0Hz ? w : min, voicedPitchWindows[0])
      lines.push(`Highest pitch section: ${highest.medianF0Hz} Hz at ${formatTime(highest.startTime)}`)
      lines.push(`Lowest pitch section: ${lowest.medianF0Hz} Hz at ${formatTime(lowest.startTime)}`)
    }

    lines.push('')
  }

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

  // Delivery character / tone observations
  const toneObs = describeDeliveryCharacter(analytics)
  if (toneObs.length > 0) {
    lines.push('')
    lines.push('Delivery observations:')
    for (const obs of toneObs) {
      lines.push(`  - ${obs}`)
    }
  }

  return lines.join('\n')
}

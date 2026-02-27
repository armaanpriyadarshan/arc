export interface TimestampedWord {
  word: string
  start: number
  end: number
}

export interface FillerInstance {
  phrase: string
  timestamp: number
  wordIndices: number[]
}

export interface FillerSummary {
  phrase: string
  count: number
  perMinute: number
}

export interface PauseInstance {
  start: number
  end: number
  duration: number
  precedingWord: string
  followingWord: string
  precedingContext: string
}

export interface PaceWindow {
  startTime: number
  endTime: number
  wpm: number
  wordCount: number
}

export interface ContentSegment {
  startTime: number
  endTime: number
  text: string
  wpm: number
  wordCount: number
  topicLabel: string
}

export interface EnergyWindow {
  startTime: number
  endTime: number
  rmsDb: number
}

export interface PitchWindow {
  startTime: number
  endTime: number
  medianF0Hz: number
  medianF0Semitones: number
  f0RangeSemitones: number
  f0StddevSemitones: number
  voicedFrameRatio: number
}

export interface DeliveryAnalytics {
  words: TimestampedWord[]
  totalDurationSeconds: number

  averageWpm: number
  paceWindows: PaceWindow[]
  paceVariation: number

  fillerInstances: FillerInstance[]
  fillerSummary: FillerSummary[]
  totalFillerCount: number
  fillersPerMinute: number

  pauses: PauseInstance[]
  totalPauseCount: number
  averagePauseDuration: number
  longestPause: PauseInstance | null

  contentSegments: ContentSegment[]

  energyWindows: EnergyWindow[]
  averageEnergyDb: number
  energyVariation: number
  peakEnergyDb: number

  pitchWindows: PitchWindow[]
  averagePitchHz: number
  averagePitchSemitones: number
  pitchRangeSemitones: number
  pitchVariationSemitones: number
  overallVoicedRatio: number
}

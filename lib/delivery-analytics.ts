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
}

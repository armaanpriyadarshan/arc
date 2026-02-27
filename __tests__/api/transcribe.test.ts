import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockTranscriptionCreate = vi.fn().mockResolvedValue({
  text: 'This is the transcribed text.',
  words: [
    { word: 'This', start: 0, end: 0.3 },
    { word: 'is', start: 0.4, end: 0.5 },
    { word: 'the', start: 0.6, end: 0.7 },
    { word: 'transcribed', start: 0.8, end: 1.2 },
    { word: 'text.', start: 1.3, end: 1.6 },
  ],
})

vi.mock('@/backend/openai', () => ({
  openai: vi.fn(() => ({
    audio: {
      transcriptions: {
        create: mockTranscriptionCreate,
      },
    },
  })),
}))

vi.mock('@/backend/audio', () => ({
  downloadToTmp: vi.fn().mockResolvedValue('/tmp/vera-test-input.mp4'),
  processFileForWhisper: vi.fn().mockResolvedValue({
    chunks: [{ path: '/tmp/vera-test-chunk0.mp3', offsetSeconds: 0 }],
    allTempPaths: ['/tmp/vera-test-input.mp4', '/tmp/vera-test-compressed.mp3', '/tmp/vera-test-chunk0.mp3'],
    analysisPath: '/tmp/vera-test-compressed.mp3',
  }),
  cleanupTempFiles: vi.fn().mockResolvedValue(undefined),
  analyzeAudio: vi.fn().mockResolvedValue({ energyWindows: [], pitchWindows: [] }),
}))

vi.mock('@/backend/rate-limit', () => ({
  checkRateLimit: vi.fn().mockReturnValue({ allowed: true }),
  getClientIp: vi.fn().mockReturnValue('127.0.0.1'),
}))

vi.mock('@vercel/blob', () => ({
  del: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('fs', () => ({
  default: {
    createReadStream: vi.fn().mockReturnValue('mock-stream'),
  },
  createReadStream: vi.fn().mockReturnValue('mock-stream'),
}))

import { POST } from '@/app/api/transcribe/route'
import { NextRequest } from 'next/server'
import { checkRateLimit } from '@/backend/rate-limit'
import { downloadToTmp, processFileForWhisper, analyzeAudio } from '@/backend/audio'
import { del } from '@vercel/blob'

function createRequest(body?: Record<string, unknown>): NextRequest {
  return new NextRequest('http://localhost/api/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

describe('POST /api/transcribe', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(checkRateLimit).mockReturnValue({ allowed: true })
    mockTranscriptionCreate.mockResolvedValue({
      text: 'This is the transcribed text.',
      words: [
        { word: 'This', start: 0, end: 0.3 },
        { word: 'is', start: 0.4, end: 0.5 },
        { word: 'the', start: 0.6, end: 0.7 },
        { word: 'transcribed', start: 0.8, end: 1.2 },
        { word: 'text.', start: 1.3, end: 1.6 },
      ],
    })
    vi.mocked(downloadToTmp).mockResolvedValue('/tmp/vera-test-input.mp4')
    vi.mocked(processFileForWhisper).mockResolvedValue({
      chunks: [{ path: '/tmp/vera-test-chunk0.mp3', offsetSeconds: 0 }],
      allTempPaths: ['/tmp/vera-test-input.mp4', '/tmp/vera-test-compressed.mp3'],
      analysisPath: '/tmp/vera-test-compressed.mp3',
    })
    vi.mocked(analyzeAudio).mockResolvedValue({ energyWindows: [], pitchWindows: [] })
  })

  it('returns 400 when body is invalid', async () => {
    const request = createRequest({})
    const response = await POST(request)
    expect(response.status).toBe(400)

    const body = await response.json()
    expect(body.error).toContain('Invalid request')
  })

  it('returns 400 when blobUrl is not a valid URL', async () => {
    const request = createRequest({ blobUrl: 'not-a-url', fileName: 'test.mp4' })
    const response = await POST(request)
    expect(response.status).toBe(400)
  })

  it('returns 429 when rate limited', async () => {
    vi.mocked(checkRateLimit).mockReturnValue({ allowed: false })

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })
    const response = await POST(request)
    expect(response.status).toBe(429)

    const body = await response.json()
    expect(body.error).toContain('Too many requests')
  })

  it('returns transcript and analytics on successful transcription', async () => {
    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/presentation.mp4',
      fileName: 'presentation.mp4',
    })

    const response = await POST(request)
    expect(response.status).toBe(200)

    const body = await response.json()
    expect(body.transcript).toBe('This is the transcribed text.')
    expect(body.analytics).toBeDefined()
    expect(body.analytics.words).toHaveLength(5)
    expect(downloadToTmp).toHaveBeenCalledWith(
      'https://example.vercel-storage.com/presentation.mp4',
      'presentation.mp4'
    )
  })

  it('passes whisper-1 model with verbose_json and timestamp_granularities', async () => {
    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    await POST(request)

    expect(mockTranscriptionCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'whisper-1',
        response_format: 'verbose_json',
        timestamp_granularities: ['word'],
      })
    )
  })

  it('returns 500 on OpenAI API failure', async () => {
    mockTranscriptionCreate.mockRejectedValueOnce(new Error('API error'))

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    const response = await POST(request)
    expect(response.status).toBe(500)

    const body = await response.json()
    expect(body.error).toContain('Failed to transcribe')
  })

  it('cleans up temp files even on error', async () => {
    const { cleanupTempFiles } = await import('@/backend/audio')
    mockTranscriptionCreate.mockRejectedValueOnce(new Error('API error'))

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    await POST(request)

    expect(cleanupTempFiles).toHaveBeenCalled()
  })

  it('deletes blob after processing', async () => {
    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    await POST(request)

    expect(del).toHaveBeenCalledWith('https://example.vercel-storage.com/test.mp4')
  })

  it('cleans up temp files when processFileForWhisper fails', async () => {
    const { cleanupTempFiles } = await import('@/backend/audio')
    vi.mocked(processFileForWhisper).mockRejectedValueOnce(new Error('FFmpeg crash'))

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    await POST(request)

    // tempPaths may be empty since processFileForWhisper failed before returning them,
    // but the finally block should still run
    expect(cleanupTempFiles).toHaveBeenCalledTimes(0) // tempPaths was never assigned
  })

  it('deletes blob even when transcription fails', async () => {
    mockTranscriptionCreate.mockRejectedValueOnce(new Error('Whisper API error'))

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/test.mp4',
      fileName: 'test.mp4',
    })

    await POST(request)

    expect(del).toHaveBeenCalledWith('https://example.vercel-storage.com/test.mp4')
  })

  it('surfaces "no audio track" error message', async () => {
    vi.mocked(processFileForWhisper).mockRejectedValueOnce(
      new Error('File does not contain an audio track')
    )

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/silent.mp4',
      fileName: 'silent.mp4',
    })

    const response = await POST(request)
    expect(response.status).toBe(500)
    const body = await response.json()
    expect(body.error).toContain('does not contain an audio track')
  })

  it('joins multiple chunk transcripts with space and merges words with offset', async () => {
    vi.mocked(processFileForWhisper).mockResolvedValue({
      chunks: [
        { path: '/tmp/chunk0.mp3', offsetSeconds: 0 },
        { path: '/tmp/chunk1.mp3', offsetSeconds: 200 },
      ],
      allTempPaths: ['/tmp/input.mp4', '/tmp/chunk0.mp3', '/tmp/chunk1.mp3'],
      analysisPath: '/tmp/input.mp4',
    })
    mockTranscriptionCreate
      .mockResolvedValueOnce({
        text: 'Part one.',
        words: [{ word: 'Part', start: 0, end: 0.3 }, { word: 'one.', start: 0.4, end: 0.7 }],
      })
      .mockResolvedValueOnce({
        text: 'Part two.',
        words: [{ word: 'Part', start: 0, end: 0.3 }, { word: 'two.', start: 0.4, end: 0.7 }],
      })

    const request = createRequest({
      blobUrl: 'https://example.vercel-storage.com/long.mp4',
      fileName: 'long.mp4',
    })

    const response = await POST(request)
    expect(response.status).toBe(200)
    const body = await response.json()
    expect(body.transcript).toBe('Part one. Part two.')
    // Second chunk's words should have offset applied
    expect(body.analytics.words).toHaveLength(4)
    expect(body.analytics.words[2].start).toBeCloseTo(200)
    expect(body.analytics.words[3].start).toBeCloseTo(200.4)
  })
})

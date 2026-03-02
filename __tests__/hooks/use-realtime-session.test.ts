import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useRealtimeSession } from '@/hooks/use-realtime-session'

// ── Track mock instances ──

let lastPcInstance: MockRTCPeerConnection | null = null
let lastDcInstance: {
  onopen: (() => void) | null
  onmessage: ((ev: { data: string }) => void) | null
  readyState: string
  send: ReturnType<typeof vi.fn>
  close: ReturnType<typeof vi.fn>
} | null = null

class MockRTCPeerConnection {
  ontrack: ((ev: { streams: MediaStream[] }) => void) | null = null
  onconnectionstatechange: (() => void) | null = null
  connectionState = 'new'
  iceGatheringState = 'complete'
  localDescription = { sdp: 'mock-sdp-offer', type: 'offer' }

  addEventListener = vi.fn()
  removeEventListener = vi.fn()
  addTrack = vi.fn()
  createDataChannel = vi.fn().mockImplementation(() => {
    const dc = {
      onopen: null as (() => void) | null,
      onmessage: null as ((ev: { data: string }) => void) | null,
      readyState: 'open',
      send: vi.fn(),
      close: vi.fn(),
    }
    lastDcInstance = dc
    return dc
  })
  createOffer = vi.fn().mockResolvedValue({ sdp: 'mock-sdp-offer', type: 'offer' })
  setLocalDescription = vi.fn().mockResolvedValue(undefined)
  setRemoteDescription = vi.fn().mockResolvedValue(undefined)
  close = vi.fn()

  constructor() {
    lastPcInstance = this
  }
}

// ── Mock getUserMedia ──

function createMockStream(): MediaStream {
  return {
    getTracks: () => [{ stop: vi.fn(), kind: 'audio' }],
  } as unknown as MediaStream
}

// ── Mock AudioContext ──

class MockAudioContext {
  createMediaStreamSource = vi.fn().mockReturnValue({ connect: vi.fn() })
  createAnalyser = vi.fn().mockReturnValue({
    fftSize: 0,
    smoothingTimeConstant: 0,
    frequencyBinCount: 4,
    getByteFrequencyData: vi.fn().mockImplementation((arr: Uint8Array) => {
      // Return silence by default
      arr.fill(0)
    }),
  })
  close = vi.fn().mockResolvedValue(undefined)
}

describe('useRealtimeSession', () => {
  let mockFetch: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    lastPcInstance = null
    lastDcInstance = null
    vi.stubGlobal('RTCPeerConnection', MockRTCPeerConnection)
    vi.stubGlobal('AudioContext', MockAudioContext)
    vi.stubGlobal('navigator', {
      ...navigator,
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue(createMockStream()),
      },
    })

    mockFetch = vi.fn()
    vi.stubGlobal('fetch', mockFetch)

    // Default fetch responses: token endpoint, then SDP endpoint (GA: /v1/realtime/calls)
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ token: 'ephemeral-token', expiresAt: 9999999999 }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve('mock-sdp-answer'),
      })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('starts in idle state', () => {
    const { result } = renderHook(() => useRealtimeSession())
    expect(result.current.connectionState).toBe('idle')
    expect(result.current.voiceState).toBe('idle')
    expect(result.current.transcriptEntries).toEqual([])
    expect(result.current.currentCaption).toBe('')
    expect(result.current.elapsed).toBe(0)
    expect(result.current.error).toBeNull()
    expect(result.current.audioDoneCount).toBe(0)
  })

  it('transitions from idle to connecting to connected', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    expect(result.current.connectionState).toBe('idle')

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    expect(result.current.connectionState).toBe('connected')
    expect(result.current.error).toBeNull()
  })

  it('uses correct SDP endpoint (GA: /v1/realtime/calls)', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    // Second fetch call is the SDP exchange
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/realtime/calls',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer ephemeral-token',
          'Content-Type': 'application/sdp',
        }),
        body: 'mock-sdp-offer',
      }),
    )
  })

  it('transitions to error state on token fetch failure', async () => {
    mockFetch.mockReset().mockResolvedValueOnce({
      ok: false,
      json: () => Promise.resolve({ error: 'Rate limited' }),
    })

    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    expect(result.current.connectionState).toBe('error')
    expect(result.current.error).toBe('Rate limited')
  })

  it('transitions to disconnected on disconnect', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    expect(result.current.connectionState).toBe('connected')

    act(() => {
      result.current.disconnect()
    })

    expect(result.current.connectionState).toBe('disconnected')
    expect(result.current.voiceState).toBe('idle')
  })

  it('transitions voice state through full VAD-driven conversation cycle', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    const dc = lastDcInstance!

    // Data channel opens → listening (ready)
    act(() => { dc.onopen?.() })
    expect(result.current.voiceState).toBe('listening')

    // VAD detects speech started → listening
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'input_audio_buffer.speech_started' }) })
    })
    expect(result.current.voiceState).toBe('listening')

    // VAD detects speech stopped → processing
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'input_audio_buffer.speech_stopped' }) })
    })
    expect(result.current.voiceState).toBe('processing')

    // Vera starts speaking
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'response.audio.delta' }) })
    })
    expect(result.current.voiceState).toBe('vera-speaking')

    // Response done → stays vera-speaking, caption stays until user speaks again
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'response.done' }) })
    })
    // Voice state stays vera-speaking until user starts speaking
    expect(result.current.voiceState).toBe('vera-speaking')
    expect(result.current.audioDoneCount).toBe(1)

    // User starts speaking → clears to listening
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'input_audio_buffer.speech_started' }) })
    })
    expect(result.current.voiceState).toBe('listening')
  })

  it('formats fullTranscript from entries', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    const dc = lastDcInstance!

    // Simulate user speech transcription
    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({
          type: 'conversation.item.input_audio_transcription.completed',
          transcript: 'Hello, my pitch is about AI.',
        }),
      })
    })

    // Simulate assistant response
    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({
          type: 'response.audio_transcript.done',
          transcript: 'Interesting, tell me more.',
        }),
      })
    })

    expect(result.current.transcriptEntries).toHaveLength(2)
    expect(result.current.fullTranscript).toContain('[Presenter]: Hello, my pitch is about AI.')
    expect(result.current.fullTranscript).toContain('[Vera]: Interesting, tell me more.')
  })

  it('uses custom presenter name in fullTranscript', async () => {
    const { result } = renderHook(() => useRealtimeSession('Alice'))

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    const dc = lastDcInstance!

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({
          type: 'conversation.item.input_audio_transcription.completed',
          transcript: 'Here is my pitch.',
        }),
      })
    })

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({
          type: 'response.audio_transcript.done',
          transcript: 'Great start!',
        }),
      })
    })

    expect(result.current.fullTranscript).toContain('[Alice]: Here is my pitch.')
    expect(result.current.fullTranscript).toContain('[Vera]: Great start!')
  })

  it('accumulates captions and keeps them visible until audio drains', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    const dc = lastDcInstance!

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.audio_transcript.delta', delta: 'Hello ' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hello ')

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.audio_transcript.delta', delta: 'world' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hello world')

    // Done event keeps caption visible (audio still playing)
    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.audio_transcript.done', transcript: 'Hello world' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hello world')

    // response.done triggers drain poll
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'response.done' }) })
    })

    // Caption stays visible until user speaks again
    expect(result.current.currentCaption).toBe('Hello world')

    // User starts speaking → caption clears
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'input_audio_buffer.speech_started' }) })
    })
    expect(result.current.currentCaption).toBe('')
  })

  it('accumulates captions from GA output_audio_transcript events', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    const dc = lastDcInstance!

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.output_audio_transcript.delta', delta: 'Hi ' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hi ')

    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.output_audio_transcript.delta', delta: 'there' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hi there')

    // GA done event adds transcript entry but keeps caption
    act(() => {
      dc.onmessage?.({
        data: JSON.stringify({ type: 'response.output_audio_transcript.done', transcript: 'Hi there' }),
      })
    })

    expect(result.current.currentCaption).toBe('Hi there')
    expect(result.current.transcriptEntries).toHaveLength(1)
    expect(result.current.transcriptEntries[0].role).toBe('assistant')
    expect(result.current.transcriptEntries[0].text).toBe('Hi there')

    // Caption stays visible after response.done (audio still playing)
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'response.done' }) })
    })
    expect(result.current.currentCaption).toBe('Hi there')

    // User starts speaking → caption clears
    act(() => {
      dc.onmessage?.({ data: JSON.stringify({ type: 'input_audio_buffer.speech_started' }) })
    })
    expect(result.current.currentCaption).toBe('')
  })

  it('cleans up on disconnect', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    act(() => {
      result.current.disconnect()
    })

    expect(result.current.connectionState).toBe('disconnected')
    expect(result.current.currentCaption).toBe('')
    expect(result.current.outputAnalyserNode).toBeNull()
  })

  it('passes setup context and research context to token endpoint', async () => {
    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(
        { topic: 'AI pitch', audience: 'VCs', goal: 'funding' },
        'VC research data',
        'auth-token',
      )
    })

    expect(mockFetch).toHaveBeenCalledWith(
      '/api/realtime/token',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          setupContext: { topic: 'AI pitch', audience: 'VCs', goal: 'funding' },
          researchContext: 'VC research data',
        }),
      }),
    )
  })

  it('handles mic permission denied error', async () => {
    const notAllowedError = new Error('Permission denied')
    notAllowedError.name = 'NotAllowedError'
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValueOnce(notAllowedError)

    const { result } = renderHook(() => useRealtimeSession())

    await act(async () => {
      await result.current.connect(null, null, 'auth-token')
    })

    expect(result.current.connectionState).toBe('error')
    expect(result.current.error).toContain('microphone access')
  })
})

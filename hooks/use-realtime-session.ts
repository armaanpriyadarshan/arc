"use client"

import { useState, useRef, useCallback, useEffect } from 'react'
import { buildAuthHeaders } from '@/lib/api-utils'
import type { SetupContext } from '@/lib/coaching-stages'

export type RealtimeConnectionState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'error'

export type RealtimeVoiceState =
  | 'idle'
  | 'listening'       // user's turn — Vera is listening, audio streams into buffer
  | 'processing'      // buffer committed, waiting for model response
  | 'vera-speaking'   // model is generating audio response

export interface TranscriptEntry {
  role: 'user' | 'assistant'
  text: string
  timestamp: number
}

export interface UseRealtimeSessionReturn {
  connectionState: RealtimeConnectionState
  voiceState: RealtimeVoiceState
  transcriptEntries: TranscriptEntry[]
  currentCaption: string
  elapsed: number
  outputAnalyserNode: AnalyserNode | null
  error: string | null
  /** Increments each time Vera's audio actually finishes playing (not just generating) */
  audioDoneCount: number
  connect: (
    setupContext: SetupContext | null,
    researchContext: string | null,
    authToken: string | null,
  ) => Promise<void>
  disconnect: () => void
  fullTranscript: string
}

const SESSION_WARN_MS = 12 * 60 * 1000 // 12 minutes
const SESSION_MAX_MS = 14 * 60 * 1000 // 14 minutes

export function useRealtimeSession(presenterName?: string): UseRealtimeSessionReturn {
  const [connectionState, setConnectionState] = useState<RealtimeConnectionState>('idle')
  const [voiceState, setVoiceState] = useState<RealtimeVoiceState>('idle')
  const [transcriptEntries, setTranscriptEntries] = useState<TranscriptEntry[]>([])
  const [currentCaption, setCurrentCaption] = useState('')
  const [audioDoneCount, setAudioDoneCount] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [outputAnalyserNode, setOutputAnalyserNode] = useState<AnalyserNode | null>(null)
  const [error, setError] = useState<string | null>(null)

  const pcRef = useRef<RTCPeerConnection | null>(null)
  const dcRef = useRef<RTCDataChannel | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElRef = useRef<HTMLAudioElement | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const warnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const startTimeRef = useRef<number>(0)
  const disconnectRef = useRef<() => void>(() => {})

  // Track whether Vera is speaking via audio response deltas
  const veraSpeakingRef = useRef(false)

  const cleanup = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (warnTimerRef.current) { clearTimeout(warnTimerRef.current); warnTimerRef.current = null }
    if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null }

    if (dcRef.current) {
      try { dcRef.current.close() } catch {}
      dcRef.current = null
    }

    if (pcRef.current) {
      try { pcRef.current.close() } catch {}
      pcRef.current = null
    }

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(t => t.stop())
      localStreamRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {})
      audioContextRef.current = null
    }

    if (audioElRef.current) {
      audioElRef.current.srcObject = null
      audioElRef.current = null
    }

    setOutputAnalyserNode(null)
    veraSpeakingRef.current = false
  }, [])

  const disconnect = useCallback(() => {
    cleanup()
    setConnectionState('disconnected')
    setVoiceState('idle')
    setCurrentCaption('')
  }, [cleanup])

  useEffect(() => { disconnectRef.current = disconnect }, [disconnect])

  // Cleanup on unmount
  useEffect(() => () => cleanup(), [cleanup])

  // Data channel event handler — uses stable React setters so safe in closure
  //
  // Caption strategy: data channel events arrive BEFORE audio finishes playing
  // (WebRTC jitter buffer). Instead of trying to detect audio silence (unreliable),
  // we keep the caption visible until the user starts speaking again. The VAD
  // `speech_started` event naturally fires at the right time.
  function handleDataChannelEvent(event: { type: string; [key: string]: unknown }) {
    switch (event.type) {
      // VAD detected user started speaking — clear previous caption
      case 'input_audio_buffer.speech_started':
        setCurrentCaption('')
        setVoiceState('listening')
        break

      // VAD detected user stopped speaking
      case 'input_audio_buffer.speech_stopped':
      case 'input_audio_buffer.committed':
        setVoiceState('processing')
        break

      // Model started generating a response
      case 'response.audio.delta':
        if (!veraSpeakingRef.current) {
          veraSpeakingRef.current = true
          setVoiceState('vera-speaking')
        }
        break

      // Live caption of Vera's speech
      case 'response.output_audio_transcript.delta':
      case 'response.audio_transcript.delta': {
        const delta = (event as { delta?: string }).delta ?? ''
        if (delta) setCurrentCaption(prev => prev + delta)
        break
      }

      // Vera finished generating one response — add to transcript, keep caption visible
      case 'response.output_audio_transcript.done':
      case 'response.audio_transcript.done': {
        const transcript = (event as { transcript?: string }).transcript ?? ''
        if (transcript) {
          setTranscriptEntries(prev => [
            ...prev,
            { role: 'assistant', text: transcript, timestamp: Date.now() },
          ])
        }
        break
      }

      // User's speech transcribed
      case 'conversation.item.input_audio_transcription.completed': {
        const userText = (event as { transcript?: string }).transcript ?? ''
        if (userText) {
          setTranscriptEntries(prev => [
            ...prev,
            { role: 'user', text: userText, timestamp: Date.now() },
          ])
        }
        break
      }

      // Entire response finished generating — keep vera-speaking & caption until user speaks
      case 'response.done':
        veraSpeakingRef.current = false
        setAudioDoneCount(c => c + 1)
        break

      case 'error': {
        const errMsg = (event as { error?: { message?: string } }).error?.message ?? 'Realtime error'
        console.error('[realtime] Error event:', errMsg)
        setError(errMsg)
        break
      }

      default:
        if (process.env.NODE_ENV !== 'production') {
          console.debug('[realtime] Event:', event.type)
        }
    }
  }

  const connect = useCallback(async (
    setupContext: SetupContext | null,
    researchContext: string | null,
    authToken: string | null,
  ) => {
    // Reset state
    setConnectionState('connecting')
    setVoiceState('idle')
    setTranscriptEntries([])
    setCurrentCaption('')
    setElapsed(0)
    setError(null)
    cleanup()

    try {
      // 1. Fetch ephemeral token from our backend
      const tokenRes = await fetch('/api/realtime/token', {
        method: 'POST',
        headers: buildAuthHeaders(authToken),
        body: JSON.stringify({
          setupContext: setupContext ?? undefined,
          researchContext: researchContext ?? undefined,
        }),
      })

      if (!tokenRes.ok) {
        const body = await tokenRes.json().catch(() => ({ error: 'Failed to get session token' }))
        throw new Error(body.error || `Token request failed: ${tokenRes.status}`)
      }

      const { token } = await tokenRes.json()

      // 2. Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      localStreamRef.current = stream

      // 3. Create RTCPeerConnection
      const pc = new RTCPeerConnection()
      pcRef.current = pc

      // Set up remote audio playback BEFORE adding tracks
      const audioEl = document.createElement('audio')
      audioEl.autoplay = true
      audioElRef.current = audioEl

      pc.ontrack = (ev) => {
        const remoteStream = ev.streams[0] ?? new MediaStream([ev.track])
        audioEl.srcObject = remoteStream

        // AnalyserNode for face mouth animation during Vera's speech
        const ctx = new AudioContext()
        audioContextRef.current = ctx
        const source = ctx.createMediaStreamSource(remoteStream)
        const analyser = ctx.createAnalyser()
        analyser.fftSize = 64
        analyser.smoothingTimeConstant = 0.75
        source.connect(analyser)
        setOutputAnalyserNode(analyser)
      }

      // Add local mic track
      pc.addTrack(stream.getTracks()[0])

      // 4. Create data channel
      const dc = pc.createDataChannel('oai-events')
      dcRef.current = dc

      dc.onopen = () => {
        // Data channel ready — user can start speaking
        setVoiceState('listening')
      }

      dc.onmessage = (ev) => {
        try {
          handleDataChannelEvent(JSON.parse(ev.data))
        } catch {
          // Ignore unparseable messages
        }
      }

      // Connection state monitoring
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
          setError('Connection lost. Try again or switch to segment mode.')
          setConnectionState('error')
          cleanup()
        }
      }

      // 5. SDP offer — wait for ICE gathering to complete
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === 'complete') return resolve()
        const check = () => {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', check)
            resolve()
          }
        }
        pc.addEventListener('icegatheringstatechange', check)
        // Safety timeout — proceed after 3s even if gathering hasn't completed
        setTimeout(() => {
          pc.removeEventListener('icegatheringstatechange', check)
          resolve()
        }, 3000)
      })

      // 6. Exchange SDP with OpenAI — POST /v1/realtime/calls
      const sdpRes = await fetch('https://api.openai.com/v1/realtime/calls', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/sdp',
        },
        body: pc.localDescription!.sdp,
      })

      if (!sdpRes.ok) {
        const errText = await sdpRes.text().catch(() => '')
        throw new Error(`WebRTC handshake failed: ${sdpRes.status} ${errText}`.trim())
      }

      const answerSdp = await sdpRes.text()
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })

      // 7. Start elapsed timer
      startTimeRef.current = Date.now()
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)

      // Session duration limits
      warnTimerRef.current = setTimeout(() => {
        setError('Session approaching time limit. Consider wrapping up.')
      }, SESSION_WARN_MS)

      maxTimerRef.current = setTimeout(() => {
        disconnectRef.current()
      }, SESSION_MAX_MS)

      setConnectionState('connected')
    } catch (err) {
      cleanup()
      const message = err instanceof Error ? err.message : 'Failed to connect'

      if (err instanceof Error && err.name === 'NotAllowedError') {
        setError('Please allow microphone access to use Practice')
      } else if (err instanceof Error && err.name === 'NotFoundError') {
        setError('No microphone found')
      } else {
        setError(message)
      }

      setConnectionState('error')
    }
  }, [cleanup])

  // Compute formatted transcript
  const label = presenterName || 'Presenter'
  const fullTranscript = transcriptEntries
    .map(e => `[${e.role === 'user' ? label : 'Vera'}]: ${e.text}`)
    .join('\n\n')

  return {
    connectionState,
    voiceState,
    transcriptEntries,
    currentCaption,
    elapsed,
    outputAnalyserNode,
    error,
    audioDoneCount,
    connect,
    disconnect,
    fullTranscript,
  }
}

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'
import { handleRealtimeToken } from '@/backend/handlers/realtime-token'

// Mock auth
vi.mock('@/backend/auth', () => ({
  requireAuth: vi.fn(),
}))

// Mock rate limit
vi.mock('@/backend/rate-limit', () => ({
  checkRateLimit: vi.fn(),
}))

// Mock system prompt
vi.mock('@/backend/system-prompt', () => ({
  buildRealtimeInstructions: vi.fn().mockReturnValue('mock-instructions'),
}))

// Mock personas (needed by system-prompt)
vi.mock('@/backend/personas', () => ({
  detectPersona: vi.fn().mockReturnValue(null),
  buildPersonaSection: vi.fn().mockReturnValue(''),
}))

import { requireAuth } from '@/backend/auth'
import { checkRateLimit } from '@/backend/rate-limit'
import { buildRealtimeInstructions } from '@/backend/system-prompt'

function makeRequest(body: object = {}): NextRequest {
  return new NextRequest('http://localhost/api/realtime/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-token' },
    body: JSON.stringify(body),
  })
}

describe('handleRealtimeToken', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(requireAuth).mockResolvedValue({ uid: 'user-1', email: 'test@example.com' })
    vi.mocked(checkRateLimit).mockReturnValue({ allowed: true })
    process.env.OPENAI_API_KEY = 'test-openai-key'

    // Mock fetch for the OpenAI client_secrets endpoint (GA API)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        value: 'ephemeral-token-123',
        expires_at: 1700000000,
      }),
    }))
  })

  it('returns 401 when not authenticated', async () => {
    const mockResponse = new Response(JSON.stringify({ error: 'Authentication required' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    })
    vi.mocked(requireAuth).mockResolvedValue(mockResponse)

    const res = await handleRealtimeToken(makeRequest())
    expect(res.status).toBe(401)
  })

  it('returns 429 when rate limited', async () => {
    vi.mocked(checkRateLimit).mockReturnValue({ allowed: false })

    const res = await handleRealtimeToken(makeRequest())
    expect(res.status).toBe(429)
    const body = await res.json()
    expect(body.error).toContain('Too many realtime sessions')
  })

  it('calls OpenAI client_secrets endpoint with correct params', async () => {
    await handleRealtimeToken(makeRequest({
      setupContext: { topic: 'My pitch', audience: 'VCs' },
      researchContext: 'VC research data',
    }))

    // Verify buildRealtimeInstructions was called with correct args
    expect(buildRealtimeInstructions).toHaveBeenCalledWith({
      setupContext: { topic: 'My pitch', audience: 'VCs' },
      researchContext: 'VC research data',
    })

    // Verify fetch was called with correct GA endpoint
    expect(fetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/realtime/client_secrets',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer test-openai-key',
          'Content-Type': 'application/json',
        }),
      }),
    )

    // Verify request body — GA API uses session wrapper with nested audio config
    const fetchCall = vi.mocked(fetch).mock.calls[0]
    const requestBody = JSON.parse(fetchCall[1]!.body as string)
    expect(requestBody.session.model).toBe('gpt-4o-realtime-preview')
    expect(requestBody.session.instructions).toBe('mock-instructions')
    expect(requestBody.session.audio.input.transcription).toEqual({ model: 'whisper-1' })
    expect(requestBody.session.audio.input.turn_detection).toEqual({
      type: 'server_vad',
      threshold: 0.5,
      silence_duration_ms: 700,
    })
    expect(requestBody.session.audio.output.voice).toBe('sage')
  })

  it('returns ephemeral token on success', async () => {
    const res = await handleRealtimeToken(makeRequest())
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.token).toBe('ephemeral-token-123')
    expect(body.expiresAt).toBe(1700000000)
  })

  it('returns 502 when OpenAI endpoint fails', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('Internal server error'),
    } as Response)

    const res = await handleRealtimeToken(makeRequest())
    expect(res.status).toBe(502)
    const body = await res.json()
    expect(body.error).toContain('Failed to create realtime session')
  })

  it('returns 500 when OPENAI_API_KEY is not set', async () => {
    delete process.env.OPENAI_API_KEY

    const res = await handleRealtimeToken(makeRequest())
    expect(res.status).toBe(500)
    const body = await res.json()
    expect(body.error).toContain('not configured')
  })

  it('rate limits per user uid', async () => {
    await handleRealtimeToken(makeRequest())

    expect(checkRateLimit).toHaveBeenCalledWith(
      'realtime:user-1',
      expect.any(Number),
      expect.any(Number),
    )
  })
})

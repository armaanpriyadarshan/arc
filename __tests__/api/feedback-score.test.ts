import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

const mockCreate = vi.fn()

vi.mock('@/backend/openai', () => ({
  openai: vi.fn(() => ({
    chat: {
      completions: {
        create: mockCreate,
      },
    },
  })),
}))

vi.mock('@/backend/auth', () => ({
  requireAuth: vi.fn().mockResolvedValue({ uid: 'user123', email: 'test@test.com' }),
}))

import { handleFeedbackScore } from '@/backend/handlers/feedback-score'
import { requireAuth } from '@/backend/auth'

function createRequest(body: object): NextRequest {
  return new NextRequest('http://localhost/api/feedback-score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

const VALID_BODY = {
  sessionId: 'sess-1',
  messages: [
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there' },
  ],
  setup: {
    topic: 'My pitch',
    audience: 'VCs',
    goal: 'Get funded',
  },
}

const MOCK_SCORES = {
  overall: 78,
  delivery: 80,
  content: 75,
  engagement: 79,
}

describe('handleFeedbackScore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(requireAuth).mockResolvedValue({ uid: 'user123', email: 'test@test.com' })
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: JSON.stringify(MOCK_SCORES) } }],
    })
  })

  it('returns 401 when not authenticated', async () => {
    vi.mocked(requireAuth).mockResolvedValue(
      new Response(JSON.stringify({ error: 'Auth required' }), { status: 401 })
    )
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(401)
  })

  it('returns 400 for invalid request schema', async () => {
    const res = await handleFeedbackScore(createRequest({ foo: 'bar' }))
    expect(res.status).toBe(400)
    const data = await res.json()
    expect(data.error).toMatch(/invalid/i)
  })

  it('returns 400 when sessionId is missing', async () => {
    const { sessionId: _, ...noSession } = VALID_BODY
    const res = await handleFeedbackScore(createRequest(noSession))
    expect(res.status).toBe(400)
  })

  it('returns scores on successful request', async () => {
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.sessionId).toBe('sess-1')
    expect(data.scores).toEqual(MOCK_SCORES)
  })

  it('passes correct model parameters to OpenAI', async () => {
    await handleFeedbackScore(createRequest(VALID_BODY))
    expect(mockCreate).toHaveBeenCalledOnce()
    const callArgs = mockCreate.mock.calls[0][0]
    expect(callArgs.model).toBe('gpt-4o')
    expect(callArgs.temperature).toBe(0.3)
    expect(callArgs.max_tokens).toBe(5000)
    expect(callArgs.response_format).toEqual({ type: 'json_object' })
  })

  it('returns 500 when model returns empty content', async () => {
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: null } }],
    })
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(500)
    const data = await res.json()
    expect(data.error).toMatch(/no response/i)
  })

  it('returns 500 when model returns no choices', async () => {
    mockCreate.mockResolvedValue({ choices: [] })
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(500)
  })

  it('returns 500 when OpenAI throws', async () => {
    mockCreate.mockRejectedValue(new Error('API down'))
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(500)
    const data = await res.json()
    expect(data.error).toMatch(/failed/i)
  })

  it('includes transcript and researchContext when provided', async () => {
    const body = {
      ...VALID_BODY,
      transcript: 'Hello everyone...',
      researchContext: 'The audience prefers...',
    }
    await handleFeedbackScore(createRequest(body))
    expect(mockCreate).toHaveBeenCalledOnce()
    const prompt = mockCreate.mock.calls[0][0].messages[0].content
    expect(prompt).toContain('Hello everyone...')
    expect(prompt).toContain('The audience prefers...')
  })

  it('accepts deliveryAnalyticsSummary in request body', async () => {
    const body = {
      ...VALID_BODY,
      deliveryAnalyticsSummary: 'Speaking pace: 145 WPM average',
    }
    const res = await handleFeedbackScore(createRequest(body))
    expect(res.status).toBe(200)
    const prompt = mockCreate.mock.calls[0][0].messages[0].content
    expect(prompt).toContain('Speaking pace: 145 WPM average')
    expect(prompt).toContain('Delivery analytics')
  })
})

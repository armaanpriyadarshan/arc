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

const mockUpdate = vi.fn().mockResolvedValue(undefined)
vi.mock('@/backend/firebase-admin', () => ({
  db: vi.fn(() => ({
    collection: vi.fn(() => ({
      doc: vi.fn(() => ({
        update: mockUpdate,
      })),
    })),
  })),
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
  feedbackLetter: 'Great presentation.',
  rubric: [{ name: 'Clarity', score: 80, summary: 'Good', evidence: [] }],
  strongestMoment: { quote: 'test', why: 'good' },
  areaToImprove: { issue: 'pacing', suggestion: 'slow down' },
  refinedTitle: 'My Pitch',
  refinedAudience: 'VCs',
  refinedGoal: 'Get funded',
}

/** Create an async iterable that yields OpenAI-style stream chunks for a JSON string. */
function createMockStream(jsonStr: string) {
  // Split into small chunks to simulate streaming
  const chunks: string[] = []
  for (let i = 0; i < jsonStr.length; i += 10) {
    chunks.push(jsonStr.slice(i, i + 10))
  }

  return {
    [Symbol.asyncIterator]: async function* () {
      for (const chunk of chunks) {
        yield { choices: [{ delta: { content: chunk } }] }
      }
    },
  }
}

/** Parse SSE response body into an array of parsed events. */
async function parseSSEResponse(res: Response): Promise<{ type: string; [key: string]: unknown }[]> {
  const text = await res.text()
  const events: { type: string; [key: string]: unknown }[] = []

  for (const line of text.split('\n\n')) {
    if (!line.startsWith('data: ')) continue
    const data = line.slice(6).trim()
    if (data === '[DONE]') continue
    try {
      events.push(JSON.parse(data))
    } catch {
      // skip
    }
  }

  return events
}

describe('handleFeedbackScore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(requireAuth).mockResolvedValue({ uid: 'user123', email: 'test@test.com' })
    mockCreate.mockResolvedValue(createMockStream(JSON.stringify(MOCK_SCORES)))
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

  it('streams letter chunks and scores on successful request', async () => {
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(200)
    expect(res.headers.get('Content-Type')).toBe('text/event-stream')

    const events = await parseSSEResponse(res)

    // Should have letter_chunk events
    const letterChunks = events.filter((e) => e.type === 'letter_chunk')
    expect(letterChunks.length).toBeGreaterThan(0)

    // Concatenated letter chunks should equal the full letter
    const fullLetter = letterChunks.map((e) => e.content).join('')
    expect(fullLetter).toBe(MOCK_SCORES.feedbackLetter)

    // Should have a scores event
    const scoresEvent = events.find((e) => e.type === 'scores')
    expect(scoresEvent).toBeDefined()
    expect(scoresEvent!.scores).toEqual(MOCK_SCORES)
  })

  it('passes correct model parameters to OpenAI', async () => {
    await handleFeedbackScore(createRequest(VALID_BODY))
    expect(mockCreate).toHaveBeenCalledOnce()
    const callArgs = mockCreate.mock.calls[0][0]
    expect(callArgs.model).toBe('gpt-4o')
    expect(callArgs.temperature).toBe(0.3)
    expect(callArgs.max_tokens).toBe(5000)
    expect(callArgs.response_format).toEqual({ type: 'json_object' })
    expect(callArgs.stream).toBe(true)
  })

  it('writes scores to Firestore after stream completes', async () => {
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    // Consume the stream to trigger the Firestore write
    await res.text()

    // Wait a tick for the fire-and-forget Firestore write
    await new Promise((r) => setTimeout(r, 50))

    expect(mockUpdate).toHaveBeenCalledWith({ scores: MOCK_SCORES })
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

  it('emits error event when stream produces invalid JSON', async () => {
    mockCreate.mockResolvedValue({
      [Symbol.asyncIterator]: async function* () {
        yield { choices: [{ delta: { content: 'not valid json {{{' } }] }
      },
    })
    const res = await handleFeedbackScore(createRequest(VALID_BODY))
    expect(res.status).toBe(200) // SSE always returns 200

    const events = await parseSSEResponse(res)
    const errorEvent = events.find((e) => e.type === 'error')
    expect(errorEvent).toBeDefined()
  })
})

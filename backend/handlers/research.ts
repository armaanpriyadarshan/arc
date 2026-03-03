import { NextRequest, NextResponse } from 'next/server'
import { researchRequestSchema, sanitizeInput } from '@/backend/validation'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import { generateSearchTerms } from '@/backend/research/search-terms'
import { conductResearch } from '@/backend/research/web-research'

export async function handleResearch(request: NextRequest) {
  const ip = getClientIp(request)

  // Tight rate limit — research is expensive (web search + multiple model calls)
  if (!checkRateLimit('research:' + ip, RATE_LIMITS.research.limit, RATE_LIMITS.research.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many research requests. Please wait.' },
      { status: 429 }
    )
  }

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'Invalid request format' },
      { status: 400 }
    )
  }

  const parsed = researchRequestSchema.safeParse(body)
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid request format' },
      { status: 400 }
    )
  }

  const { transcript, audienceDescription, topic, goal, additionalContext } = parsed.data

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      let closed = false
      function send(event: string, data: Record<string, unknown>) {
        if (closed) return
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ event, ...data })}\n\n`))
        } catch {
          closed = true // client disconnected
        }
      }

      try {
        // Stage 1: Generate search terms
        const { searchTerms, audienceSummary } = await generateSearchTerms(
          sanitizeInput(audienceDescription),
          transcript ? sanitizeInput(transcript) : undefined,
          topic ? sanitizeInput(topic) : undefined,
          goal ? sanitizeInput(goal) : undefined,
          additionalContext ? sanitizeInput(additionalContext) : undefined,
        )

        send('terms', { searchTerms, audienceSummary })

        // Stage 2: Conduct web research
        const researchContext = await conductResearch(searchTerms, audienceSummary, goal)

        send('complete', { researchContext, audienceSummary, searchTerms })
      } catch (error) {
        console.error('Research pipeline error:', error)
        send('error', { error: 'Research failed. Coaching will proceed without enrichment.' })
      } finally {
        if (!closed) try { controller.close() } catch { /* already closed */ }
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}

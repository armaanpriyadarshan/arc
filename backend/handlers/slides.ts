import { NextRequest, NextResponse } from 'next/server'
import { openai } from '@/backend/openai'
import { slideAnalyzeRequestSchema } from '@/backend/validation'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import {
  extractSlideTextsAuto,
  type SlideFeedback,
  type DeckFeedback,
} from '@/backend/slides'
import {
  buildFullDeckSystemPrompt,
  buildFullDeckUserMessage,
} from '@/backend/slides-prompt'
import { downloadToTmp, cleanupTempFiles } from '@/backend/audio'
import { SSE_HEADERS } from '@/backend/request-utils'

function sseEvent(type: string, data: unknown): string {
  return `data: ${JSON.stringify({ type, data })}\n\n`
}

export async function handleSlidesAnalyze(request: NextRequest) {
  // Auth required — no trial mode for slides (Vision API cost)
  const authHeader = request.headers.get('authorization')
  if (!authHeader) {
    return NextResponse.json(
      { error: 'Sign in to use slide deck review.' },
      { status: 401 }
    )
  }

  const ip = getClientIp(request)
  if (!checkRateLimit('slides:' + ip, RATE_LIMITS.slidesAnalyze.limit, RATE_LIMITS.slidesAnalyze.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait a moment.' },
      { status: 429 }
    )
  }

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body.' },
      { status: 400 }
    )
  }

  const parsed = slideAnalyzeRequestSchema.safeParse(body)
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid request. Provide blobUrl and fileName.' },
      { status: 400 }
    )
  }

  const { blobUrl, fileName, audienceContext } = parsed.data
  const tempPaths: string[] = []

  const encoder = new TextEncoder()
  const readable = new ReadableStream({
    async start(controller) {
      const enqueue = (chunk: string) =>
        controller.enqueue(encoder.encode(chunk))

      try {
        // 1. Download PDF
        enqueue(sseEvent('status', { step: 'downloading' }))
        const downloadedPath = await downloadToTmp(blobUrl, fileName)
        tempPaths.push(downloadedPath)

        // 2. Extract text from slides (PDF or PPTX)
        enqueue(sseEvent('status', { step: 'rendering' }))
        const slideTexts = await extractSlideTextsAuto(downloadedPath)
        const totalSlides = slideTexts.length

        // 3. Single API call: analyze all slides with full deck context
        enqueue(sseEvent('status', { step: 'analyzing', total: totalSlides }))

        const client = openai()
        const systemPrompt = buildFullDeckSystemPrompt(totalSlides, audienceContext)
        const userMessage = buildFullDeckUserMessage(slideTexts)

        const response = await client.chat.completions.create({
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userMessage },
          ],
          response_format: { type: 'json_object' },
          temperature: 0.4,
          max_tokens: 8000,
        })

        const raw = response.choices[0]?.message?.content ?? '{}'
        const result = JSON.parse(raw) as {
          deckTitle?: string
          audienceAssumed?: string
          overallRating?: number
          executiveSummary?: string
          topPriorities?: string[]
          slides?: SlideFeedback[]
        }

        // 4. Emit per-slide feedback
        const slides: SlideFeedback[] = Array.isArray(result.slides)
          ? result.slides
          : []
        for (const slide of slides) {
          enqueue(sseEvent('slide_feedback', {
            ...slide,
            strengths: Array.isArray(slide.strengths) ? slide.strengths : [],
            improvements: Array.isArray(slide.improvements) ? slide.improvements : [],
          }))
        }

        // 5. Emit deck summary
        const deckFeedback: DeckFeedback = {
          deckTitle: result.deckTitle ?? 'Presentation',
          audienceAssumed: result.audienceAssumed ?? '',
          overallRating: result.overallRating ?? 0,
          executiveSummary: result.executiveSummary ?? '',
          slides,
          topPriorities: result.topPriorities ?? [],
        }
        enqueue(sseEvent('deck_summary', deckFeedback))

        enqueue('data: [DONE]\n\n')
        controller.close()
      } catch (err) {
        console.error('Slides analyze error:', err)
        enqueue(
          sseEvent('error', {
            message: 'Failed to analyze slide deck. Please try again.',
          })
        )
        controller.close()
      } finally {
        if (tempPaths.length > 0) {
          await cleanupTempFiles(tempPaths)
        }
      }
    },
  })

  return new Response(readable, {
    headers: SSE_HEADERS,
  })
}

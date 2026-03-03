import { NextRequest, NextResponse } from 'next/server'
import { openai } from '@/backend/openai'
import { chatRequestSchema, sanitizeInput } from '@/backend/validation'
import { buildSystemPrompt } from '@/backend/system-prompt'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import { verifyAuth } from '@/backend/auth'
import { SSE_HEADERS } from '@/backend/request-utils'

export async function handleChat(request: NextRequest) {
  const ip = getClientIp(request)
  if (!checkRateLimit('chat:' + ip, RATE_LIMITS.chatIp.limit, RATE_LIMITS.chatIp.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait a moment.' },
      { status: 429 }
    )
  }

  // Auth temporarily optional — all users get full access
  await verifyAuth(request)

  try {
    const body = await request.json()

    const parsed = chatRequestSchema.safeParse(body)
    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid request format' },
        { status: 400 }
      )
    }

    const { messages, transcript, researchContext, slideContext, stage, setupContext } = parsed.data

    const systemPrompt = buildSystemPrompt({
      stage,
      transcript,
      researchContext,
      slideContext,
      setupContext,
    })

    const openaiMessages: Array<{
      role: 'system' | 'user' | 'assistant'
      content: string
    }> = [
      { role: 'system', content: systemPrompt },
      ...messages.map((m) => ({
        role: m.role as 'user' | 'assistant',
        content: sanitizeInput(m.content),
      })),
    ]

    const client = openai()
    const stream = await client.chat.completions.create({
      model: 'gpt-4o',
      messages: openaiMessages,
      stream: true,
      temperature: 0.7,
      max_tokens: stage === 'feedback' ? 3000 : 2000,
    })

    const encoder = new TextEncoder()
    const readable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ content })}\n\n`)
              )
            }
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        } catch (err) {
          console.error('Stream error:', err)
          controller.error(err)
        }
      },
    })

    return new Response(readable, { headers: SSE_HEADERS })
  } catch (error) {
    console.error('Chat error:', error)
    return NextResponse.json(
      { error: 'Failed to generate response. Please try again.' },
      { status: 500 }
    )
  }
}

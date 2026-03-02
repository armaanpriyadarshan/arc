import { NextRequest, NextResponse } from 'next/server'
import { requireAuth } from '@/backend/auth'
import { checkRateLimit } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import { buildRealtimeInstructions } from '@/backend/system-prompt'
import type { SetupContext } from '@/lib/coaching-stages'

interface RealtimeTokenRequestBody {
  setupContext?: SetupContext
  researchContext?: string
}

export async function handleRealtimeToken(request: NextRequest) {
  // Require authentication — realtime sessions are expensive, no trial mode
  const authResult = await requireAuth(request)
  if (authResult instanceof Response) return authResult

  // Rate limit per user
  const rlKey = `realtime:${authResult.uid}`
  const { allowed } = checkRateLimit(
    rlKey,
    RATE_LIMITS.realtimeToken.limit,
    RATE_LIMITS.realtimeToken.windowMs,
  )
  if (!allowed) {
    return NextResponse.json(
      { error: 'Too many realtime sessions. Please wait a few minutes.' },
      { status: 429 },
    )
  }

  try {
    const body = (await request.json()) as RealtimeTokenRequestBody

    const instructions = buildRealtimeInstructions({
      setupContext: body.setupContext,
      researchContext: body.researchContext,
    })

    const apiKey = process.env.OPENAI_API_KEY
    if (!apiKey) {
      return NextResponse.json(
        { error: 'OpenAI API key not configured' },
        { status: 500 },
      )
    }

    // GA endpoint: POST /v1/realtime/client_secrets
    const response = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session: {
          type: 'realtime',
          model: 'gpt-4o-realtime-preview',
          instructions,
          audio: {
            input: {
              transcription: { model: 'whisper-1' },
              turn_detection: {
                type: 'server_vad',
                threshold: 0.5,
                silence_duration_ms: 700,
              },
            },
            output: {
              voice: 'sage',
            },
          },
        },
      }),
    })

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error')
      console.error('[realtime-token] OpenAI client_secrets endpoint failed:', response.status, errorText)
      return NextResponse.json(
        { error: 'Failed to create realtime session' },
        { status: 502 },
      )
    }

    const data = await response.json()

    return NextResponse.json({
      // GA response: { value: "...", expires_at: ... }
      token: data.value,
      expiresAt: data.expires_at,
    })
  } catch (error) {
    console.error('[realtime-token] Error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 },
    )
  }
}

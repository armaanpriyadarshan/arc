import { NextRequest, NextResponse } from "next/server"
import { openai } from "@/backend/openai"
import { requireAuth } from "@/backend/auth"
import { feedbackScoreRequestSchema } from "@/backend/validation"
import { buildScoringPrompt } from "@/backend/scoring-prompt"
import { SSE_HEADERS } from "@/backend/request-utils"
import { db } from "@/backend/firebase-admin"
import type { SessionScoresV2 } from "@/lib/sessions"

/**
 * Extract feedbackLetter content from a partial JSON buffer.
 *
 * The model emits JSON like: {"feedbackLetter":"I was genuinely...","rubric":[...]}
 * We detect the opening `"feedbackLetter":"` marker and then decode the string
 * value incrementally, handling JSON escape sequences (\n, \", \\, \uXXXX).
 *
 * Returns [extractedSoFar, previouslySent] so the caller can emit only the delta.
 */
function extractLetterContent(
  buffer: string,
  prevExtracted: number
): { text: string; newOffset: number } | null {
  // Find the start of the feedbackLetter value
  const marker = '"feedbackLetter"'
  const markerIdx = buffer.indexOf(marker)
  if (markerIdx === -1) return null

  // Find the opening quote of the string value
  let i = markerIdx + marker.length
  // skip whitespace and colon
  while (i < buffer.length && (buffer[i] === ' ' || buffer[i] === ':')) i++
  if (i >= buffer.length || buffer[i] !== '"') return null
  i++ // skip opening quote

  const startIdx = i
  let decoded = ""
  let pos = startIdx

  while (pos < buffer.length) {
    const ch = buffer[pos]

    if (ch === '"') {
      // End of string — return everything
      return { text: decoded.slice(prevExtracted), newOffset: decoded.length }
    }

    if (ch === '\\') {
      // Need at least one more character
      if (pos + 1 >= buffer.length) break // incomplete escape, wait for more

      const next = buffer[pos + 1]
      if (next === 'n') { decoded += '\n'; pos += 2 }
      else if (next === 't') { decoded += '\t'; pos += 2 }
      else if (next === 'r') { decoded += '\r'; pos += 2 }
      else if (next === '"') { decoded += '"'; pos += 2 }
      else if (next === '\\') { decoded += '\\'; pos += 2 }
      else if (next === '/') { decoded += '/'; pos += 2 }
      else if (next === 'u') {
        if (pos + 5 >= buffer.length) break // incomplete \uXXXX
        const hex = buffer.slice(pos + 2, pos + 6)
        decoded += String.fromCharCode(parseInt(hex, 16))
        pos += 6
      } else {
        decoded += next
        pos += 2
      }
    } else {
      decoded += ch
      pos++
    }
  }

  // String not yet terminated — emit what we have so far (minus a safety margin
  // of 1 character to avoid partial escape sequences)
  if (decoded.length > prevExtracted) {
    return { text: decoded.slice(prevExtracted), newOffset: decoded.length }
  }
  return null
}

export async function handleFeedbackScore(request: NextRequest) {
  const authResult = await requireAuth(request)
  if (authResult instanceof Response) return authResult

  try {
    const body = await request.json()
    const parsed = feedbackScoreRequestSchema.safeParse(body)
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request format" },
        { status: 400 }
      )
    }

    const { sessionId } = parsed.data
    const prompt = buildScoringPrompt(parsed.data)

    const stream = await openai().chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "system", content: prompt }],
      response_format: { type: "json_object" },
      temperature: 0.3,
      max_tokens: 5000,
      stream: true,
    })

    let fullBuffer = ""
    let letterOffset = 0

    const readable = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder()

        function sendSSE(data: string) {
          controller.enqueue(encoder.encode(`data: ${data}\n\n`))
        }

        try {
          for await (const chunk of stream) {
            const token = chunk.choices[0]?.delta?.content
            if (!token) continue

            fullBuffer += token

            // Try to extract letter content incrementally
            const result = extractLetterContent(fullBuffer, letterOffset)
            if (result && result.text.length > 0) {
              letterOffset = result.newOffset
              sendSSE(JSON.stringify({ type: "letter_chunk", content: result.text }))
            }
          }

          // Parse full JSON and emit scores
          const scores = JSON.parse(fullBuffer) as SessionScoresV2

          // Emit any remaining letter content not yet sent
          if (scores.feedbackLetter && letterOffset < scores.feedbackLetter.length) {
            const remaining = scores.feedbackLetter.slice(letterOffset)
            sendSSE(JSON.stringify({ type: "letter_chunk", content: remaining }))
          }

          sendSSE(JSON.stringify({ type: "scores", scores }))
          sendSSE("[DONE]")
          controller.close()

          // Write scores to Firestore via admin SDK (fire-and-forget)
          db().collection("sessions").doc(sessionId).update({ scores }).catch((err) => {
            console.error("[feedback-score] Firestore write failed:", err)
          })
        } catch (err) {
          console.error("[feedback-score] Stream error:", err)
          sendSSE(JSON.stringify({ type: "error", error: "Failed to generate scores" }))
          sendSSE("[DONE]")
          controller.close()
        }
      },
    })

    return new Response(readable, { headers: SSE_HEADERS })
  } catch (error) {
    console.error("Feedback scoring error:", error)
    return NextResponse.json(
      { error: "Failed to generate scores. Please try again." },
      { status: 500 }
    )
  }
}

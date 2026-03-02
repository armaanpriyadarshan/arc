import { NextRequest } from 'next/server'
import { openai } from '@/backend/openai'

const VALID_EMOTIONS = new Set([
  'neutral', 'interested', 'skeptical', 'confused',
  'amused', 'impressed', 'concerned', 'bored',
])

function isValidEmotion(v: unknown): v is string {
  return typeof v === 'string' && VALID_EMOTIONS.has(v)
}

const SYSTEM_PROMPT = `You are the audience — the specific person being presented to. The conversation tells you who you are (investor, executive, board member, etc.). You ARE that person. Write exactly 3 short inner thoughts.

If the presentation hasn't started yet (you only know the topic and who you are), generate anticipatory thoughts — what you're expecting, what you care about going in, what you're hoping to hear or dreading. You're sitting in the room waiting for it to begin.

If you've heard content, react to what was actually said. These are the real things going through your head. You walked in with your own priorities and your own skepticism. You're not a coach evaluating technique — you're a person in a chair deciding if this matters to you.

Rules:
- Each thought is 5–10 words, capitalized first letter, ending with a period
- First person — you are thinking these
- Grounded in specific things that were actually said, not generic
- Be honest. If something was unclear, you're confused. If a claim seems unsupported, you're skeptical. If you've heard this pitch 50 times, you're bored. If something genuinely surprised you, say so
- Don't default to positive thoughts. Real audience members are mostly neutral-to-skeptical. Positive reactions should be earned by genuinely strong moments
- Each thought has an emotion tag from: neutral, interested, skeptical, confused, amused, impressed, concerned, bored
- Return a JSON object: {"labels": [{"text": "...", "emotion": "..."}, ...]}

Good examples:
{"labels": [{"text": "That number doesn't match what I've seen.", "emotion": "skeptical"}, {"text": "Okay wait how does that actually work.", "emotion": "confused"}, {"text": "Still haven't heard the actual ask.", "emotion": "bored"}]}
{"labels": [{"text": "That's a bold claim with no evidence.", "emotion": "skeptical"}, {"text": "This part is more specific at least.", "emotion": "interested"}, {"text": "Wonder what their churn looks like.", "emotion": "neutral"}]}
{"labels": [{"text": "Already know where this is going.", "emotion": "bored"}, {"text": "The competitive angle is interesting though.", "emotion": "interested"}, {"text": "Would need to see this validated.", "emotion": "skeptical"}]}

Bad examples (too positive, too dramatic, too generic, too third-person):
{"labels": [{"text": "This is really compelling stuff.", "emotion": "impressed"}, {"text": "Great energy from the speaker.", "emotion": "impressed"}, {"text": "Love the vision here.", "emotion": "interested"}]}
{"labels": [{"text": "A room of seasoned investors leaning forward.", "emotion": "impressed"}, {"text": "The tension is palpable.", "emotion": "concerned"}, {"text": "Wondering if this will change everything.", "emotion": "impressed"}]}`

export async function handleAudiencePulse(request: NextRequest) {
  let body: { messages?: unknown }
  try {
    body = await request.json()
  } catch {
    return Response.json({ labels: [] })
  }

  const messages = Array.isArray(body?.messages) ? body.messages : []
  if (messages.length === 0) return Response.json({ labels: [] })

  const completion = await openai().chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      ...messages,
    ],
    max_tokens: 200,
    temperature: 0.8,
    response_format: { type: 'json_object' },
  })

  try {
    const raw = completion.choices[0]?.message?.content ?? '{}'
    const parsed = JSON.parse(raw)
    // Accept either { labels: [...] } or a bare array as the first array value found
    const rawLabels: unknown[] = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed.labels)
      ? parsed.labels
      : (Object.values(parsed).find(v => Array.isArray(v)) as unknown[] | undefined) ?? []

    // Normalize: accept both {text, emotion} objects and plain strings (backwards compat)
    const labels = rawLabels
      .slice(0, 3)
      .map((item) => {
        if (typeof item === 'string') {
          return { text: item, emotion: 'neutral' as const }
        }
        if (item && typeof item === 'object' && 'text' in item) {
          const obj = item as { text: unknown; emotion?: unknown }
          const text = typeof obj.text === 'string' ? obj.text : ''
          const emotion = isValidEmotion(obj.emotion)
            ? obj.emotion
            : 'neutral'
          return { text, emotion }
        }
        return null
      })
      .filter((item): item is { text: string; emotion: string } => item !== null && item.text.length > 0)

    return Response.json({ labels })
  } catch {
    return Response.json({ labels: [] })
  }
}

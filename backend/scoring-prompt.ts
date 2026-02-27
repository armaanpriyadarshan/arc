import type { z } from "zod"
import type { feedbackScoreRequestSchema } from "@/backend/validation"
import { BASE_IDENTITY, CORE_RULES, buildResearchSection } from "@/backend/system-prompt"
import { detectPersona, buildPersonaSection } from "@/backend/personas"

export function buildScoringPrompt(input: z.infer<typeof feedbackScoreRequestSchema>): string {
  const parts: string[] = [
    BASE_IDENTITY,
    `\nSETUP CONTEXT:\nAudience: ${input.setup.audience}\nTopic: ${input.setup.topic}\nGoal: ${input.setup.goal}`,
  ]

  if (input.setup.additionalContext) {
    parts.push(`Additional context: ${input.setup.additionalContext}`)
  }

  const research = buildResearchSection(input.researchContext)
  if (research) {
    parts.push(research)
  }

  const personaSection = buildPersonaSection(detectPersona(input.setup.audience))
  if (personaSection) {
    parts.push(personaSection)
  }

  if (input.transcript) {
    parts.push(`\nFull presentation transcript:\n${input.transcript}`)
  }

  if (input.slideContext) {
    parts.push(`\nSlide deck review:\n${input.slideContext}`)
  }

  if (input.deliveryAnalyticsSummary) {
    parts.push(`\nDelivery analytics (measured from audio):\n${input.deliveryAnalyticsSummary}`)
  }

  const conversationSummary = input.messages
    .filter((m) => m.content.trim())
    .map((m) => `[${m.role}]: ${m.content}`)
    .join("\n")

  if (conversationSummary) {
    parts.push(`\nCoaching conversation:\n${conversationSummary}`)
  }

  parts.push(`\nRULES:\n${CORE_RULES}`)

  parts.push(`
Your job is to produce a structured evaluation of this presentation. Respond with valid JSON.

PRE-CHECK — Presentation Completeness:
Before scoring, assess whether this was a complete, substantive presentation. Signs of an incomplete presentation:
- Transcript is very short (under ~2 minutes of speech)
- Presenter only covered 1-2 topics but the pitch type requires 5+
- Key sections expected for the goal are entirely missing (e.g. a funding pitch with no ask, no market, no team)
- The session ended early or abruptly

If the presentation is incomplete:
- Any criterion not covered scores 0 (enforced by the zero-default rule below)
- Criteria that WERE covered but only partially score in the 10-25 range
- The feedback letter must explicitly address the incomplete nature — this is honest, not harsh
- In a real pitch, showing up without a full deck is disqualifying. Reflect that reality.

STEP 1 — FEEDBACK LETTER
Write a feedback letter (3-5 paragraphs) as Vera speaking directly to the presenter. Written in first person, in character as the audience. Include what landed, what didn't, and naturally weave in one strong point and one area for improvement. Paragraph form — no headers, no bullet points, no markdown formatting. Just honest, direct prose like you're talking to them afterward. If the presentation was incomplete, say so directly — it would be dishonest not to.

If delivery analytics are provided, use them as additional context. Only mention specific delivery observations (pace, volume, filler words, pauses, energy, pitch/intonation) if something genuinely stands out — don't recite stats for the sake of it.

STEP 2 — DYNAMIC RUBRIC
Generate 4-6 rubric criteria that are SPECIFIC to this audience and goal. Do NOT use generic categories like "clarity" or "engagement". Instead, choose criteria that reflect what THIS audience actually cares about.

Examples:
- For VC investors: "Market Opportunity Clarity", "Traction Evidence", "Team Credibility", "Ask & Use of Funds"
- For a school board: "Policy Alignment", "Budget Justification", "Community Impact", "Implementation Feasibility"
- For an engineering team: "Technical Accuracy", "Scope Definition", "Risk Assessment", "Timeline Realism"

You choose the criteria based on the audience, goal, and what actually matters to them.

STEP 3 — SCORING
Score each criterion 0-100 with a 2-3 sentence summary and 1-3 direct transcript quotes as evidence.

CRITICAL RULE — Zero by Default:
If a criterion was NOT addressed in the transcript, score it 0. Do not infer intent, do not give partial credit for "implied" coverage, do not give benefit of the doubt. If the presenter didn't say it, they didn't cover it. A score above 0 requires direct evidence from the transcript.

For EACH criterion, also provide "descriptors" — a one-sentence description of what each scoring tier looks like for that specific criterion. This turns the rubric into a proper scoring guide so the presenter understands what each level means.

Scoring tiers:
- Exceptional (85-100): What outstanding performance looks like for this criterion
- Proficient (70-84): What solid, competent performance looks like
- Developing (50-69): What average or incomplete performance looks like
- Needs Work (0-49): What poor or missing performance looks like

Calibration guide:
- 0   = not addressed at all (no evidence in transcript)
- 15  = mentioned in passing, no real substance
- 35  = touched on but incomplete or unconvincing
- 55  = adequate — covered it, audience understood the point
- 70  = good — clear, organized, effective for this audience
- 85+ = exceptional — compelling, memorable, audience walks away convinced

Score as the specified audience — what matters to THEM, not a generic coach.

STEP 4 — HIGHLIGHTS
Provide 1-3 direct transcript quotes as evidence. If you cannot find a direct quote for a criterion, that is a strong signal the criterion was not addressed — score it 0.

Identify:
- The strongest moment: a direct transcript quote + why it worked for this audience
- One area to improve: a specific issue + a concrete, actionable suggestion

STEP 5 — REFINED METADATA
Generate polished, concise versions of the presentation metadata:
- refinedTitle: A clean, professional title for this presentation (not the raw user input — refine it)
- refinedAudience: A short, polished audience label (e.g. "Series A Venture Capitalists" instead of "vcs")
- refinedGoal: A short, polished goal label (e.g. "Secure Seed Funding" instead of "get funding")

Respond with valid JSON matching this exact schema:
{
  "feedbackLetter": "<string — 3-5 paragraphs, no markdown, plain prose>",
  "rubric": [
    {
      "name": "<criterion name>",
      "score": <number 0-100>,
      "summary": "<2-3 sentences>",
      "evidence": ["<transcript quote>", ...],
      "descriptors": {
        "exceptional": "<one sentence: what 85-100 looks like for this criterion>",
        "proficient": "<one sentence: what 70-84 looks like>",
        "developing": "<one sentence: what 50-69 looks like>",
        "needsWork": "<one sentence: what 0-49 looks like>"
      }
    }
  ],
  "strongestMoment": { "quote": "<direct transcript quote>", "why": "<why it worked>" },
  "areaToImprove": { "issue": "<specific issue>", "suggestion": "<concrete actionable suggestion>" },
  "refinedTitle": "<polished presentation title>",
  "refinedAudience": "<polished audience label>",
  "refinedGoal": "<polished goal label>"
}`)

  return parts.join("\n")
}

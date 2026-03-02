import { Document } from "@react-pdf/renderer"
import type { SessionScoresV2, SessionDocument } from "@/lib/sessions"
import { PdfPageShell } from "./pdf-page-shell"
import { PdfHeaderSection } from "./sections/pdf-header"
import { PdfFeedbackLetter } from "./sections/pdf-feedback-letter"
import { PdfRubric } from "./sections/pdf-rubric"
import { PdfHighlights } from "./sections/pdf-highlights"
import { PdfActionItems } from "./sections/pdf-action-items"
import { PdfTranscript } from "./sections/pdf-transcript"
import { PdfDeliveryFeedback } from "./sections/pdf-delivery-feedback"

export interface PdfReportProps {
  scores: SessionScoresV2
  setup: SessionDocument["setup"]
  transcript: string | null
  date: Date
}

/** Assembles all PDF sections into a complete @react-pdf/renderer Document. */
export function PdfReport({ scores, setup, transcript, date }: PdfReportProps) {
  const title = scores.refinedTitle ?? setup.topic
  const audience = scores.refinedAudience ?? setup.audience
  const goal = scores.refinedGoal ?? setup.goal

  return (
    <Document
      title={`${title} — Vera Feedback Report`}
      author="Vera"
      subject="Presentation Feedback Report"
    >
      <PdfPageShell>
        {/* Title block */}
        <PdfHeaderSection
          title={title}
          audience={audience}
          goal={goal}
          date={date}
        />

        {/* Feedback letter */}
        <PdfFeedbackLetter letter={scores.feedbackLetter} />

        {/* Highlights */}
        <PdfHighlights
          strongestMoment={scores.strongestMoment}
          areaToImprove={scores.areaToImprove}
        />

        {/* Action items */}
        {scores.actionItems && scores.actionItems.length > 0 && (
          <PdfActionItems items={scores.actionItems} />
        )}

        {/* Rubric cards */}
        <PdfRubric rubric={scores.rubric} />

        {/* Delivery feedback */}
        {scores.deliveryFeedback && scores.deliveryFeedback.length > 0 && (
          <PdfDeliveryFeedback observations={scores.deliveryFeedback} />
        )}

        {/* Transcript (on new page) */}
        {transcript && <PdfTranscript transcript={transcript} />}
      </PdfPageShell>
    </Document>
  )
}

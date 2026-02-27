import { Document } from "@react-pdf/renderer"
import type { SessionScoresV2, SessionDocument } from "@/lib/sessions"
import type { DeliveryAnalytics } from "@/lib/delivery-analytics"
import { PdfPageShell } from "./pdf-page-shell"
import { PdfHeaderSection } from "./sections/pdf-header"
import { PdfFeedbackLetter } from "./sections/pdf-feedback-letter"
import { PdfRubric } from "./sections/pdf-rubric"
import { PdfHighlights } from "./sections/pdf-highlights"
import { PdfDeliveryAnalytics } from "./sections/pdf-delivery-analytics"
import { PdfTranscript } from "./sections/pdf-transcript"

export interface PdfReportProps {
  scores: SessionScoresV2
  setup: SessionDocument["setup"]
  transcript: string | null
  date: Date
  analytics: DeliveryAnalytics | null
}

/** Assembles all PDF sections into a complete @react-pdf/renderer Document. */
export function PdfReport({ scores, setup, transcript, date, analytics }: PdfReportProps) {
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

        {/* Rubric radar + cards */}
        <PdfRubric rubric={scores.rubric} />

        {/* Delivery analytics */}
        {analytics && <PdfDeliveryAnalytics analytics={analytics} />}

        {/* Transcript (on new page) */}
        {transcript && <PdfTranscript transcript={transcript} />}
      </PdfPageShell>
    </Document>
  )
}

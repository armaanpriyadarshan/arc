"use client"

import { useState } from "react"
import { Activity, ChevronDown } from "lucide-react"
import { motion } from "framer-motion"
import type { DeliveryAnalytics } from "@/lib/delivery-analytics"
import { PaceChart } from "./delivery/pace-chart"
import { FillerWordBreakdown } from "./delivery/filler-word-breakdown"
import { PauseTimeline } from "./delivery/pause-timeline"
import { ContentPaceMap } from "./delivery/content-pace-map"

interface DeliveryAnalyticsSectionProps {
  analytics: DeliveryAnalytics
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  if (m === 0) return `${s}s`
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

export function DeliveryAnalyticsSection({ analytics }: DeliveryAnalyticsSectionProps) {
  const [open, setOpen] = useState(true)

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.12, ease: [0.16, 1, 0.3, 1] }}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 rounded-xl border border-border/60 bg-card px-5 py-4 text-left transition-colors hover:bg-muted/30"
      >
        <Activity className="h-4 w-4 text-muted-foreground/50" />
        <span className="flex-1 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
          Delivery Analytics
        </span>
        <span className="mr-2 rounded-full bg-muted px-2.5 py-0.5 text-[10px] font-medium text-muted-foreground">
          {analytics.averageWpm} WPM &middot; {formatDuration(analytics.totalDurationSeconds)}
        </span>
        <ChevronDown
          className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="mt-2 space-y-6 rounded-xl border border-border/60 bg-card px-5 py-5">
          {/* Summary cards */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <SummaryCard
              label="Avg Pace"
              value={`${analytics.averageWpm}`}
              unit="WPM"
            />
            <SummaryCard
              label="Filler Words"
              value={`${analytics.totalFillerCount}`}
              unit={`${analytics.fillersPerMinute}/min`}
            />
            <SummaryCard
              label="Pauses"
              value={`${analytics.totalPauseCount}`}
              unit=">1.5s"
            />
            <SummaryCard
              label="Duration"
              value={formatDuration(analytics.totalDurationSeconds)}
            />
          </div>

          <PaceChart
            paceWindows={analytics.paceWindows}
            averageWpm={analytics.averageWpm}
          />

          <ContentPaceMap
            contentSegments={analytics.contentSegments}
            averageWpm={analytics.averageWpm}
          />

          <FillerWordBreakdown
            fillerSummary={analytics.fillerSummary}
            totalFillerCount={analytics.totalFillerCount}
            fillersPerMinute={analytics.fillersPerMinute}
          />

          <PauseTimeline
            pauses={analytics.pauses}
            totalDurationSeconds={analytics.totalDurationSeconds}
          />
        </div>
      )}
    </motion.section>
  )
}

function SummaryCard({
  label,
  value,
  unit,
}: {
  label: string
  value: string
  unit?: string
}) {
  return (
    <div className="rounded-lg border border-border/40 bg-muted/20 px-3 py-2.5">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
        {label}
      </p>
      <p className="mt-0.5 text-lg font-semibold text-foreground/90">
        {value}
        {unit && (
          <span className="ml-1 text-xs font-normal text-muted-foreground">{unit}</span>
        )}
      </p>
    </div>
  )
}

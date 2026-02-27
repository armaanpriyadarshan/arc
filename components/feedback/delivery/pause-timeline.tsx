"use client"

import type { PauseInstance } from "@/lib/delivery-analytics"

interface PauseTimelineProps {
  pauses: PauseInstance[]
  totalDurationSeconds: number
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function PauseTimeline({ pauses, totalDurationSeconds }: PauseTimelineProps) {
  if (pauses.length === 0) {
    return (
      <div>
        <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
          Significant Pauses
        </h4>
        <p className="text-sm text-muted-foreground">
          No significant pauses detected (&gt;1.5s).
        </p>
      </div>
    )
  }

  return (
    <div>
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
        Significant Pauses ({pauses.length})
      </h4>

      {/* Visual timeline bar */}
      <div className="relative mb-4 h-6 w-full rounded-full bg-muted/40">
        {pauses.map((p, i) => {
          const leftPct = totalDurationSeconds > 0
            ? (p.start / totalDurationSeconds) * 100
            : 0
          return (
            <div
              key={i}
              className="absolute top-1/2 -translate-y-1/2"
              style={{ left: `${Math.min(leftPct, 98)}%` }}
              title={`${p.duration.toFixed(1)}s pause at ${formatTime(p.start)}`}
            >
              <div className="h-3 w-3 rounded-full bg-amber-500/80 ring-2 ring-background" />
            </div>
          )
        })}
      </div>

      {/* Pause list */}
      <div className="space-y-2">
        {pauses.map((p, i) => (
          <div
            key={i}
            className="flex items-start gap-3 rounded-lg border border-border/40 bg-muted/20 px-3 py-2"
          >
            <span className="shrink-0 rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-amber-600">
              {p.duration.toFixed(1)}s
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-xs text-muted-foreground">
                at {formatTime(p.start)}
              </p>
              <p className="mt-0.5 truncate text-xs text-foreground/70">
                &ldquo;...{p.precedingContext}&rdquo;
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

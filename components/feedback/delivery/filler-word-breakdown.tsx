"use client"

import type { FillerSummary } from "@/lib/delivery-analytics"

interface FillerWordBreakdownProps {
  fillerSummary: FillerSummary[]
  totalFillerCount: number
  fillersPerMinute: number
}

export function FillerWordBreakdown({
  fillerSummary,
  totalFillerCount,
  fillersPerMinute,
}: FillerWordBreakdownProps) {
  if (totalFillerCount === 0) {
    return (
      <div>
        <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
          Filler Words
        </h4>
        <p className="text-sm text-muted-foreground">
          No filler words detected — nice work keeping your language clean.
        </p>
      </div>
    )
  }

  const maxCount = fillerSummary[0]?.count ?? 1

  return (
    <div>
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
        Filler Words
      </h4>
      <p className="mb-3 text-xs text-muted-foreground">
        {totalFillerCount} total ({fillersPerMinute}/min)
      </p>
      <div className="space-y-2">
        {fillerSummary.map((f) => (
          <div key={f.phrase} className="flex items-center gap-3">
            <span className="w-20 shrink-0 text-right text-xs text-foreground/80">
              &ldquo;{f.phrase}&rdquo;
            </span>
            <div className="flex-1">
              <div
                className="h-5 rounded-sm bg-primary/20"
                style={{ width: `${Math.max(8, (f.count / maxCount) * 100)}%` }}
              >
                <span className="flex h-full items-center pl-2 text-[10px] font-medium text-foreground/70">
                  {f.count}x ({f.perMinute}/min)
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[10px] text-muted-foreground/50">
        Note: contextual uses of &ldquo;like&rdquo;, &ldquo;so&rdquo;, etc. may include non-filler instances.
      </p>
    </div>
  )
}

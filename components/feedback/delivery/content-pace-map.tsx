"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts"
import type { ContentSegment } from "@/lib/delivery-analytics"

interface ContentPaceMapProps {
  contentSegments: ContentSegment[]
  averageWpm: number
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function ContentPaceMap({ contentSegments, averageWpm }: ContentPaceMapProps) {
  if (contentSegments.length <= 1) return null

  const data = contentSegments.map((seg) => ({
    label: `${formatTime(seg.startTime)}-${formatTime(seg.endTime)}`,
    wpm: seg.wpm,
    topic: seg.topicLabel,
  }))

  return (
    <div>
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
        Pace by Content Section
      </h4>
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [`${value} WPM`, "Pace"]}
              labelFormatter={(_label: string, payload: { payload?: { topic?: string } }[]) =>
                payload?.[0]?.payload?.topic ?? ""
              }
            />
            <ReferenceLine
              y={averageWpm}
              stroke="hsl(var(--muted-foreground))"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
              label={{
                value: `avg ${averageWpm}`,
                position: "right",
                fontSize: 10,
                fill: "hsl(var(--muted-foreground))",
              }}
            />
            <Bar dataKey="wpm" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={index}
                  fill={
                    entry.wpm > averageWpm
                      ? "hsl(30 90% 55%)" // orange — rushed
                      : "hsl(210 70% 55%)" // blue — measured
                  }
                  fillOpacity={0.7}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex gap-4 text-[10px] text-muted-foreground/60">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[hsl(210_70%_55%)]" /> Below avg (measured)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[hsl(30_90%_55%)]" /> Above avg (rushed)
        </span>
      </div>
    </div>
  )
}

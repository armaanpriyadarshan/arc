"use client"

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from "recharts"
import type { PaceWindow } from "@/lib/delivery-analytics"

interface PaceChartProps {
  paceWindows: PaceWindow[]
  averageWpm: number
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function PaceChart({ paceWindows, averageWpm }: PaceChartProps) {
  if (paceWindows.length === 0) return null

  const data = paceWindows.map((w) => ({
    time: formatTime(w.startTime),
    wpm: w.wpm,
  }))

  return (
    <div>
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
        Speaking Pace Over Time
      </h4>
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <defs>
              <linearGradient id="paceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(36 72% 50%)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(36 72% 50%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
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
              labelStyle={{ color: "hsl(var(--muted-foreground))" }}
              formatter={(value: number) => [`${value} WPM`, "Pace"]}
            />
            {/* Ideal zone band: 120-160 WPM */}
            <ReferenceArea y1={120} y2={160} fill="hsl(142 71% 45%)" fillOpacity={0.06} />
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
            <Area
              type="monotone"
              dataKey="wpm"
              stroke="hsl(36 72% 50%)"
              strokeWidth={2}
              fill="url(#paceGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground/60">
        Green band: ideal conversational range (120-160 WPM)
      </p>
    </div>
  )
}

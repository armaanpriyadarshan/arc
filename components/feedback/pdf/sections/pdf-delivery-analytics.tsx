import { View, Text, StyleSheet } from "@react-pdf/renderer"
import { colors, fonts, fontSize } from "../theme"
import type { DeliveryAnalytics } from "@/lib/delivery-analytics"

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  heading: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    color: colors.primary,
    marginBottom: 10,
  },
  statsRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 10,
  },
  statBox: {
    flex: 1,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 4,
    padding: 8,
  },
  statLabel: {
    fontFamily: fonts.sans,
    fontSize: fontSize.xs,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    color: colors.textMuted,
    marginBottom: 2,
  },
  statValue: {
    fontFamily: fonts.sans,
    fontSize: fontSize.md,
    fontWeight: 700,
    color: colors.text,
  },
  statUnit: {
    fontFamily: fonts.sans,
    fontSize: fontSize.xs,
    fontWeight: 400,
    color: colors.textSecondary,
  },
  subheading: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    fontWeight: 600,
    color: colors.text,
    marginTop: 8,
    marginBottom: 4,
  },
  body: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    lineHeight: 1.5,
    color: colors.textSecondary,
  },
  tableRow: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: colors.borderLight,
    paddingVertical: 3,
  },
  tableCell: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    color: colors.textSecondary,
  },
})

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  if (m === 0) return `${s}s`
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

interface PdfDeliveryAnalyticsProps {
  analytics: DeliveryAnalytics
}

export function PdfDeliveryAnalytics({ analytics }: PdfDeliveryAnalyticsProps) {
  return (
    <View style={styles.container} wrap={false}>
      <Text style={styles.heading}>Delivery Analytics</Text>

      {/* Key stats */}
      <View style={styles.statsRow}>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>Avg Pace</Text>
          <Text style={styles.statValue}>
            {analytics.averageWpm} <Text style={styles.statUnit}>WPM</Text>
          </Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>Filler Words</Text>
          <Text style={styles.statValue}>
            {analytics.totalFillerCount} <Text style={styles.statUnit}>({analytics.fillersPerMinute}/min)</Text>
          </Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>Pauses</Text>
          <Text style={styles.statValue}>
            {analytics.totalPauseCount} <Text style={styles.statUnit}>&gt;1.5s</Text>
          </Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>Duration</Text>
          <Text style={styles.statValue}>
            {formatDuration(analytics.totalDurationSeconds)}
          </Text>
        </View>
      </View>

      {/* Pace by section */}
      {analytics.contentSegments.length > 1 && (
        <View>
          <Text style={styles.subheading}>Pace by Section</Text>
          {analytics.contentSegments.map((seg, i) => (
            <View key={i} style={styles.tableRow}>
              <Text style={[styles.tableCell, { width: 80 }]}>
                {formatTime(seg.startTime)}-{formatTime(seg.endTime)}
              </Text>
              <Text style={[styles.tableCell, { width: 50, fontWeight: 600 }]}>
                {seg.wpm} WPM
              </Text>
              <Text style={[styles.tableCell, { flex: 1 }]} numberOfLines={1}>
                {seg.topicLabel}
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* Top fillers */}
      {analytics.fillerSummary.length > 0 && (
        <View>
          <Text style={styles.subheading}>Top Filler Words</Text>
          <Text style={styles.body}>
            {analytics.fillerSummary
              .slice(0, 5)
              .map((f) => `"${f.phrase}" (${f.count}x)`)
              .join(", ")}
          </Text>
        </View>
      )}

      {/* Longest pause */}
      {analytics.longestPause && (
        <View>
          <Text style={styles.subheading}>Longest Pause</Text>
          <Text style={styles.body}>
            {analytics.longestPause.duration.toFixed(1)}s at {formatTime(analytics.longestPause.start)}
            {" — after \""}{analytics.longestPause.precedingContext}{"\""}
          </Text>
        </View>
      )}
    </View>
  )
}

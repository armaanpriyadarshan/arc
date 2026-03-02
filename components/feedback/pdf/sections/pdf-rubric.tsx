import { View, Text, StyleSheet } from "@react-pdf/renderer"
import type { RubricCriterion } from "@/lib/sessions"
import { colors, fonts, fontSize, getScoringTier } from "../theme"

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontFamily: fonts.sans,
    fontSize: fontSize.lg,
    fontWeight: 600,
    color: colors.text,
    marginBottom: 12,
  },
  card: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 6,
    padding: 12,
    marginBottom: 8,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  criterionName: {
    fontFamily: fonts.sans,
    fontSize: fontSize.md,
    fontWeight: 600,
    color: colors.text,
  },
  tierBadge: {
    borderRadius: 8,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  tierBadgeText: {
    fontFamily: fonts.sans,
    fontSize: fontSize.xs,
    fontWeight: 600,
  },
  scoreText: {
    fontFamily: fonts.sans,
    fontSize: fontSize.xs,
    color: colors.textMuted,
    marginLeft: 6,
  },
  summary: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    lineHeight: 1.5,
    color: colors.textSecondary,
    marginBottom: 6,
  },
  evidenceQuote: {
    fontFamily: fonts.sans,
    fontSize: fontSize.xs,
    fontStyle: "italic",
    lineHeight: 1.5,
    color: colors.textMuted,
    borderLeftWidth: 2,
    borderLeftColor: colors.primary,
    paddingLeft: 8,
    marginTop: 4,
    marginLeft: 2,
  },
})

interface PdfRubricProps {
  rubric: RubricCriterion[]
}

/** Radar chart + rubric criterion cards with scores, summaries, and evidence. */
export function PdfRubric({ rubric }: PdfRubricProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Scoring Rubric</Text>

      {/* Criterion cards */}
      {rubric.map((c) => {
        const tier = getScoringTier(c.score)
        return (
          <View key={c.name} style={styles.card} wrap={false}>
            <View style={styles.cardHeader}>
              <Text style={styles.criterionName}>{c.name}</Text>
              <View style={{ flexDirection: "row", alignItems: "center" }}>
                <View style={[styles.tierBadge, { backgroundColor: tier.bg }]}>
                  <Text style={[styles.tierBadgeText, { color: tier.color }]}>
                    {tier.label}
                  </Text>
                </View>
                <Text style={styles.scoreText}>{c.score}</Text>
              </View>
            </View>
            <Text style={styles.summary}>{c.summary}</Text>
            {c.evidence.slice(0, 2).map((quote, i) => (
              <Text key={i} style={styles.evidenceQuote}>
                &ldquo;{quote}&rdquo;
              </Text>
            ))}
          </View>
        )
      })}
    </View>
  )
}

import { View, Text, StyleSheet } from "@react-pdf/renderer"
import { colors, fonts, fontSize } from "../theme"

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
  },
  item: {
    flexDirection: "row",
    padding: 12,
    gap: 10,
  },
  divider: {
    borderTopWidth: 1,
    borderTopColor: colors.borderLight,
  },
  number: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    fontWeight: 600,
    color: colors.primary,
    width: 16,
    textAlign: "center",
  },
  content: {
    flex: 1,
  },
  title: {
    fontFamily: fonts.sans,
    fontSize: fontSize.md,
    fontWeight: 600,
    color: colors.text,
    marginBottom: 2,
  },
  description: {
    fontFamily: fonts.sans,
    fontSize: fontSize.sm,
    lineHeight: 1.5,
    color: colors.textSecondary,
  },
})

interface PdfActionItemsProps {
  items: { title: string; description: string }[]
}

export function PdfActionItems({ items }: PdfActionItemsProps) {
  if (items.length === 0) return null

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Key Action Items</Text>
      <View style={styles.card}>
        {items.map((item, i) => (
          <View key={i} style={[styles.item, i > 0 ? styles.divider : {}]} wrap={false}>
            <Text style={styles.number}>{i + 1}</Text>
            <View style={styles.content}>
              <Text style={styles.title}>{item.title}</Text>
              <Text style={styles.description}>{item.description}</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  )
}

"use client"

import { motion } from "framer-motion"

interface ActionItemsProps {
  items: { title: string; description: string }[]
}

export function ActionItems({ items }: ActionItemsProps) {
  if (items.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-xl border border-border/60 bg-card"
    >
      {items.map((item, i) => (
        <div
          key={i}
          className={`flex gap-4 px-5 py-4 ${i > 0 ? "border-t border-border/40" : ""}`}
        >
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
            {i + 1}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">{item.title}</p>
            <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">
              {item.description}
            </p>
          </div>
        </div>
      ))}
    </motion.div>
  )
}

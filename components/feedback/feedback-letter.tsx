"use client"

import ReactMarkdown from "react-markdown"
import { motion } from "framer-motion"

interface FeedbackLetterProps {
  letter: string
  isStreaming?: boolean
}

export function FeedbackLetter({ letter, isStreaming }: FeedbackLetterProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="prose prose-sm max-w-none text-[0.9375rem] leading-[1.85] text-foreground/90 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_p]:mb-4 [&_strong]:text-foreground [&_blockquote]:border-primary/20 [&_blockquote]:text-foreground/70">
        <ReactMarkdown>{letter}</ReactMarkdown>
        {isStreaming && (
          <span className="inline-block h-4 w-0.5 animate-pulse bg-foreground/60" aria-hidden="true" />
        )}
      </div>
    </motion.section>
  )
}

"use client"

import React, { useState, useRef, useEffect, useLayoutEffect, useCallback } from "react"
import dynamic from "next/dynamic"
import { AnimatePresence, motion } from "framer-motion"
import { Loader2, ArrowRight, Upload, Mic, Paperclip, FileText, X } from "lucide-react"
import { toast } from "sonner"
import { FadeIn } from "@/components/motion"

const ResearchCard = dynamic(() => import("@/components/research-card").then(m => ({ default: m.ResearchCard })))
import type { SetupContext } from "@/lib/coaching-stages"
import type { ResearchMeta } from "@/hooks/use-research-pipeline"
import type { ContextFileInfo } from "@/hooks/use-context-file"

const SETUP_EXAMPLES = [
  { topic: "my Series A pitch", audience: "VC investors", goal: "secure funding" },
  { topic: "a Q3 revenue review", audience: "the board of directors", goal: "get buy-in" },
  { topic: "our product roadmap", audience: "my engineering team", goal: "ship on time" },
  { topic: "a keynote talk", audience: "500 conference attendees", goal: "inspire action" },
  { topic: "a client proposal", audience: "the procurement team", goal: "close the deal" },
]

interface SetupWizardProps {
  isResearching: boolean
  researchMeta: ResearchMeta | null
  researchSearchTerms: string[] | null
  isCompressing: boolean
  isTranscribing: boolean
  onResearchStart: (audience: string, opts?: { topic?: string; goal?: string; additionalContext?: string }) => void
  onModeSelect: (mode: "present" | "upload-recording", setupContext: SetupContext | null, contextMessage: string | null) => void
  /** If provided, called instead of the normal submit flow (e.g. to redirect unauthenticated users). */
  onReady?: () => void
  contextFile?: ContextFileInfo | null
  isExtractingContext?: boolean
  contextFileError?: string | null
  onContextFileUpload?: (file: File) => void
  onContextFileRemove?: () => void
}

export const SetupWizard = React.memo(function SetupWizard({
  isResearching,
  researchMeta,
  researchSearchTerms,
  isCompressing,
  isTranscribing,
  onResearchStart,
  onModeSelect,
  onReady,
  contextFile,
  isExtractingContext,
  contextFileError,
  onContextFileUpload,
  onContextFileRemove,
}: SetupWizardProps) {
  const [setupTopic, setSetupTopic] = useState("")
  const [setupAudience, setSetupAudience] = useState("")
  const [setupGoal, setSetupGoal] = useState("")
  const [setupAdditional, setSetupAdditional] = useState("")
  const [showAdditional, setShowAdditional] = useState(false)
  const [setupPhase, setSetupPhase] = useState<"fields" | "researching" | "review" | "mode-select" | "uploading">("fields")
  const contextFileInputRef = useRef<HTMLInputElement>(null)
  const additionalTextareaRef = useRef<HTMLTextAreaElement>(null)
  const filePickerOpenRef = useRef(false)

  /* ── Example cycling ── */
  const [exampleIndex, setExampleIndex] = useState(0)
  const [sizerIndex, setSizerIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => setExampleIndex((i) => (i + 1) % SETUP_EXAMPLES.length), 3000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const t = setTimeout(() => setSizerIndex(exampleIndex), 250)
    return () => clearTimeout(t)
  }, [exampleIndex])

  /* ── Sizer measurement ── */
  const topicSizerRef = useRef<HTMLSpanElement>(null)
  const audienceSizerRef = useRef<HTMLSpanElement>(null)
  const goalSizerRef = useRef<HTMLSpanElement>(null)
  const additionalSizerRef = useRef<HTMLSpanElement>(null)
  const [blankWidths, setBlankWidths] = useState({ topic: 0, audience: 0, goal: 0 })
  const [additionalLineWidth, setAdditionalLineWidth] = useState(0)
  const hasMeasuredRef = useRef(false)

  const measureWidths = useCallback(() => {
    setBlankWidths({
      topic: topicSizerRef.current?.offsetWidth ?? 0,
      audience: audienceSizerRef.current?.offsetWidth ?? 0,
      goal: goalSizerRef.current?.offsetWidth ?? 0,
    })
  }, [])

  useLayoutEffect(() => {
    measureWidths()
  }, [setupTopic, setupAudience, setupGoal, sizerIndex, measureWidths])

  useLayoutEffect(() => {
    const w = additionalSizerRef.current?.offsetWidth ?? 0
    setAdditionalLineWidth(Math.min(w, 320))
  }, [setupAdditional, showAdditional])

  useEffect(() => {
    document.fonts.ready.then(() => {
      measureWidths()
      requestAnimationFrame(() => { hasMeasuredRef.current = true })
    })
  }, [measureWidths])

  // When the file picker dialog closes, window regains focus — re-focus the textarea
  // and clear the flag so blur works normally again
  useEffect(() => {
    if (!showAdditional) return
    const onWindowFocus = () => {
      if (!filePickerOpenRef.current) return
      filePickerOpenRef.current = false
      additionalTextareaRef.current?.focus()
    }
    window.addEventListener('focus', onWindowFocus)
    return () => window.removeEventListener('focus', onWindowFocus)
  }, [showAdditional])

  /* ── Research snippet cycling ── */
  const [researchSnippetIndex, setResearchSnippetIndex] = useState(0)
  const researchSnippet = researchSearchTerms
    ? researchSearchTerms[researchSnippetIndex % researchSearchTerms.length]
    : null

  useEffect(() => {
    if (!researchSearchTerms || researchSearchTerms.length <= 1) return
    setResearchSnippetIndex(0)
    const interval = setInterval(
      () => setResearchSnippetIndex((i) => (i + 1) % researchSearchTerms.length),
      2500
    )
    return () => clearInterval(interval)
  }, [researchSearchTerms])

  /* ── Phase transitions ── */
  useEffect(() => {
    if (setupPhase === "researching" && !isResearching) {
      if (researchMeta) {
        setSetupPhase("review")
      } else {
        toast("Audience research unavailable — proceeding without it")
        setSetupPhase("mode-select")
      }
    }
  }, [setupPhase, isResearching, researchMeta])

  useEffect(() => {
    if (setupPhase === "uploading" && !isCompressing && !isTranscribing) {
      setSetupPhase("fields")
    }
  }, [setupPhase, isCompressing, isTranscribing])

  // Enter to proceed from research review phase
  useEffect(() => {
    if (setupPhase !== "review" || !researchMeta) return
    let armed = false
    function handleKeyUp(e: KeyboardEvent) { if (e.key === "Enter") armed = true }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Enter" && armed) { e.preventDefault(); setSetupPhase("mode-select") }
    }
    window.addEventListener("keyup", handleKeyUp)
    window.addEventListener("keydown", handleKeyDown)
    return () => { window.removeEventListener("keyup", handleKeyUp); window.removeEventListener("keydown", handleKeyDown) }
  }, [setupPhase, researchMeta])

  // Show toast when context file extraction fails
  useEffect(() => {
    if (contextFileError) toast.error(contextFileError)
  }, [contextFileError])

  // Enter to default to "present live" on mode-select phase
  useEffect(() => {
    if (setupPhase !== "mode-select") return
    let armed = false
    function handleKeyUp(e: KeyboardEvent) { if (e.key === "Enter") armed = true }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Enter" && armed) { e.preventDefault(); handleModeSelectInternal("present") }
    }
    window.addEventListener("keyup", handleKeyUp)
    window.addEventListener("keydown", handleKeyDown)
    return () => { window.removeEventListener("keyup", handleKeyUp); window.removeEventListener("keydown", handleKeyDown) }
  }, [setupPhase]) // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Helpers ── */
  const hasSetupContent = !!(setupTopic.trim() && setupAudience.trim() && setupGoal.trim())

  function buildSetupContext(): SetupContext | null {
    const topic = setupTopic.trim()
    const audience = setupAudience.trim()
    const goal = setupGoal.trim()
    const additionalContext = setupAdditional.trim()
    if (!topic && !audience && !goal && !additionalContext) return null
    return {
      ...(topic ? { topic } : {}),
      ...(audience ? { audience } : {}),
      ...(goal ? { goal } : {}),
      ...(additionalContext ? { additionalContext } : {}),
    }
  }

  function buildContextMessage(): string | null {
    const parts: string[] = []
    if (setupTopic.trim()) parts.push(`I'm presenting on: ${setupTopic.trim()}`)
    if (setupAudience.trim()) parts.push(`My audience is: ${setupAudience.trim()}`)
    if (setupGoal.trim()) parts.push(`My goal is to: ${setupGoal.trim()}`)
    if (setupAdditional.trim()) parts.push(`Additional context: ${setupAdditional.trim()}`)
    return parts.length > 0 ? parts.join(". ") + "." : null
  }

  function handleSetupSubmit() {
    if (!hasSetupContent) return
    if (onReady) { onReady(); return }
    const audience = setupAudience.trim()
    if (audience) {
      setSetupPhase("researching")
      onResearchStart(audience, {
        topic: setupTopic.trim() || undefined,
        goal: setupGoal.trim() || undefined,
        additionalContext: setupAdditional.trim() || undefined,
      })
    } else {
      setSetupPhase("mode-select")
    }
  }

  function handleSetupKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && hasSetupContent) {
      e.preventDefault()
      handleSetupSubmit()
    }
  }

  function handleModeSelectInternal(mode: "present" | "upload-recording") {
    onModeSelect(mode, buildSetupContext(), buildContextMessage())
  }

  /* ── Render ── */
  return (
    <motion.div
      key="empty-state"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="relative flex flex-1 flex-col items-center justify-center overflow-y-auto overflow-x-hidden px-4 py-8 sm:px-6"
    >
      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden="true">
        <div className="absolute -left-40 -top-40 h-[500px] w-[500px] rounded-full opacity-[0.08] blur-3xl">
          <div className="h-full w-full rounded-full" style={{ background: "radial-gradient(circle, hsl(36 56% 48% / 0.6), transparent 70%)" }} />
        </div>
        <div className="absolute -right-32 top-1/3 h-[400px] w-[400px] rounded-full opacity-[0.06] blur-3xl">
          <div className="h-full w-full rounded-full" style={{ background: "radial-gradient(circle, hsl(34 35% 74%), transparent 70%)" }} />
        </div>
      </div>

      <div className="flex w-full max-w-2xl flex-col items-center overflow-hidden">
        <AnimatePresence mode="wait">
          {setupPhase === "fields" && (
            <motion.div
              key="setup-fields"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="flex w-full flex-col items-center"
            >
              <FadeIn delay={0.1}>
                <div className="mt-2 flex w-full flex-col items-center gap-6 font-display text-center text-lg text-muted-foreground sm:text-xl md:text-2xl">
                  {/* Topic */}
                  <div className="flex flex-col items-center">
                    <span>I&apos;m presenting</span>
                    <span className="relative mt-1 inline-block pb-1" style={{ minWidth: 60 }}>
                      <span ref={topicSizerRef} className="invisible whitespace-nowrap px-1">{setupTopic ? setupTopic : SETUP_EXAMPLES[sizerIndex].topic}</span>
                      <input
                        type="text"
                        value={setupTopic}
                        onChange={(e) => setSetupTopic(e.target.value)}
                        onKeyDown={handleSetupKeyDown}
                        className="absolute inset-x-0 top-0 z-10 w-full bg-transparent text-center text-foreground caret-primary focus:outline-none"
                      />
                      {!setupTopic && (
                        <AnimatePresence mode="wait">
                          <motion.span
                            key={SETUP_EXAMPLES[exampleIndex].topic}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.25, ease: "easeInOut" }}
                            className="pointer-events-none absolute inset-x-0 top-0 whitespace-nowrap text-center text-muted-foreground/30"
                          >
                            {SETUP_EXAMPLES[exampleIndex].topic}
                          </motion.span>
                        </AnimatePresence>
                      )}
                      <motion.span
                        className="absolute bottom-0 left-1/2 h-[2.5px] -translate-x-1/2 rounded-full bg-primary/30"
                        initial={false}
                        animate={{ width: blankWidths.topic + 20 }}
                        transition={hasMeasuredRef.current ? { duration: 0.35, ease: "easeInOut" } : { duration: 0 }}
                      />
                    </span>
                  </div>

                  {/* Audience */}
                  <div className="flex flex-col items-center">
                    <span>to</span>
                    <span className="relative mt-1 inline-block pb-1" style={{ minWidth: 60 }}>
                      <span ref={audienceSizerRef} className="invisible whitespace-nowrap px-1">{setupAudience ? setupAudience : SETUP_EXAMPLES[sizerIndex].audience}</span>
                      <input
                        type="text"
                        value={setupAudience}
                        onChange={(e) => setSetupAudience(e.target.value)}
                        onKeyDown={handleSetupKeyDown}
                        className="absolute inset-x-0 top-0 z-10 w-full bg-transparent text-center text-foreground caret-primary focus:outline-none"
                      />
                      {!setupAudience && (
                        <AnimatePresence mode="wait">
                          <motion.span
                            key={SETUP_EXAMPLES[exampleIndex].audience}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.25, ease: "easeInOut" }}
                            className="pointer-events-none absolute inset-x-0 top-0 whitespace-nowrap text-center text-muted-foreground/30"
                          >
                            {SETUP_EXAMPLES[exampleIndex].audience}
                          </motion.span>
                        </AnimatePresence>
                      )}
                      <motion.span
                        className="absolute bottom-0 left-1/2 h-[2.5px] -translate-x-1/2 rounded-full bg-primary/30"
                        initial={false}
                        animate={{ width: blankWidths.audience + 20 }}
                        transition={hasMeasuredRef.current ? { duration: 0.35, ease: "easeInOut" } : { duration: 0 }}
                      />
                    </span>
                  </div>

                  {/* Goal */}
                  <div className="flex flex-col items-center">
                    <span>and I want to</span>
                    <span className="relative mt-1 inline-block pb-1" style={{ minWidth: 60 }}>
                      <span ref={goalSizerRef} className="invisible whitespace-nowrap px-1">{setupGoal ? setupGoal : SETUP_EXAMPLES[sizerIndex].goal}</span>
                      <input
                        type="text"
                        value={setupGoal}
                        onChange={(e) => setSetupGoal(e.target.value)}
                        onKeyDown={handleSetupKeyDown}
                        className="absolute inset-x-0 top-0 z-10 w-full bg-transparent text-center text-foreground caret-primary focus:outline-none"
                      />
                      {!setupGoal && (
                        <AnimatePresence mode="wait">
                          <motion.span
                            key={SETUP_EXAMPLES[exampleIndex].goal}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.25, ease: "easeInOut" }}
                            className="pointer-events-none absolute inset-x-0 top-0 whitespace-nowrap text-center text-muted-foreground/30"
                          >
                            {SETUP_EXAMPLES[exampleIndex].goal}
                          </motion.span>
                        </AnimatePresence>
                      )}
                      <motion.span
                        className="absolute bottom-0 left-1/2 h-[2.5px] -translate-x-1/2 rounded-full bg-primary/30"
                        initial={false}
                        animate={{ width: blankWidths.goal + 20 }}
                        transition={hasMeasuredRef.current ? { duration: 0.35, ease: "easeInOut" } : { duration: 0 }}
                      />
                    </span>
                  </div>
                </div>
              </FadeIn>

              {/* Optional additional context */}
              <FadeIn delay={0.15}>
                <div className="mt-6 flex justify-center">
                  {!showAdditional ? (
                    <button
                      type="button"
                      onClick={() => setShowAdditional(true)}
                      className="font-display text-xs text-muted-foreground/50 transition-colors duration-300 hover:text-muted-foreground"
                    >
                      + Add more context
                    </button>
                  ) : (
                    <div className="relative pb-1 text-center" style={{ width: 320, maxWidth: "90vw" }}>
                      {/* hidden sizer to measure text width */}
                      <span
                        ref={additionalSizerRef}
                        className="invisible absolute whitespace-nowrap px-1 font-display text-sm"
                        aria-hidden="true"
                      >
                        {setupAdditional || "Anything else Vera should know..."}
                      </span>
                      <textarea
                        ref={additionalTextareaRef}
                        autoFocus
                        rows={1}
                        value={setupAdditional}
                        onChange={(e) => {
                          setSetupAdditional(e.target.value)
                          const el = e.target
                          el.style.height = "auto"
                          el.style.height = `${el.scrollHeight}px`
                        }}
                        onKeyDown={handleSetupKeyDown}
                        onBlur={() => { if (!filePickerOpenRef.current && !setupAdditional.trim() && !contextFile) setShowAdditional(false) }}
                        placeholder="Anything else Vera should know..."
                        className="w-full resize-none overflow-hidden bg-transparent text-center font-display text-sm text-foreground caret-primary placeholder:text-muted-foreground/40 focus:outline-none"
                        style={{ minHeight: "1.5rem" }}
                      />
                      <motion.span
                        className="absolute bottom-0 left-1/2 h-[2px] -translate-x-1/2 rounded-full bg-primary/30"
                        initial={false}
                        animate={{ width: additionalLineWidth > 0 ? additionalLineWidth + 20 : 0 }}
                        transition={{ duration: 0.35, ease: "easeInOut" }}
                      />
                    </div>
                  )}
                </div>
              </FadeIn>

              {/* Attach file — visible when additional context is expanded */}
              {showAdditional && onContextFileUpload && (
                <div className="mt-3 flex flex-col items-center gap-2">
                  {!contextFile ? (
                    <button
                      type="button"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => { filePickerOpenRef.current = true; contextFileInputRef.current?.click() }}
                      className="flex items-center gap-1 text-xs text-muted-foreground/40 transition-colors hover:text-muted-foreground/70"
                    >
                      <Paperclip className="h-3 w-3" />
                      Attach a file
                    </button>
                  ) : (
                    <div className="flex items-center gap-1.5 rounded-full border border-border/50 bg-muted/30 px-3 py-1 text-xs text-muted-foreground">
                      {isExtractingContext ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <FileText className="h-3 w-3" />
                      )}
                      <span className="max-w-[150px] truncate">{contextFile.name}</span>
                      {onContextFileRemove && (
                        <button
                          type="button"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={onContextFileRemove}
                          className="ml-0.5 rounded-full p-0.5 transition-colors hover:bg-muted"
                        >
                          <X className="h-2.5 w-2.5" />
                        </button>
                      )}
                    </div>
                  )}
                  <input
                    ref={contextFileInputRef}
                    type="file"
                    accept=".pdf,.txt,.docx,audio/*"
                    onChange={(e) => {
                      filePickerOpenRef.current = false
                      const file = e.target.files?.[0]
                      if (file) onContextFileUpload(file)
                      if (contextFileInputRef.current) contextFileInputRef.current.value = ""
                    }}
                    className="hidden"
                    aria-label="Attach reference file"
                  />
                </div>
              )}

              {/* Ready button */}
              <FadeIn delay={0.2}>
                <div className="mt-8 flex h-11 items-center justify-center">
                  <button
                    type="button"
                    onClick={handleSetupSubmit}
                    disabled={!hasSetupContent || !!isExtractingContext}
                    className={`flex items-center gap-2 rounded-full border px-6 py-2.5 text-sm font-medium transition-all duration-300 ease-out active:scale-[0.98] ${
                      hasSetupContent && !isExtractingContext
                        ? "border-primary/30 bg-primary/5 text-foreground opacity-100 hover:bg-primary/10 hover:border-primary/40"
                        : "pointer-events-none border-transparent bg-transparent text-transparent opacity-0"
                    }`}
                  >
                    <ArrowRight className="h-3.5 w-3.5 text-primary/70" />
                    Ready
                  </button>
                </div>
              </FadeIn>
            </motion.div>
          )}

          {setupPhase === "researching" && (
            <motion.div
              key="setup-researching"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="flex w-full flex-col items-center gap-6 py-8"
            >
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary/60" />
                <div className="text-center">
                  <p className="font-display text-lg text-foreground/80">Researching your audience</p>
                  <div className="mt-2 h-5 flex items-center justify-center">
                    <AnimatePresence mode="wait">
                      <motion.p
                        key={researchSnippet ?? "generating"}
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                        className="max-w-sm truncate text-sm italic text-muted-foreground/50"
                      >
                        {researchSnippet
                          ? `Searching "${researchSnippet}"`
                          : "Figuring out what to look up..."}
                      </motion.p>
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {setupPhase === "review" && researchMeta && (
            <motion.div
              key="setup-review"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="flex w-full flex-col items-center gap-6 py-4"
            >
              <div className="text-center">
                <p className="font-display text-lg text-foreground/80">Here&apos;s what I found</p>
                <p className="mt-1 text-sm text-muted-foreground/60">Review the research before we continue</p>
              </div>

              <div className="w-full max-w-lg">
                <ResearchCard meta={researchMeta} />
              </div>

              <button
                type="button"
                onClick={() => setSetupPhase("mode-select")}
                className="flex items-center gap-2 rounded-full border border-primary/30 bg-primary/5 px-6 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-primary/10 hover:border-primary/40 active:scale-[0.98]"
              >
                <ArrowRight className="h-3.5 w-3.5 text-primary/70" />
                Looks good
              </button>
            </motion.div>
          )}

          {setupPhase === "mode-select" && (
            <motion.div
              key="setup-mode-select"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="flex w-full flex-col items-center gap-6 py-8"
            >
              <div className="text-center">
                <p className="font-display text-lg text-foreground/80">How do you want to present?</p>
                <p className="mt-1 text-sm text-muted-foreground/60">Choose how you&apos;d like to deliver your presentation</p>
              </div>

              <button
                type="button"
                onClick={() => handleModeSelectInternal("present")}
                className="group flex items-center gap-3 rounded-full border border-primary/30 bg-primary/5 px-6 py-3 text-sm font-medium text-foreground transition-all hover:bg-primary/10 hover:border-primary/40 active:scale-[0.98]"
              >
                <Mic className="h-4 w-4 text-primary/70" />
                Present live
              </button>

              <div className="flex items-center gap-3 mt-2">
                <button
                  type="button"
                  onClick={() => handleModeSelectInternal("upload-recording")}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground/50 transition-colors hover:text-muted-foreground"
                >
                  <Upload className="h-3 w-3" />
                  Upload a recording
                </button>
              </div>
            </motion.div>
          )}

          {setupPhase === "uploading" && (
            <motion.div
              key="setup-uploading"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="flex w-full flex-col items-center gap-6 py-8"
            >
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary/60" />
                <div className="text-center">
                  <p className="font-display text-lg text-foreground/80">Processing your recording</p>
                  <p className="mt-1 text-sm text-muted-foreground/60">
                    {isCompressing
                      ? "Compressing audio..."
                      : isTranscribing
                      ? "Transcribing your recording..."
                      : "Wrapping up..."}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
})

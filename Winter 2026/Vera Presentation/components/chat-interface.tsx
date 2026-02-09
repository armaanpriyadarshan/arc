"use client"

import React from "react"

import { useState, useRef, useEffect, type FormEvent } from "react"
import { Send, Paperclip, FileVideo, FileAudio, Loader2 } from "lucide-react"

interface Attachment {
  name: string
  type: string
  size: number
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  attachment?: Attachment
}

const VERA_RESPONSES = [
  "Thanks for sharing that context. Who specifically will be in the room for this presentation? Understanding your audience will help me give you more targeted feedback.",
  "That's a great starting point. I'd recommend structuring your key message around the outcome your audience cares about most. What's the one thing you want them to walk away remembering?",
  "I've noted your setup. One thing I've seen with similar audiences is they tend to challenge the data early. Consider leading with your strongest evidence to build credibility from the start.",
  "Interesting approach. Based on what you've told me about your audience, I'd suggest softening the technical jargon in the first few minutes. You want to build rapport before diving into specifics.",
  "Good thinking. Let me simulate a likely reaction from your audience: they'll probably ask about ROI within the first five minutes. Do you have a clear, concise answer ready for that?",
]

const FILE_RESPONSE_INITIAL =
  "I've received your recording. Give me a moment to analyze it..."

const FILE_RESPONSE_FOLLOWUP =
  "I've analyzed your recording. Here are my initial observations from the perspective of your target audience:\n\n1. Your opening is strong — you establish credibility within the first 30 seconds.\n2. Around the 2-minute mark, the pacing slows down. Consider tightening that section to maintain engagement.\n3. Your closing call-to-action could be more direct. Try ending with a specific next step rather than an open-ended question.\n4. Overall tone is confident and professional. Well done."

let responseIndex = 0

function getNextResponse(): string {
  const response = VERA_RESPONSES[responseIndex % VERA_RESPONSES.length]
  responseIndex++
  return response
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 11)
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: generateId(),
      role: "assistant",
      content:
        "Welcome to Vera. I'm your AI presentation coach. Tell me about the presentation you're preparing for — who's your audience, what's the context, and what are you hoping to achieve?",
    },
  ])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isTyping])

  function addAssistantMessage(content: string) {
    setMessages((prev) => [
      ...prev,
      { id: generateId(), role: "assistant", content },
    ])
  }

  async function simulateResponse(isFileUpload: boolean) {
    setIsTyping(true)

    if (isFileUpload) {
      await new Promise((r) => setTimeout(r, 800))
      setIsTyping(false)
      addAssistantMessage(FILE_RESPONSE_INITIAL)

      setIsTyping(true)
      await new Promise((r) => setTimeout(r, 2000))
      setIsTyping(false)
      addAssistantMessage(FILE_RESPONSE_FOLLOWUP)
    } else {
      const delay = 500 + Math.random() * 500
      await new Promise((r) => setTimeout(r, delay))
      setIsTyping(false)
      addAssistantMessage(getNextResponse())
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return

    setMessages((prev) => [
      ...prev,
      { id: generateId(), role: "user", content: trimmed },
    ])
    setInput("")
    simulateResponse(false)
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    setMessages((prev) => [
      ...prev,
      {
        id: generateId(),
        role: "user",
        content: `Uploaded a file for review`,
        attachment: {
          name: file.name,
          type: file.type,
          size: file.size,
        },
      },
    ])

    simulateResponse(true)

    // Reset the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 md:px-0"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] md:max-w-[75%] ${
                  msg.role === "user"
                    ? "rounded-2xl rounded-br-md bg-primary px-5 py-3 text-primary-foreground"
                    : "rounded-2xl rounded-bl-md border border-border/60 bg-card px-5 py-3 text-card-foreground"
                }`}
              >
                {/* Attachment card */}
                {msg.attachment && (
                  <div className="mb-2 flex items-center gap-3 rounded-lg border border-primary-foreground/20 bg-primary-foreground/10 px-3 py-2">
                    {msg.attachment.type.startsWith("video") ? (
                      <FileVideo className="h-5 w-5 flex-shrink-0" />
                    ) : (
                      <FileAudio className="h-5 w-5 flex-shrink-0" />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">
                        {msg.attachment.name}
                      </p>
                      <p className="text-xs opacity-70">
                        {formatFileSize(msg.attachment.size)}
                      </p>
                    </div>
                  </div>
                )}

                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </p>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-md border border-border/60 bg-card px-5 py-3">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:0ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:150ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 border-t border-border/50 bg-background/80 px-4 py-4 backdrop-blur-sm">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-3"
        >
          {/* File upload */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,audio/*"
            onChange={handleFileUpload}
            className="hidden"
            aria-label="Upload video or audio file"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isTyping}
            className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
            aria-label="Attach a file"
          >
            <Paperclip className="h-4 w-4" />
          </button>

          {/* Text input */}
          <div className="flex-1">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              placeholder="Type your message..."
              rows={1}
              disabled={isTyping}
              className="w-full resize-none rounded-lg border border-input bg-card px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
            />
          </div>

          {/* Send button */}
          <button
            type="submit"
            disabled={!input.trim() || isTyping}
            className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-md shadow-primary/20 transition-all hover:bg-primary/90 disabled:opacity-50 disabled:shadow-none"
            aria-label="Send message"
          >
            {isTyping ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </form>
      </div>
    </div>
  )
}

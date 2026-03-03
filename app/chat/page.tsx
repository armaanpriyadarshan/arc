"use client"

import { Suspense, useState, useEffect, useCallback } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Clock } from "lucide-react"
import { toast } from "sonner"
import { useAuth } from "@/contexts/auth-context"
import { ChatNavbar } from "@/components/chat-navbar"
import { CoachingInterface } from "@/components/coaching-interface"
import { SessionHistorySidebar } from "@/components/session-history-sidebar"
import { useSessionHistory } from "@/hooks/use-session-history"

function ChatContent() {
  const router = useRouter()
  const { user, loading, plan, refreshSubscription } = useAuth()
  const [idToken, setIdToken] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const searchParams = useSearchParams()
  const getAuthToken = useCallback(async () => {
    if (!user) return null
    return user.getIdToken()
  }, [user])

  const { sessions, loading: sessionsLoading, error: sessionsError, refresh: refreshSessions, removeSession } =
    useSessionHistory({ userId: user?.uid ?? null, getAuthToken })

  const handleHistoryToggle = useCallback(() => {
    if (!historyOpen) refreshSessions()
    setHistoryOpen((prev) => !prev)
  }, [historyOpen, refreshSessions])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await removeSession(sessionId)
    } catch {
      toast.error("Failed to delete presentation")
    }
  }, [removeSession])

  useEffect(() => {
    if (user) {
      user.getIdToken().then(setIdToken)
    } else {
      setIdToken(null)
    }
  }, [user])

  // Login wall temporarily disabled — all users get full access

  // Handle post-checkout success: verify the session server-side, then refresh local state
  useEffect(() => {
    if (loading) return // wait for auth to resolve before verifying
    if (searchParams.get("checkout") !== "success") return
    const sessionId = searchParams.get("session_id")

    async function verifyCheckout() {
      if (user && sessionId) {
        try {
          const token = await user.getIdToken()
          const res = await fetch("/api/verify-checkout", {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ session_id: sessionId }),
          })

          if (!res.ok) {
            console.error("Checkout verification failed:", await res.text())
          }
        } catch (err) {
          console.error("Checkout verification error:", err)
        }
      }

      await refreshSubscription()
      toast.success("Welcome to Pro! You now have unlimited access.")
      window.history.replaceState({}, "", "/chat")
    }

    verifyCheckout()
  }, [searchParams, refreshSubscription, user, loading])

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col">
      <ChatNavbar plan={plan} isAuthenticated={!!user} />
      <div className="relative flex min-h-0 flex-1 flex-col">
        {user && (
          <SessionHistorySidebar
            open={historyOpen}
            onClose={() => setHistoryOpen(false)}
            sessions={sessions}
            loading={sessionsLoading}
            error={sessionsError}
            onDelete={handleDeleteSession}
          />
        )}
        <CoachingInterface authToken={idToken} />
      </div>
      {/* Floating history button — bottom left, only for logged-in users */}
      {user && (
        <button
          type="button"
          onClick={handleHistoryToggle}
          className={`fixed bottom-6 left-6 z-30 flex h-10 w-10 items-center justify-center rounded-full border shadow-lg backdrop-blur-sm transition-all active:scale-[0.98] ${
            historyOpen
              ? "border-primary/20 bg-primary/[0.03] text-foreground"
              : "border-border/40 bg-background/90 text-muted-foreground/50 hover:border-primary/20 hover:bg-primary/[0.03] hover:text-primary/70"
          }`}
          aria-label="Session history"
          aria-pressed={historyOpen}
        >
          <Clock className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}

export default function ChatPage() {
  return (
    <Suspense fallback={
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    }>
      <ChatContent />
    </Suspense>
  )
}

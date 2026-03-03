"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { useAuth } from "@/contexts/auth-context"
import { SetupWizard } from "@/components/setup-wizard"
import { About } from "@/components/about"
import { HowItWorks } from "@/components/how-it-works"
import { Stats } from "@/components/stats"
import { Footer } from "@/components/footer"

export default function Page() {
  const router = useRouter()
  const { user, loading } = useAuth()

  // Authenticated users go straight to /chat
  useEffect(() => {
    if (!loading && user) {
      router.replace("/chat")
    }
  }, [user, loading, router])

  if (loading || user) {
    return (
      <div className="flex h-dvh items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  return (
    <>
      {/* First screen: nav + setup wizard fill exactly one viewport */}
      <div className="flex h-dvh flex-col">
        <nav className="flex-shrink-0 border-b border-border/50 bg-background/70 backdrop-blur-xl">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
            <span className="font-display text-xl font-bold tracking-tight text-foreground">
              Vera
            </span>
            <Link
              href="/login"
              className="text-sm font-medium text-foreground transition-colors hover:text-foreground/80"
            >
              Sign in
            </Link>
          </div>
        </nav>

        <div className="relative flex min-h-0 flex-1 flex-col">
          <SetupWizard
            isResearching={false}
            researchMeta={null}
            researchSearchTerms={null}
            isCompressing={false}
            isTranscribing={false}
            onResearchStart={() => router.push("/chat")}
            onModeSelect={() => router.push("/chat")}
            onReady={(fields) => {
              sessionStorage.setItem("vera_setup_fields", JSON.stringify(fields))
              router.push("/chat")
            }}
          />
        </div>
      </div>

      {/* Landing content below the fold */}
      <About />
      <HowItWorks />
      <Stats />
      <Footer />
    </>
  )
}

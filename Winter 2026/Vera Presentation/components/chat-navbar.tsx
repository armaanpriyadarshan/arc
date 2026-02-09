"use client"

import { useRouter } from "next/navigation"

export function ChatNavbar() {
  const router = useRouter()

  return (
    <nav className="flex-shrink-0 border-b border-border/50 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="/" className="text-xl font-bold tracking-tight text-foreground">
          Vera
        </a>
        <button
          type="button"
          onClick={() => router.push("/")}
          className="text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          Logout
        </button>
      </div>
    </nav>
  )
}

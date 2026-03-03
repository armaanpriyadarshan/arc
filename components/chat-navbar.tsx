"use client"

import Link from "next/link"

interface ChatNavbarProps {
  plan?: 'free' | 'pro' | null
  isAuthenticated?: boolean
}

export function ChatNavbar({ plan, isAuthenticated }: ChatNavbarProps) {
  return (
    <nav className="flex-shrink-0 border-b border-border/50 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="/" className="font-display text-xl font-bold tracking-tight text-foreground">
          Vera
        </a>
        {isAuthenticated ? (
          <div className="flex items-center gap-4">
            {plan === 'pro' ? (
              <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-semibold text-primary">
                Pro
              </span>
            ) : (
              <Link
                href="/plans"
                className="text-sm font-medium text-primary transition-colors hover:text-primary/80"
              >
                Upgrade
              </Link>
            )}
            <Link
              href="/account"
              className="text-sm font-medium text-foreground transition-colors hover:text-foreground/80"
            >
              Account
            </Link>
          </div>
        ) : (
          <Link
            href="/login"
            className="text-sm font-medium text-foreground transition-colors hover:text-foreground/80"
          >
            Sign in
          </Link>
        )}
      </div>
    </nav>
  )
}

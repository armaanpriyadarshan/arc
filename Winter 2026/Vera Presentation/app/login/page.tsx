"use client"

import { useState, type FormEvent } from "react"
import { useRouter } from "next/navigation"

export default function LoginPage() {
  const router = useRouter()
  const [isSignUp, setIsSignUp] = useState(false)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    router.push("/chat")
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center px-6">
      {/* Same soft gradient as hero */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
        style={{
          background:
            "linear-gradient(135deg, hsl(200 40% 95%) 0%, hsl(165 35% 95%) 50%, hsl(190 30% 96%) 100%)",
        }}
      />

      <div className="w-full max-w-md">
        {/* Brand */}
        <div className="mb-10 text-center">
          <a
            href="/"
            className="text-2xl font-bold tracking-tight text-foreground"
          >
            Vera
          </a>
          <p className="mt-2 text-sm text-muted-foreground">
            {isSignUp
              ? "Create your account to get started"
              : "Sign in to your rehearsal room"}
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-border/60 bg-card/80 p-8 shadow-lg backdrop-blur-sm">
          <h2 className="text-xl font-semibold text-foreground">
            {isSignUp ? "Sign Up" : "Log In"}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {isSignUp
              ? "Fill in your details below"
              : "Enter your credentials to continue"}
          </p>

          <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-4">
            {isSignUp && (
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="name"
                  className="text-sm font-medium text-foreground"
                >
                  Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  className="rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>
            )}

            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="email"
                className="text-sm font-medium text-foreground"
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="password"
                className="text-sm font-medium text-foreground"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <button
              type="submit"
              className="mt-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              {isSignUp ? "Create Account" : "Log In"}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              {isSignUp
                ? "Already have an account? Log in"
                : "Don't have an account? Sign up"}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

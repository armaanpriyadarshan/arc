export function Hero() {
  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 pt-20">
      {/* Soft gradient background wash */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
        style={{
          background:
            "linear-gradient(135deg, hsl(200 40% 95%) 0%, hsl(165 35% 95%) 50%, hsl(190 30% 96%) 100%)",
        }}
      />

      <div className="mx-auto max-w-3xl text-center">
        <p className="mb-4 text-sm font-medium uppercase tracking-widest text-primary">
          AI-Powered Presentation Coaching
        </p>
        <h1 className="text-balance text-5xl font-bold leading-tight tracking-tight text-foreground md:text-7xl">
          Rehearse with your
          <br />
          <span className="text-primary">real audience</span>
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground md:text-xl">
          Think of it as a rehearsal room powered by AI. Describe who you are presenting to, and
          Vera simulates that specific audience to give you feedback that actually matters.
        </p>
        <div className="mt-10">
          <a
            href="/login"
            className="inline-flex items-center rounded-lg bg-primary px-8 py-3.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            Get Started
          </a>
        </div>
      </div>
    </section>
  )
}

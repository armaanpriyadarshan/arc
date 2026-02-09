export function About() {
  return (
    <section id="about" className="px-6 py-24 md:py-32">
      <div className="mx-auto max-w-4xl">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-balance text-3xl font-bold tracking-tight text-foreground md:text-5xl">
            Not just any another generic coach
          </h2>
          <p className="mt-6 text-pretty text-lg leading-relaxed text-muted-foreground">
            Traditional presentation coaching gives you surface-level advice: speak slower, make eye
            contact, use fewer slides. Vera is fundamentally different.
          </p>
        </div>

        <div className="mt-16 grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-border/60 bg-card/60 p-8 backdrop-blur-sm">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <span className="text-sm font-bold text-muted-foreground">01</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground">Audience-Specific Feedback</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Vera doesn&apos;t just evaluate your delivery. It evaluates your message from the
              perspective of the exact people you&apos;ll be presenting to — whether that&apos;s Nike
              executives, a technical review board, or a VC firm.
            </p>
          </div>

          <div className="rounded-2xl border border-border/60 bg-card/60 p-8 backdrop-blur-sm">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <span className="text-sm font-bold text-muted-foreground">02</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground">Context-Aware Intelligence</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Our AI understands industry dynamics, organizational priorities, and stakeholder
              psychology. The feedback you get feels like it came from someone who was actually in
              the room.
            </p>
          </div>

          <div className="rounded-2xl border border-border/60 bg-card/60 p-8 backdrop-blur-sm">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <span className="text-sm font-bold text-muted-foreground">03</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground">Personalized, Not Generic</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              No more &quot;speak more slowly&quot; or &quot;add a summary slide.&quot; Vera tells you which arguments
              will land, which claims need evidence, and what questions your audience is likely to
              ask.
            </p>
          </div>

          <div className="rounded-2xl border border-border/60 bg-card/60 p-8 backdrop-blur-sm">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <span className="text-sm font-bold text-muted-foreground">04</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground">Built for High Stakes</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Whether you are pitching to investors, presenting a quarterly review, or defending a
              thesis, Vera helps you prepare for the moments that matter most in your career.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

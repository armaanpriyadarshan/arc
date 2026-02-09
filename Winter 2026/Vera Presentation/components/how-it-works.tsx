const steps = [
  {
    number: "1",
    title: "Describe your scenario",
    description:
      "Tell Vera who you are presenting to. A board of directors? A prospective client? A hiring committee? The more context you give, the sharper the feedback.",
  },
  {
    number: "2",
    title: "Upload or record your presentation",
    description:
      "Upload a video, audio recording, or slide deck of your presentation. Vera analyzes your content, structure, tone, and delivery.",
  },
  {
    number: "3",
    title: "Get simulated audience feedback",
    description:
      "Receive detailed, perspective-driven feedback as if your target audience just watched you present. Know exactly what landed and what needs work.",
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="relative px-6 py-24 md:py-32">
      {/* Subtle gradient background */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
        style={{
          background:
            "linear-gradient(180deg, hsl(190 30% 96%) 0%, hsl(170 30% 96%) 50%, hsl(200 30% 97%) 100%)",
        }}
      />

      <div className="mx-auto max-w-4xl">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-balance text-3xl font-bold tracking-tight text-foreground md:text-5xl">
            How it works
          </h2>
          <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
            Three simple steps to presentation confidence.
          </p>
        </div>

        <div className="mt-16 grid gap-6 md:grid-cols-3">
          {steps.map((step) => (
            <div key={step.number} className="relative flex flex-col items-start rounded-2xl border border-border/60 bg-card/70 p-8 backdrop-blur-sm">
              <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-lg font-bold text-primary-foreground">
                {step.number}
              </div>
              <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

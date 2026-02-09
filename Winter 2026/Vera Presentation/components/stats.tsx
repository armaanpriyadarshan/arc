const stats = [
  {
    value: "10+",
    label: "Audience Personas",
    description: "Simulated audience archetypes covering industries, roles, and temperaments",
  },
  {
    value: "92%",
    label: "Felt More Prepared",
    description: "Of users reported feeling significantly more confident after using Vera",
  },
  {
    value: "500+",
    label: "Presentations Coached",
    description: "High-stakes presentations refined through AI-driven audience simulation",
  },
  {
    value: "4.8/5",
    label: "Average Rating",
    description: "Rated by professionals across consulting, tech, finance, and academia",
  },
]

export function Stats() {
  return (
    <section className="px-6 py-24 md:py-32">
      <div className="mx-auto max-w-4xl">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-balance text-3xl font-bold tracking-tight text-foreground md:text-5xl">
            Trusted by ambitious professionals
          </h2>
          <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
            The numbers speak for themselves.
          </p>
        </div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="flex flex-col rounded-2xl border border-border/60 bg-card/60 p-8 text-center backdrop-blur-sm"
            >
              <span className="text-4xl font-bold tracking-tight text-primary">{stat.value}</span>
              <span className="mt-2 text-sm font-semibold text-foreground">{stat.label}</span>
              <span className="mt-2 text-xs leading-relaxed text-muted-foreground">
                {stat.description}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

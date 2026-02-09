export function Footer() {
  return (
    <footer className="border-t border-border/50 px-6 py-12">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-6 md:flex-row">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-foreground">Vera</span>
          <span className="text-sm text-muted-foreground">
            — A rehearsal room powered by AI.
          </span>
        </div>

        <nav aria-label="Footer navigation">
          <ul className="flex flex-wrap items-center justify-center gap-6">
            <li>
              <a href="#" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                Privacy
              </a>
            </li>
            <li>
              <a href="#" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                Terms
              </a>
            </li>
            <li>
              <a href="#" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                Contact
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </footer>
  )
}

import { Navbar } from "@/components/navbar"
import { Hero } from "@/components/hero"
import { About } from "@/components/about"
import { HowItWorks } from "@/components/how-it-works"
import { Stats } from "@/components/stats"
import { Footer } from "@/components/footer"

export default function Page() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <About />
        <HowItWorks />
        <Stats />
      </main>
      <Footer />
    </>
  )
}

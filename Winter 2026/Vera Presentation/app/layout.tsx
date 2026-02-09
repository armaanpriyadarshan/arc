import React from "react"
import type { Metadata, Viewport } from 'next'

import './globals.css'

export const metadata: Metadata = {
  title: 'Vera — Rehearse with Your Real Audience, Powered by AI',
  description:
    'Vera simulates your target audience and gives detailed, personalized feedback on your presentations. Think of it as a rehearsal room powered by AI.',
}

export const viewport: Viewport = {
  themeColor: '#0891b2',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}

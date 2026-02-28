"use client"

import type * as PdfJsLib from 'pdfjs-dist'

let pdfjsModule: typeof PdfJsLib | null = null

async function getPdfJs(): Promise<typeof PdfJsLib> {
  if (!pdfjsModule) {
    const pdfjs = await import('pdfjs-dist')
    pdfjs.GlobalWorkerOptions.workerSrc = new URL(
      'pdfjs-dist/build/pdf.worker.min.mjs',
      import.meta.url
    ).href
    pdfjsModule = pdfjs
  }
  return pdfjsModule
}

export async function renderPdfThumbnails(
  pdfUrl: string,
  maxPages = 40
): Promise<Record<number, string>> {
  const pdfjs = await getPdfJs()
  const loadingTask = pdfjs.getDocument({ url: pdfUrl })
  const pdf = await loadingTask.promise

  const numPages = Math.min(pdf.numPages, maxPages)
  const thumbnails: Record<number, string> = {}

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return thumbnails

  for (let pageNum = 1; pageNum <= numPages; pageNum++) {
    try {
      const page = await pdf.getPage(pageNum)
      // Scale so the longest dimension is ~800px
      const unscaled = page.getViewport({ scale: 1 })
      const scale = 800 / Math.max(unscaled.width, unscaled.height)
      const viewport = page.getViewport({ scale })

      canvas.width = viewport.width
      canvas.height = viewport.height
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      await page.render({ canvasContext: ctx, viewport }).promise
      thumbnails[pageNum] = canvas.toDataURL('image/jpeg', 0.85)
    } catch {
      // Skip pages that fail to render
    }
  }

  return thumbnails
}

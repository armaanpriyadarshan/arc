import fs from 'fs/promises'
import path from 'path'
import { extractText } from 'unpdf'
import JSZip from 'jszip'
import { tempPath } from '@/backend/audio'

export interface SlideText {
  slideNumber: number
  text: string
}

export interface SlideFeedback {
  slideNumber: number
  title: string
  rating: 'strong' | 'needs-work' | 'critical'
  headline: string
  strengths: string[]
  improvements: string[]
  quote?: string
}

export interface DeckFeedback {
  deckTitle: string
  audienceAssumed: string
  overallRating: number
  executiveSummary: string
  slides: SlideFeedback[]
  topPriorities: string[]
}

export const MAX_SLIDES = 40

export function slidesTempPath(ext: string): string {
  return tempPath(ext, 'vera-slides')
}

/**
 * Extract text content from each page of a PDF.
 * Uses unpdf which works in serverless Node.js (no browser APIs needed).
 * Caps at MAX_SLIDES pages.
 */
export async function extractSlideTexts(pdfPath: string): Promise<SlideText[]> {
  const pdfBuffer = await fs.readFile(pdfPath)
  const { totalPages, text: pages } = await extractText(
    new Uint8Array(pdfBuffer),
    { mergePages: false }
  )

  const pageCount = Math.min(totalPages, MAX_SLIDES)
  return pages.slice(0, pageCount).map((text, i) => ({
    slideNumber: i + 1,
    text: text.trim() || '[No text content on this slide]',
  }))
}

/**
 * Extract plain text from `<a:t>` elements in a PPTX slide XML.
 * Groups text by paragraph (`<a:p>`) boundaries.
 */
function extractTextFromSlideXml(xml: string): string {
  const paragraphs: string[] = []
  // Match each <a:p>...</a:p> paragraph block
  const pRegex = /<a:p\b[^>]*>([\s\S]*?)<\/a:p>/g
  let pMatch: RegExpExecArray | null
  while ((pMatch = pRegex.exec(xml)) !== null) {
    const pContent = pMatch[1]
    // Extract all <a:t>...</a:t> text runs within this paragraph
    const runs: string[] = []
    const tRegex = /<a:t>([\s\S]*?)<\/a:t>/g
    let tMatch: RegExpExecArray | null
    while ((tMatch = tRegex.exec(pContent)) !== null) {
      runs.push(tMatch[1])
    }
    if (runs.length > 0) {
      paragraphs.push(runs.join(''))
    }
  }
  return paragraphs.join('\n')
}

/**
 * Extract text content from each slide of a PPTX file.
 * PPTX files are ZIP archives with slide XML at ppt/slides/slideN.xml.
 * Caps at MAX_SLIDES slides.
 */
export async function extractSlideTextsFromPptx(pptxPath: string): Promise<SlideText[]> {
  const buffer = await fs.readFile(pptxPath)
  const zip = await JSZip.loadAsync(buffer)

  // Collect slide entries and sort by slide number
  const slideEntries: { num: number; entry: JSZip.JSZipObject }[] = []
  zip.forEach((relativePath, entry) => {
    const match = relativePath.match(/^ppt\/slides\/slide(\d+)\.xml$/)
    if (match) {
      slideEntries.push({ num: parseInt(match[1], 10), entry })
    }
  })
  slideEntries.sort((a, b) => a.num - b.num)

  const capped = slideEntries.slice(0, MAX_SLIDES)
  const results: SlideText[] = []

  for (let i = 0; i < capped.length; i++) {
    const xml = await capped[i].entry.async('string')
    const text = extractTextFromSlideXml(xml)
    results.push({
      slideNumber: i + 1,
      text: text.trim() || '[No text content on this slide]',
    })
  }

  return results
}

/**
 * Detect file type and extract slide texts from PDF or PPTX.
 */
export async function extractSlideTextsAuto(filePath: string): Promise<SlideText[]> {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.pptx') {
    return extractSlideTextsFromPptx(filePath)
  }
  return extractSlideTexts(filePath)
}

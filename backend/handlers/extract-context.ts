import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'
import { del } from '@vercel/blob'
import { extractText } from 'unpdf'
import mammoth from 'mammoth'
import { openai } from '@/backend/openai'
import { extractContextRequestSchema } from '@/backend/validation'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import { downloadToTmp, processFileForWhisper, cleanupTempFiles } from '@/backend/audio'

const MAX_CONTEXT_CHARS = 50_000

const AUDIO_EXTENSIONS = new Set([
  '.mp3', '.mp4', '.m4a', '.wav', '.mpeg', '.mpga', '.webm',
  '.ogg', '.flac', '.aac', '.wma', '.mov', '.avi', '.mkv',
  '.m4v', '.opus', '.3gp',
])

function getFileType(fileName: string): 'pdf' | 'txt' | 'docx' | 'audio' {
  const ext = path.extname(fileName).toLowerCase()
  if (ext === '.pdf') return 'pdf'
  if (ext === '.txt') return 'txt'
  if (ext === '.docx') return 'docx'
  if (AUDIO_EXTENSIONS.has(ext)) return 'audio'
  return 'txt' // fallback
}

async function extractFromPdf(tmpPath: string): Promise<string> {
  const pdfBuffer = await fs.readFile(tmpPath)
  const { text: pages } = await extractText(
    new Uint8Array(pdfBuffer),
    { mergePages: false }
  )
  const text = (pages as string[]).join('\n\n').trim()
  if (!text) {
    throw new Error("This PDF doesn't contain extractable text. It may be image-only.")
  }
  return text
}

async function extractFromTxt(tmpPath: string): Promise<string> {
  return fs.readFile(tmpPath, 'utf-8')
}

async function extractFromDocx(tmpPath: string): Promise<string> {
  const result = await mammoth.extractRawText({ path: tmpPath })
  return result.value
}

async function extractFromAudio(tmpPath: string): Promise<string> {
  const { chunkPaths, allTempPaths } = await processFileForWhisper(tmpPath)

  try {
    const client = openai()
    const MAX_CONCURRENCY = 3

    async function transcribeWithConcurrencyLimit(
      paths: string[],
      limit: number
    ): Promise<string[]> {
      const results: string[] = new Array(paths.length)
      let nextIndex = 0

      async function worker() {
        while (nextIndex < paths.length) {
          const i = nextIndex++
          const fileStream = (await import('fs')).createReadStream(paths[i])
          const transcription = await client.audio.transcriptions.create({
            model: 'gpt-4o-mini-transcribe',
            file: fileStream,
          })
          results[i] = transcription.text
        }
      }

      const workers = Array.from(
        { length: Math.min(limit, paths.length) },
        () => worker()
      )
      await Promise.all(workers)
      return results
    }

    const transcriptParts = await transcribeWithConcurrencyLimit(chunkPaths, MAX_CONCURRENCY)
    return transcriptParts.join(' ')
  } finally {
    // Clean up audio temp files (excluding the input which is cleaned up by the caller)
    const extraPaths = allTempPaths.filter(p => p !== tmpPath)
    if (extraPaths.length > 0) {
      await cleanupTempFiles(extraPaths)
    }
  }
}

export async function handleExtractContext(request: NextRequest) {
  const ip = getClientIp(request)
  if (!checkRateLimit('context:' + ip, RATE_LIMITS.contextExtract.limit, RATE_LIMITS.contextExtract.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait before extracting another file.' },
      { status: 429 }
    )
  }

  let tmpPath: string | undefined
  let blobUrl: string | undefined

  try {
    const body = await request.json()
    const parsed = extractContextRequestSchema.safeParse(body)

    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid request. Provide blobUrl and fileName.' },
        { status: 400 }
      )
    }

    blobUrl = parsed.data.blobUrl
    const { fileName } = parsed.data

    tmpPath = await downloadToTmp(blobUrl, fileName)
    const fileType = getFileType(fileName)

    let text: string
    switch (fileType) {
      case 'pdf':
        text = await extractFromPdf(tmpPath)
        break
      case 'txt':
        text = await extractFromTxt(tmpPath)
        break
      case 'docx':
        text = await extractFromDocx(tmpPath)
        break
      case 'audio':
        text = await extractFromAudio(tmpPath)
        break
    }

    // Truncate to prevent prompt bloat
    if (text.length > MAX_CONTEXT_CHARS) {
      text = text.slice(0, MAX_CONTEXT_CHARS)
    }

    return NextResponse.json({ text })
  } catch (error: unknown) {
    console.error('Context extraction error:', error)

    const message =
      error instanceof Error && (
        error.message.includes("doesn't contain extractable text") ||
        error.message.includes('does not contain an audio track')
      )
        ? error.message
        : 'Failed to extract text from file. Please try again.'

    return NextResponse.json(
      { error: message },
      { status: 500 }
    )
  } finally {
    if (tmpPath) {
      await cleanupTempFiles([tmpPath])
    }
    if (blobUrl) {
      await del(blobUrl).catch(() => {
        // Best-effort cleanup — blob TTL will handle it if this fails
      })
    }
  }
}

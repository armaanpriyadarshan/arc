import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import { del } from '@vercel/blob'
import { openai } from '@/backend/openai'
import { transcribeRequestSchema } from '@/backend/validation'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'
import { downloadToTmp, processFileForWhisper, cleanupTempFiles, analyzeAudio } from '@/backend/audio'
import type { ChunkInfo } from '@/backend/audio'
import { computeDeliveryAnalytics } from '@/backend/delivery-analytics'
import type { TimestampedWord } from '@/lib/delivery-analytics'

export async function handleTranscribe(request: NextRequest) {
  const ip = getClientIp(request)
  if (!checkRateLimit(ip, RATE_LIMITS.transcribe.limit, RATE_LIMITS.transcribe.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait before uploading again.' },
      { status: 429 }
    )
  }

  let tempPaths: string[] = []
  let blobUrl: string | undefined

  try {
    const body = await request.json()
    const parsed = transcribeRequestSchema.safeParse(body)

    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid request. Provide blobUrl and fileName.' },
        { status: 400 }
      )
    }

    blobUrl = parsed.data.blobUrl
    const { fileName } = parsed.data

    // Download from blob storage to a temp file
    const inputPath = await downloadToTmp(blobUrl, fileName)

    // Process file: extract audio, compress, split if needed
    const { chunks, allTempPaths, analysisPath } = await processFileForWhisper(inputPath)
    tempPaths = allTempPaths

    // Transcribe chunks and analyze energy in parallel
    const client = openai()
    const MAX_CONCURRENCY = 3

    interface ChunkResult {
      text: string
      words: TimestampedWord[]
    }

    async function transcribeWithConcurrencyLimit(
      chunkInfos: ChunkInfo[],
      limit: number
    ): Promise<ChunkResult[]> {
      const results: ChunkResult[] = new Array(chunkInfos.length)
      let nextIndex = 0

      async function worker() {
        while (nextIndex < chunkInfos.length) {
          const i = nextIndex++
          const chunk = chunkInfos[i]
          const fileStream = fs.createReadStream(chunk.path)
          const transcription = await client.audio.transcriptions.create({
            model: 'whisper-1',
            file: fileStream,
            response_format: 'verbose_json',
            timestamp_granularities: ['word'],
          })

          // Cast to verbose response type for word timestamps
          const verbose = transcription as unknown as {
            text: string
            words?: { word: string; start: number; end: number }[]
          }

          const words: TimestampedWord[] = (verbose.words ?? []).map((w) => ({
            word: w.word,
            start: w.start + chunk.offsetSeconds,
            end: w.end + chunk.offsetSeconds,
          }))

          results[i] = { text: verbose.text, words }
        }
      }

      const workers = Array.from(
        { length: Math.min(limit, chunkInfos.length) },
        () => worker()
      )
      await Promise.all(workers)
      return results
    }

    const [chunkResults, audioAnalysis] = await Promise.all([
      transcribeWithConcurrencyLimit(chunks, MAX_CONCURRENCY),
      analyzeAudio(analysisPath).catch((err) => {
        console.warn('[transcribe] Audio analysis failed, continuing without:', err)
        return { energyWindows: [], pitchWindows: [] } as Awaited<ReturnType<typeof analyzeAudio>>
      }),
    ])

    const transcript = chunkResults.map((r) => r.text).join(' ')
    const allWords = chunkResults.flatMap((r) => r.words)

    const analytics = computeDeliveryAnalytics(allWords, audioAnalysis.energyWindows, audioAnalysis.pitchWindows)

    return NextResponse.json({ transcript, analytics })
  } catch (error: unknown) {
    console.error('Transcription error:', error)

    // Surface user-friendly messages (e.g. "no audio track") instead of generic error
    const message =
      error instanceof Error && error.message.includes('does not contain an audio track')
        ? error.message
        : 'Failed to transcribe file. Please try again.'

    return NextResponse.json(
      { error: message },
      { status: 500 }
    )
  } finally {
    // Always clean up temp files
    if (tempPaths.length > 0) {
      await cleanupTempFiles(tempPaths)
    }
    // Delete the blob from Vercel Blob storage
    if (blobUrl) {
      await del(blobUrl).catch(() => {
        // Best-effort cleanup — blob TTL will handle it if this fails
      })
    }
  }
}

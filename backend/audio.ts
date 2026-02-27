import ffmpeg from 'fluent-ffmpeg'
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg'
import fs from 'fs/promises'
import { existsSync, statSync } from 'fs'
import path from 'path'
import os from 'os'
import crypto from 'crypto'

ffmpeg.setFfmpegPath(ffmpegInstaller.path)

const WHISPER_MAX_SIZE = 25 * 1024 * 1024 // 25MB
const MAX_CHUNK_DURATION = 1400 // seconds — Whisper API limit is 1500s

export interface ChunkInfo {
  path: string
  offsetSeconds: number
}

/** Formats Whisper accepts natively — no FFmpeg conversion needed */
const WHISPER_NATIVE_FORMATS = new Set([
  '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm',
])

export function tempPath(ext: string, prefix = 'vera'): string {
  const id = crypto.randomBytes(8).toString('hex')
  return path.join(os.tmpdir(), `${prefix}-${id}${ext}`)
}

/**
 * Extract and compress audio from a video or audio file to 64kbps mono mp3.
 * This dramatically reduces file size — a 200MB video typically yields ~10MB of audio.
 */
export function extractAndCompressAudio(
  inputPath: string,
  outputPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .noVideo()
      .audioCodec('libmp3lame')
      .audioBitrate('64k')
      .audioChannels(1)
      .audioFrequency(16000)
      .format('mp3')
      .on('error', (err: Error) => reject(err))
      .on('end', () => resolve())
      .save(outputPath)
  })
}

/**
 * Probe a file and return its ffprobe metadata.
 */
function probeFile(filePath: string): Promise<ffmpeg.FfprobeData> {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) return reject(err)
      resolve(metadata)
    })
  })
}

/**
 * Check whether a file contains at least one audio stream.
 */
async function hasAudioStream(filePath: string): Promise<boolean> {
  const metadata = await probeFile(filePath)
  return metadata.streams.some((s) => s.codec_type === 'audio')
}

/**
 * Get the duration of an audio file in seconds.
 */
async function getAudioDuration(filePath: string): Promise<number> {
  const metadata = await probeFile(filePath)
  return metadata.format.duration ?? 0
}

/**
 * Split an audio file into chunks that each fit under the Whisper API size limit.
 * Returns an array of file paths (in order). If the file is already small enough,
 * returns a single-element array with the original path.
 */
export async function splitAudioIfNeeded(
  filePath: string,
  maxSizeBytes: number = WHISPER_MAX_SIZE
): Promise<ChunkInfo[]> {
  const stat = statSync(filePath)
  const estimatedDuration = await getAudioDuration(filePath)

  const sizeChunks = Math.ceil(stat.size / maxSizeBytes)
  const durationChunks = Math.ceil(estimatedDuration / MAX_CHUNK_DURATION)
  const numChunks = Math.max(sizeChunks, durationChunks)

  if (numChunks <= 1) {
    return [{ path: filePath, offsetSeconds: 0 }]
  }

  const chunkDuration = Math.floor(estimatedDuration / numChunks)

  const chunks: ChunkInfo[] = []

  for (let i = 0; i < numChunks; i++) {
    const startTime = i * chunkDuration
    const chunkPath = tempPath(`-chunk${i}.mp3`)
    chunks.push({ path: chunkPath, offsetSeconds: startTime })

    await new Promise<void>((resolve, reject) => {
      let cmd = ffmpeg(filePath)
        .setStartTime(startTime)
        .audioCodec('libmp3lame')
        .audioBitrate('64k')
        .audioChannels(1)
        .audioFrequency(16000)
        .format('mp3')

      // For all chunks except the last, set duration
      if (i < numChunks - 1) {
        cmd = cmd.setDuration(chunkDuration)
      }

      cmd
        .on('error', (err: Error) => reject(err))
        .on('end', () => resolve())
        .save(chunkPath)
    })
  }

  return chunks
}

/**
 * Download a file from a URL to a temp path on disk.
 * Retries with exponential backoff to handle CDN propagation delays after Vercel Blob uploads.
 */
export async function downloadToTmp(url: string, fileName: string): Promise<string> {
  const ext = path.extname(fileName) || '.bin'
  const tmpFile = tempPath(ext)

  // Verify blob exists via API (bypasses CDN) before attempting CDN fetch
  try {
    const { head } = await import('@vercel/blob')
    const blobMeta = await head(url)
    console.log('[downloadToTmp] Blob exists in store:', blobMeta.size, 'bytes, url:', url)
  } catch (headErr) {
    console.error('[downloadToTmp] Blob NOT found via head() API:', url, headErr)
    throw new Error(`Blob does not exist in store: ${url}`)
  }

  const MAX_RETRIES = 3
  const INITIAL_DELAY_MS = 500

  let lastStatus = 0
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      const delay = INITIAL_DELAY_MS * Math.pow(2, attempt - 1)
      console.warn(`[downloadToTmp] Retry ${attempt}/${MAX_RETRIES} after ${delay}ms (last status: ${lastStatus})`)
      await new Promise((r) => setTimeout(r, delay))
    }
    const response = await fetch(url)
    if (response.ok) {
      const buffer = Buffer.from(await response.arrayBuffer())
      await fs.writeFile(tmpFile, buffer)
      return tmpFile
    }
    lastStatus = response.status
  }

  throw new Error(`Failed to download file after ${MAX_RETRIES} attempts: ${lastStatus}`)
}

/**
 * Remove temp files. Silently ignores files that don't exist.
 */
export async function cleanupTempFiles(paths: string[]): Promise<void> {
  await Promise.all(
    paths.map((p) =>
      fs.unlink(p).catch(() => {
        // File may already be deleted or never created
      })
    )
  )
}

/**
 * Full pipeline: extract audio from a file already on disk, compress, split if needed.
 * Returns the list of chunk paths ready for Whisper, plus all temp paths for cleanup.
 *
 * @param inputPath - Path to the file already on disk
 */
export async function processFileForWhisper(inputPath: string): Promise<{
  chunks: ChunkInfo[]
  allTempPaths: string[]
}> {
  const ext = path.extname(inputPath).toLowerCase()
  const stat = statSync(inputPath)

  // Small files in a Whisper-native format can skip FFmpeg entirely
  if (stat.size <= WHISPER_MAX_SIZE && WHISPER_NATIVE_FORMATS.has(ext)) {
    console.log(`[processFileForWhisper] Skipping FFmpeg — native format (${ext}, ${(stat.size / 1024 / 1024).toFixed(1)}MB)`)
    return { chunks: [{ path: inputPath, offsetSeconds: 0 }], allTempPaths: [inputPath] }
  }

  // Verify the file actually contains audio before attempting extraction
  if (!(await hasAudioStream(inputPath))) {
    throw new Error(
      'This file does not contain an audio track. Please upload a file with audible speech.'
    )
  }

  const compressedPath = tempPath('.mp3')
  const allTempPaths = [inputPath, compressedPath]

  await extractAndCompressAudio(inputPath, compressedPath)

  const chunks = await splitAudioIfNeeded(compressedPath)

  // If chunks were created (different from compressedPath), track them too
  for (const c of chunks) {
    if (c.path !== compressedPath) {
      allTempPaths.push(c.path)
    }
  }

  return { chunks, allTempPaths }
}

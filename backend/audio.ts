import ffmpeg from 'fluent-ffmpeg'
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg'
import fs from 'fs/promises'
import { existsSync, statSync } from 'fs'
import path from 'path'
import os from 'os'
import crypto from 'crypto'
import Pitchfinder from 'pitchfinder'
import type { PitchWindow } from '@/lib/delivery-analytics'

ffmpeg.setFfmpegPath(ffmpegInstaller.path)

const WHISPER_MAX_SIZE = 25 * 1024 * 1024 // 25MB
const MAX_CHUNK_DURATION = 1400 // seconds — Whisper API limit is 1500s

export interface ChunkInfo {
  path: string
  offsetSeconds: number
}

export interface EnergyWindow {
  startTime: number
  endTime: number
  rmsDb: number
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

const ENERGY_WINDOW_SECONDS = 30
const ANALYSIS_SAMPLE_RATE = 16000
const BYTES_PER_SAMPLE = 4 // f32le
const SILENCE_FLOOR_DB = -96

// Pitch analysis constants
const PITCH_FRAME_SECONDS = 0.03   // 30ms frames
const PITCH_HOP_SECONDS = 0.01     // 10ms hop
const PITCH_MIN_HZ = 50
const PITCH_MAX_HZ = 600

// ── Helpers ──

function hzToSemitones(hz: number): number {
  return 12 * Math.log2(hz)
}

function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const idx = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

function stddev(values: number[]): number {
  if (values.length < 2) return 0
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const sqDiffs = values.map((v) => (v - mean) ** 2)
  return Math.sqrt(sqDiffs.reduce((a, b) => a + b, 0) / values.length)
}

// ── PCM Decode ──

/**
 * Decode audio file to raw PCM float32 mono at 16kHz.
 */
export async function decodeToRawPcm(filePath: string): Promise<Buffer> {
  const rawPath = tempPath('.pcm')

  await new Promise<void>((resolve, reject) => {
    ffmpeg(filePath)
      .format('f32le')
      .audioCodec('pcm_f32le')
      .audioChannels(1)
      .audioFrequency(ANALYSIS_SAMPLE_RATE)
      .on('error', (err: Error) => reject(err))
      .on('end', () => resolve())
      .save(rawPath)
  })

  const rawBuffer = await fs.readFile(rawPath)
  await fs.unlink(rawPath).catch(() => {})
  return rawBuffer
}

// ── Energy Analysis ──

/**
 * Compute RMS energy per time window from raw PCM float32 buffer.
 */
export function computeEnergyFromPcm(
  rawBuffer: Buffer,
  windowSeconds: number = ENERGY_WINDOW_SECONDS
): EnergyWindow[] {
  const totalSamples = rawBuffer.length / BYTES_PER_SAMPLE
  if (totalSamples === 0) return []

  const totalDuration = totalSamples / ANALYSIS_SAMPLE_RATE
  const samplesPerWindow = Math.floor(ANALYSIS_SAMPLE_RATE * windowSeconds)
  const numWindows = Math.max(1, Math.ceil(totalSamples / samplesPerWindow))
  const actualWindowDuration = totalDuration / numWindows

  const windows: EnergyWindow[] = []

  for (let i = 0; i < numWindows; i++) {
    const sampleStart = i * samplesPerWindow
    const sampleEnd = Math.min(sampleStart + samplesPerWindow, totalSamples)
    const count = sampleEnd - sampleStart

    let sumSquares = 0
    for (let j = sampleStart; j < sampleEnd; j++) {
      const sample = rawBuffer.readFloatLE(j * BYTES_PER_SAMPLE)
      sumSquares += sample * sample
    }

    const rms = Math.sqrt(sumSquares / count)
    const rmsDb = rms > 0 ? 20 * Math.log10(rms) : SILENCE_FLOOR_DB

    windows.push({
      startTime: i * actualWindowDuration,
      endTime: (i + 1) * actualWindowDuration,
      rmsDb: Math.round(rmsDb * 10) / 10,
    })
  }

  return windows
}

// ── Pitch Analysis ──

/**
 * Compute pitch (F0) per time window from raw PCM float32 buffer.
 * Uses YIN algorithm to detect fundamental frequency per frame, then aggregates
 * into windows with median, range, stddev (in semitones), and voiced ratio.
 */
export function computePitchFromPcm(
  rawBuffer: Buffer,
  windowSeconds: number = ENERGY_WINDOW_SECONDS
): PitchWindow[] {
  const totalSamples = rawBuffer.length / BYTES_PER_SAMPLE
  if (totalSamples === 0) return []

  const detectPitch = Pitchfinder.YIN({
    sampleRate: ANALYSIS_SAMPLE_RATE,
    threshold: 0.15,
  })

  // Extract per-frame pitch values
  const frameSamples = Math.floor(ANALYSIS_SAMPLE_RATE * PITCH_FRAME_SECONDS)
  const hopSamples = Math.floor(ANALYSIS_SAMPLE_RATE * PITCH_HOP_SECONDS)
  const totalDuration = totalSamples / ANALYSIS_SAMPLE_RATE

  interface FramePitch {
    time: number
    hz: number | null
  }

  const framePitches: FramePitch[] = []
  for (let offset = 0; offset + frameSamples <= totalSamples; offset += hopSamples) {
    const frame = new Float32Array(frameSamples)
    for (let j = 0; j < frameSamples; j++) {
      frame[j] = rawBuffer.readFloatLE((offset + j) * BYTES_PER_SAMPLE)
    }

    const raw = detectPitch(frame)
    const hz = raw !== null && raw >= PITCH_MIN_HZ && raw <= PITCH_MAX_HZ ? raw : null
    framePitches.push({
      time: offset / ANALYSIS_SAMPLE_RATE,
      hz,
    })
  }

  if (framePitches.length === 0) return []

  // Aggregate into windows
  const samplesPerWindow = Math.floor(ANALYSIS_SAMPLE_RATE * windowSeconds)
  const numWindows = Math.max(1, Math.ceil(totalSamples / samplesPerWindow))
  const actualWindowDuration = totalDuration / numWindows

  const windows: PitchWindow[] = []

  for (let i = 0; i < numWindows; i++) {
    const winStart = i * actualWindowDuration
    const winEnd = (i + 1) * actualWindowDuration

    const framesInWindow = framePitches.filter((f) => f.time >= winStart && f.time < winEnd)
    const totalFrames = framesInWindow.length
    const voicedHz = framesInWindow.filter((f) => f.hz !== null).map((f) => f.hz as number)

    if (totalFrames === 0) {
      windows.push({
        startTime: winStart,
        endTime: winEnd,
        medianF0Hz: 0,
        medianF0Semitones: 0,
        f0RangeSemitones: 0,
        f0StddevSemitones: 0,
        voicedFrameRatio: 0,
      })
      continue
    }

    if (voicedHz.length === 0) {
      windows.push({
        startTime: winStart,
        endTime: winEnd,
        medianF0Hz: 0,
        medianF0Semitones: 0,
        f0RangeSemitones: 0,
        f0StddevSemitones: 0,
        voicedFrameRatio: 0,
      })
      continue
    }

    const semitones = voicedHz.map(hzToSemitones)
    const medHz = median(voicedHz)
    const medSt = median(semitones)
    const p10 = percentile(semitones, 10)
    const p90 = percentile(semitones, 90)

    windows.push({
      startTime: Math.round(winStart * 10) / 10,
      endTime: Math.round(winEnd * 10) / 10,
      medianF0Hz: Math.round(medHz * 10) / 10,
      medianF0Semitones: Math.round(medSt * 10) / 10,
      f0RangeSemitones: Math.round((p90 - p10) * 10) / 10,
      f0StddevSemitones: Math.round(stddev(semitones) * 10) / 10,
      voicedFrameRatio: Math.round((voicedHz.length / totalFrames) * 100) / 100,
    })
  }

  return windows
}

// ── Combined Analysis ──

/**
 * Decode audio once and run both energy and pitch analysis on the same PCM buffer.
 */
export async function analyzeAudio(
  filePath: string,
  windowSeconds: number = ENERGY_WINDOW_SECONDS
): Promise<{ energyWindows: EnergyWindow[]; pitchWindows: PitchWindow[] }> {
  const rawBuffer = await decodeToRawPcm(filePath)
  const energyWindows = computeEnergyFromPcm(rawBuffer, windowSeconds)
  const pitchWindows = computePitchFromPcm(rawBuffer, windowSeconds)
  return { energyWindows, pitchWindows }
}

/**
 * Backward-compatible: decode and compute energy only.
 */
export async function analyzeAudioEnergy(
  filePath: string,
  windowSeconds: number = ENERGY_WINDOW_SECONDS
): Promise<EnergyWindow[]> {
  const rawBuffer = await decodeToRawPcm(filePath)
  return computeEnergyFromPcm(rawBuffer, windowSeconds)
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
  analysisPath: string
}> {
  const ext = path.extname(inputPath).toLowerCase()
  const stat = statSync(inputPath)

  // Small files in a Whisper-native format can skip FFmpeg entirely
  if (stat.size <= WHISPER_MAX_SIZE && WHISPER_NATIVE_FORMATS.has(ext)) {
    console.log(`[processFileForWhisper] Skipping FFmpeg — native format (${ext}, ${(stat.size / 1024 / 1024).toFixed(1)}MB)`)
    return { chunks: [{ path: inputPath, offsetSeconds: 0 }], allTempPaths: [inputPath], analysisPath: inputPath }
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

  return { chunks, allTempPaths, analysisPath: compressedPath }
}

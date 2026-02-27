import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock ffmpeg-installer before fluent-ffmpeg import
vi.mock('@ffmpeg-installer/ffmpeg', () => ({
  default: { path: '/usr/bin/ffmpeg' },
  path: '/usr/bin/ffmpeg',
}))

// Mock fluent-ffmpeg — factory must be self-contained (no top-level var references)
vi.mock('fluent-ffmpeg', () => {
  const createChain = () => {
    const callbacks: Record<string, (...args: unknown[]) => void> = {}
    const chain: Record<string, unknown> = {}
    chain.noVideo = vi.fn().mockReturnValue(chain)
    chain.audioCodec = vi.fn().mockReturnValue(chain)
    chain.audioBitrate = vi.fn().mockReturnValue(chain)
    chain.audioChannels = vi.fn().mockReturnValue(chain)
    chain.audioFrequency = vi.fn().mockReturnValue(chain)
    chain.format = vi.fn().mockReturnValue(chain)
    chain.setStartTime = vi.fn().mockReturnValue(chain)
    chain.setDuration = vi.fn().mockReturnValue(chain)
    chain.on = vi.fn().mockImplementation((event: string, cb: (...args: unknown[]) => void) => {
      callbacks[event] = cb
      return chain
    })
    chain.save = vi.fn().mockImplementation(() => {
      // When save() is called, trigger 'end' synchronously (simulating ffmpeg completing)
      if (callbacks['end']) callbacks['end']()
      return chain
    })
    return chain
  }

  const mockFfprobe = vi.fn()
  const fn = Object.assign(vi.fn().mockImplementation(() => createChain()), {
    setFfmpegPath: vi.fn(),
    ffprobe: mockFfprobe,
  })

  return { default: fn }
})

// Mock fs modules — share references between default and named exports
// (CJS-to-ESM interop means named imports may come from the default object)
vi.mock('fs/promises', () => {
  const _unlink = vi.fn().mockResolvedValue(undefined)
  const _writeFile = vi.fn().mockResolvedValue(undefined)
  return {
    default: { unlink: _unlink, writeFile: _writeFile },
    unlink: _unlink,
    writeFile: _writeFile,
  }
})

vi.mock('fs', () => {
  const _statSync = vi.fn().mockReturnValue({ size: 1024 })
  const _existsSync = vi.fn().mockReturnValue(true)
  const _createReadStream = vi.fn().mockReturnValue('mock-stream')
  return {
    default: { statSync: _statSync, existsSync: _existsSync, createReadStream: _createReadStream },
    statSync: _statSync,
    existsSync: _existsSync,
    createReadStream: _createReadStream,
  }
})

vi.mock('@vercel/blob', () => ({
  head: vi.fn().mockResolvedValue({ size: 1024, url: 'https://example.com/test.mp4' }),
}))

import ffmpeg from 'fluent-ffmpeg'
import { tempPath, cleanupTempFiles, processFileForWhisper, splitAudioIfNeeded } from '@/backend/audio'
import { statSync } from 'fs'
import fsPromises from 'fs/promises'

function mockFfprobe(data: { streams: { codec_type: string }[]; format: { duration: number } }) {
  vi.mocked(ffmpeg.ffprobe).mockImplementation(
    (_path: unknown, cb: unknown) => {
      (cb as (err: null, data: unknown) => void)(null, data)
    }
  )
}

describe('tempPath', () => {
  it('generates paths in the temp directory', () => {
    const p = tempPath('.mp3')
    expect(p).toMatch(/\.mp3$/)
    expect(p).toContain('vera-')
  })

  it('generates unique paths on each call', () => {
    const p1 = tempPath('.mp3')
    const p2 = tempPath('.mp3')
    expect(p1).not.toBe(p2)
  })

  it('uses custom prefix when provided', () => {
    const p = tempPath('.wav', 'custom')
    expect(p).toContain('custom-')
    expect(p).toMatch(/\.wav$/)
  })
})

describe('cleanupTempFiles', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('calls unlink for each path', async () => {
    await cleanupTempFiles(['/tmp/file1.mp3', '/tmp/file2.mp3'])
    expect(fsPromises.unlink).toHaveBeenCalledTimes(2)
    expect(fsPromises.unlink).toHaveBeenCalledWith('/tmp/file1.mp3')
    expect(fsPromises.unlink).toHaveBeenCalledWith('/tmp/file2.mp3')
  })

  it('silently handles files that do not exist', async () => {
    vi.mocked(fsPromises.unlink).mockRejectedValue(new Error('ENOENT'))
    await expect(cleanupTempFiles(['/tmp/missing.mp3'])).resolves.toBeUndefined()
  })

  it('handles empty array', async () => {
    await cleanupTempFiles([])
    expect(fsPromises.unlink).not.toHaveBeenCalled()
  })
})

describe('splitAudioIfNeeded', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns ChunkInfo with original path and offsetSeconds: 0 when file is small enough', async () => {
    vi.mocked(statSync).mockReturnValue({ size: 1024 } as ReturnType<typeof statSync>)
    mockFfprobe({ streams: [{ codec_type: 'audio' }], format: { duration: 60 } })

    const result = await splitAudioIfNeeded('/tmp/small.mp3')
    expect(result).toEqual([{ path: '/tmp/small.mp3', offsetSeconds: 0 }])
  })

  it('splits file when it exceeds max size and returns ChunkInfo with offsets', async () => {
    // 60MB file, exceeds 25MB limit → ceil(60/25) = 3 chunks
    vi.mocked(statSync).mockReturnValue({ size: 60 * 1024 * 1024 } as ReturnType<typeof statSync>)
    mockFfprobe({ streams: [{ codec_type: 'audio' }], format: { duration: 600 } })

    const result = await splitAudioIfNeeded('/tmp/large.mp3')
    expect(result).toHaveLength(3)
    for (const chunk of result) {
      expect(chunk.path).toContain('chunk')
      expect(typeof chunk.offsetSeconds).toBe('number')
    }
    expect(result[0].offsetSeconds).toBe(0)
    expect(result[1].offsetSeconds).toBe(200) // floor(600/3) = 200
    expect(result[2].offsetSeconds).toBe(400)
  })

  it('splits file based on duration if that produces more chunks', async () => {
    // Small file but very long duration
    vi.mocked(statSync).mockReturnValue({ size: 1024 } as ReturnType<typeof statSync>)
    // 3000s / 1400s limit = ceil(2.14) = 3 chunks
    mockFfprobe({ streams: [{ codec_type: 'audio' }], format: { duration: 3000 } })

    const result = await splitAudioIfNeeded('/tmp/long.mp3')
    expect(result).toHaveLength(3)
    expect(result[0].offsetSeconds).toBe(0)
  })
})

describe('processFileForWhisper', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('skips FFmpeg for small native-format files', async () => {
    // 10MB mp3 file — under 25MB and native format
    vi.mocked(statSync).mockReturnValue({ size: 10 * 1024 * 1024 } as ReturnType<typeof statSync>)

    const result = await processFileForWhisper('/tmp/small.mp3')
    expect(result.chunks).toEqual([{ path: '/tmp/small.mp3', offsetSeconds: 0 }])
    expect(result.allTempPaths).toEqual(['/tmp/small.mp3'])
    // FFmpeg constructor should NOT have been called
    expect(ffmpeg).not.toHaveBeenCalled()
  })

  it('skips FFmpeg for small wav files', async () => {
    vi.mocked(statSync).mockReturnValue({ size: 5 * 1024 * 1024 } as ReturnType<typeof statSync>)

    const result = await processFileForWhisper('/tmp/recording.wav')
    expect(result.chunks).toEqual([{ path: '/tmp/recording.wav', offsetSeconds: 0 }])
    expect(ffmpeg).not.toHaveBeenCalled()
  })

  it('throws when file has no audio stream', async () => {
    // Large file forces FFmpeg path
    vi.mocked(statSync).mockReturnValue({ size: 30 * 1024 * 1024 } as ReturnType<typeof statSync>)
    // ffprobe returns no audio streams
    mockFfprobe({ streams: [{ codec_type: 'video' }], format: { duration: 120 } })

    await expect(processFileForWhisper('/tmp/video-only.avi')).rejects.toThrow(
      'does not contain an audio track'
    )
  })

  it('processes non-native format files through FFmpeg', async () => {
    // Small AVI file — non-native format needs conversion regardless of size
    vi.mocked(statSync).mockReturnValue({ size: 5 * 1024 * 1024 } as ReturnType<typeof statSync>)
    mockFfprobe({ streams: [{ codec_type: 'audio' }], format: { duration: 60 } })

    const result = await processFileForWhisper('/tmp/recording.avi')
    // Should have called ffmpeg for extraction
    expect(ffmpeg).toHaveBeenCalled()
    expect(result.allTempPaths).toContain('/tmp/recording.avi')
    // compressed path should also be tracked
    expect(result.allTempPaths.length).toBeGreaterThanOrEqual(2)
  })

  it('processes large native-format files through FFmpeg', async () => {
    // 30MB mp3 — native format but over 25MB limit
    vi.mocked(statSync).mockReturnValue({ size: 30 * 1024 * 1024 } as ReturnType<typeof statSync>)
    mockFfprobe({ streams: [{ codec_type: 'audio' }], format: { duration: 300 } })

    const result = await processFileForWhisper('/tmp/large.mp3')
    expect(ffmpeg).toHaveBeenCalled()
    expect(result.chunks.length).toBeGreaterThanOrEqual(1)
    expect(result.chunks[0].offsetSeconds).toBe(0)
    expect(result.allTempPaths.length).toBeGreaterThanOrEqual(2)
  })
})

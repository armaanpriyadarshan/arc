import { z } from 'zod'

export const MAX_FILE_SIZE = 500 * 1024 * 1024 // 500MB
export const MAX_CONTEXT_DOC_SIZE = 50 * 1024 * 1024 // 50MB

// Audio/video extensions that ffmpeg can process
const MEDIA_EXTENSIONS = [
  '.mp3', '.mp4', '.m4a', '.wav', '.mpeg', '.mpga', '.webm',
  '.ogg', '.flac', '.aac', '.wma', '.mov', '.avi', '.mkv',
  '.m4v', '.opus', '.3gp',
] as const

// Document extensions accepted for context file uploads
const CONTEXT_DOC_EXTENSIONS = ['.pdf', '.txt', '.docx'] as const

const CONTEXT_DOC_MIMES = [
  'application/pdf',
  'text/plain',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
] as const

export function validateFile(
  file: { name: string; type: string; size: number }
): { valid: true } | { valid: false; error: string } {
  if (file.size === 0) {
    return { valid: false, error: 'File is empty' }
  }

  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File size exceeds 500MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`,
    }
  }

  // Accept any audio/* or video/* MIME type (ffmpeg can convert most formats)
  const mimeValid =
    file.type.startsWith('audio/') || file.type.startsWith('video/')
  const name = file.name.toLowerCase()
  const extValid = MEDIA_EXTENSIONS.some((ext) => name.endsWith(ext))

  if (!mimeValid && !extValid) {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload an audio or video file.',
    }
  }

  return { valid: true }
}

export function validateContextFile(
  file: { name: string; type: string; size: number }
): { valid: true } | { valid: false; error: string } {
  if (file.size === 0) {
    return { valid: false, error: 'File is empty' }
  }

  const name = file.name.toLowerCase()
  const isAudio = file.type.startsWith('audio/') || MEDIA_EXTENSIONS.some((ext) => name.endsWith(ext))

  if (isAudio) {
    if (file.size > MAX_FILE_SIZE) {
      return {
        valid: false,
        error: `Audio file size exceeds 500MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`,
      }
    }
    return { valid: true }
  }

  // Document files
  const extValid = CONTEXT_DOC_EXTENSIONS.some((ext) => name.endsWith(ext))
  const mimeValid = CONTEXT_DOC_MIMES.some((m) => file.type === m)

  if (!extValid && !mimeValid) {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload a PDF, .txt, .docx, or audio file.',
    }
  }

  if (file.size > MAX_CONTEXT_DOC_SIZE) {
    return {
      valid: false,
      error: `Document size exceeds 50MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`,
    }
  }

  return { valid: true }
}

export const extractContextRequestSchema = z.object({
  blobUrl: z.string().url(),
  fileName: z.string().min(1).max(255),
})

const chatMessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string().min(1).max(50_000),
})

export const chatRequestSchema = z.object({
  messages: z.array(chatMessageSchema).min(1).max(200),
  transcript: z.string().max(100_000).optional(),
  researchContext: z.string().max(20_000).optional(),
  slideContext: z.string().max(30_000).optional(),
  stage: z.enum(['define', 'present', 'feedback', 'followup']).default('define'),
  setupContext: z.object({
    topic: z.string().max(500).optional(),
    audience: z.string().max(500).optional(),
    goal: z.string().max(500).optional(),
    additionalContext: z.string().max(2000).optional(),
    fileContext: z.string().max(50_000).optional(),
  }).optional(),
})

export const researchRequestSchema = z.object({
  transcript: z.string().max(100_000).optional(),
  audienceDescription: z.string().min(1).max(10_000),
  topic: z.string().max(2_000).optional(),
  goal: z.string().max(500).optional(),
  additionalContext: z.string().max(2_000).optional(),
})

export const transcribeRequestSchema = z.object({
  blobUrl: z.string().url(),
  fileName: z.string().min(1).max(255),
})

export const MAX_SLIDE_FILE_SIZE = 50 * 1024 * 1024 // 50MB

export function validateSlideFile(
  file: { name: string; type: string; size: number }
): { valid: true } | { valid: false; error: string } {
  if (file.size === 0) {
    return { valid: false, error: 'File is empty' }
  }

  if (file.size > MAX_SLIDE_FILE_SIZE) {
    return {
      valid: false,
      error: `File size exceeds 50MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`,
    }
  }

  const name = file.name.toLowerCase()
  const extValid = name.endsWith('.pdf') || name.endsWith('.pptx')
  const mimeValid = file.type === 'application/pdf' ||
    file.type === 'application/vnd.openxmlformats-officedocument.presentationml.presentation'

  if (!extValid && !mimeValid) {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload a PDF or PPTX file.',
    }
  }

  return { valid: true }
}

export const slideAnalyzeRequestSchema = z.object({
  blobUrl: z.string().url(),
  fileName: z.string().min(1).max(255),
  audienceContext: z.string().max(2_000).optional(),
})

export const blobDeleteRequestSchema = z.object({
  urls: z.array(z.string().url()).min(1).max(50),
})

export const feedbackScoreRequestSchema = z.object({
  sessionId: z.string().min(1),
  transcript: z.string().max(100_000).optional(),
  setup: z.object({
    topic: z.string().max(500),
    audience: z.string().max(500),
    goal: z.string().max(500),
    additionalContext: z.string().max(2000).optional(),
    fileContext: z.string().max(50_000).optional(),
  }),
  messages: z
    .array(
      z.object({
        role: z.string(),
        content: z.string(),
      })
    )
    .max(200),
  researchContext: z.string().max(20_000).optional(),
  slideContext: z.string().max(30_000).optional(),
  deliveryAnalyticsSummary: z.string().max(5_000).optional(),
})

export function sanitizeInput(text: string): string {
  return text.trim()
}

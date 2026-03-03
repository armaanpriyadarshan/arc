import { NextRequest, NextResponse } from 'next/server'
import { handleUpload, type HandleUploadBody } from '@vercel/blob/client'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'

export async function handleUploadRoute(request: NextRequest) {
  const ip = getClientIp(request)
  if (!checkRateLimit('upload:' + ip, RATE_LIMITS.upload.limit, RATE_LIMITS.upload.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait before uploading again.' },
      { status: 429 }
    )
  }

  try {
    const body = (await request.json()) as HandleUploadBody
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (_pathname) => {
        return {
          allowedContentTypes: [
            'audio/*',
            'video/*',
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
          ],
          maximumSizeInBytes: 500 * 1024 * 1024, // 500MB
          addRandomSuffix: true,
        }
      },
    })

    return NextResponse.json(jsonResponse)
  } catch (error) {
    console.error('Upload token error:', error)
    return NextResponse.json(
      { error: 'Failed to generate upload token.' },
      { status: 500 }
    )
  }
}

import { NextRequest, NextResponse } from 'next/server'
import { del } from '@vercel/blob'
import { blobDeleteRequestSchema } from '@/backend/validation'
import { checkRateLimit, getClientIp } from '@/backend/rate-limit'
import { RATE_LIMITS } from '@/backend/rate-limit-config'

export async function handleBlobDelete(request: NextRequest) {
  const ip = getClientIp(request)
  if (!checkRateLimit('blob-delete:' + ip, RATE_LIMITS.blobDelete.limit, RATE_LIMITS.blobDelete.windowMs).allowed) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait before trying again.' },
      { status: 429 }
    )
  }

  const body = await request.json().catch(() => null)
  const parsed = blobDeleteRequestSchema.safeParse(body)

  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid request. Provide an array of blob URLs.' },
      { status: 400 }
    )
  }

  // Best-effort deletion — don't fail the request if blob deletion fails
  await del(parsed.data.urls).catch(() => {})

  return NextResponse.json({ deleted: true })
}

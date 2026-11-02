import { ApiError, api } from '../lib/api'
import { parseSSEStream } from '../lib/sse'
import type { StreamDoneEvent } from '../lib/types'

export interface StreamCallbacks {
  onToken: (token: string) => void
}

/**
 * Streams a chat answer token-by-token via POST /chat/stream. Falls back to
 * the "done" event's final answer if no token events arrived at all.
 */
export async function streamChat(
  token: string,
  sessionId: string,
  userQuery: string,
  { onToken }: StreamCallbacks,
  signal?: AbortSignal,
): Promise<StreamDoneEvent | null> {
  const response = await fetch(`${api.baseUrl}/chat/stream`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ session_id: sessionId, user_query: userQuery }),
    signal,
  })

  if (response.status === 401) {
    throw new ApiError(401, 'Session expired. Please sign in again.')
  }
  if (response.status >= 400) {
    const detail = await response.text().catch(() => '')
    throw new ApiError(response.status, detail || `HTTP ${response.status}`)
  }

  let sawToken = false
  let done: StreamDoneEvent | null = null

  for await (const { event, data } of parseSSEStream(response)) {
    if (!data) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(data)
    } catch {
      continue
    }

    if (event === 'token') {
      const tokenText = (parsed as { token?: string }).token
      if (tokenText) {
        sawToken = true
        onToken(tokenText)
      }
    } else if (event === 'done') {
      done = parsed as StreamDoneEvent
      if (!sawToken && done.answer) {
        onToken(done.answer)
      }
    }
  }

  return done
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

import { api } from '../lib/api'
import type { UploadedDocument } from '../lib/types'

const POLL_INTERVAL_MS = 2000
const POLL_TIMEOUT_MS = 60_000

export interface IngestionStatus {
  filename: string
  status: 'queued' | 'processing' | 'ready' | 'failed'
  failure_reason: string | null
}

export type IngestionStatusMap = Record<string, IngestionStatus>

/**
 * Uploads files, then polls each document's status until every one reaches
 * a terminal state (ready/failed) or POLL_TIMEOUT_MS elapses.
 */
export async function uploadAndTrackDocuments(
  token: string,
  sessionId: string,
  files: File[],
  onStatusUpdate: (statuses: IngestionStatusMap) => void,
): Promise<UploadedDocument[]> {
  const { uploaded } = await api.uploadDocuments(token, sessionId, files)
  if (uploaded.length === 0) {
    return uploaded
  }

  const statuses: IngestionStatusMap = {}
  for (const doc of uploaded) {
    statuses[doc.document_id] = {
      filename: doc.filename,
      status: 'queued',
      failure_reason: null,
    }
  }
  onStatusUpdate({ ...statuses })

  const deadline = Date.now() + POLL_TIMEOUT_MS
  let pending = uploaded.map((doc) => doc.document_id)

  while (pending.length > 0 && Date.now() < deadline) {
    const results = await Promise.all(
      pending.map((documentId) =>
        api.getDocumentStatus(token, documentId).catch(() => null),
      ),
    )

    const stillPending: string[] = []
    results.forEach((result, index) => {
      const documentId = pending[index]
      if (!result) {
        stillPending.push(documentId)
        return
      }
      statuses[documentId] = {
        filename: statuses[documentId].filename,
        status: result.status,
        failure_reason: result.failure_reason,
      }
      if (result.status === 'queued' || result.status === 'processing') {
        stillPending.push(documentId)
      }
    })

    onStatusUpdate({ ...statuses })
    pending = stillPending
    if (pending.length > 0) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
    }
  }

  return uploaded
}

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, ClipboardEvent } from 'react'
import {
  ACCEPTED_EXTENSIONS,
  MAX_FILES_PER_REQUEST,
  MAX_TOTAL_ATTACHMENT_BYTES,
  formatBytes,
  totalBytes,
  uploadFiles,
} from '../../lib/api'
import type { UploadedFile } from '../../types/chat'
import styles from './Chat.module.css'

interface AttachmentBarProps {
  files: UploadedFile[]
  userId: string
  disabled: boolean
  onChange: (files: UploadedFile[]) => void
}

/**
 * Per-type badge. Kept as text glyphs rather than an icon font so there is no
 * new dependency and no flash of missing glyphs.
 */
function TypeBadge({ file }: { file: UploadedFile }) {
  const label: Record<UploadedFile['category'], string> = {
    pdf: 'PDF',
    word: 'DOC',
    excel: 'XLS',
    image: 'IMG',
    powerpoint: 'PPT',
    text: 'TXT',
  }

  return (
    <span className={styles.typeBadge} data-category={file.category}>
      {label[file.category]}
    </span>
  )
}

/**
 * Pull image files out of a paste event.
 *
 * A clipboard screenshot arrives as an image blob with no filename, so one is
 * synthesised from the current time. Without it the backend sees an empty name,
 * rejects the upload for having no extension, and pasting silently does
 * nothing.
 */
function imagesFromClipboard(event: ClipboardEvent): File[] {
  const items = Array.from(event.clipboardData?.items ?? [])

  return items
    .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
    .map((item, index) => {
      const blob = item.getAsFile()
      if (!blob) return null

      const stamp = new Date().toISOString().replace(/[:.]/g, '-')

      return new File([blob], `pasted-${stamp}-${index + 1}.png`, {
        type: blob.type,
      })
    })
    .filter((file): file is File => file !== null)
}

/**
 * Upload control for files to be attached to a later email.
 *
 * Uploads immediately rather than on send, so the returned file_ids exist before
 * the message is composed and the preview can name the exact files the user
 * will be asked to approve.
 */
export function AttachmentBar({
  files,
  userId,
  disabled,
  onChange,
}: AttachmentBarProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ingest = useCallback(
    async (picked: File[]) => {
      if (picked.length === 0) return

      setError(null)

      if (files.length + picked.length > MAX_FILES_PER_REQUEST) {
        setError(
          `At most ${MAX_FILES_PER_REQUEST} files per message. You already have ${files.length}.`,
        )
        return
      }

      const projected =
        totalBytes(files) + picked.reduce((n, f) => n + f.size, 0)

      if (projected > MAX_TOTAL_ATTACHMENT_BYTES) {
        setError(
          `Attachments would total ${formatBytes(projected)}, over the ` +
            `${formatBytes(MAX_TOTAL_ATTACHMENT_BYTES)} limit.`,
        )
        return
      }

      setBusy(true)

      try {
        const result = await uploadFiles(picked, userId)
        onChange([...files, ...result.files])
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Upload failed.')
      } finally {
        setBusy(false)
      }
    },
    [files, onChange, userId],
  )

  const handleChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? [])

    // Reset so picking the same file twice in a row still fires a change.
    event.target.value = ''

    await ingest(picked)
  }

  // Paste anywhere on the page uploads the image, so a screenshot can be
  // pasted straight into the chat box without aiming at a control first.
  useEffect(() => {
    if (disabled || busy) return

    const onPaste = (event: globalThis.ClipboardEvent) => {
      const images = imagesFromClipboard(event as unknown as ClipboardEvent)

      if (images.length === 0) return

      // Only images are consumed. A paste of text into the composer must still
      // reach the textarea, so preventDefault is limited to this branch.
      event.preventDefault()
      void ingest(images)
    }

    window.addEventListener('paste', onPaste)

    return () => window.removeEventListener('paste', onPaste)
  }, [busy, disabled, ingest])

  const remove = (fileId: string) => {
    onChange(files.filter((file) => file.file_id !== fileId))
  }

  const atLimit = files.length >= MAX_FILES_PER_REQUEST

  return (
    <div className={styles.attachmentBar}>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS.join(',')}
        className={styles.fileInput}
        onChange={handleChange}
        disabled={disabled || busy || atLimit}
        aria-label="Attach files"
      />

      <button
        type="button"
        className={styles.attachButton}
        onClick={() => inputRef.current?.click()}
        disabled={disabled || busy || atLimit}
        title="Attach files, or paste an image anywhere"
      >
        {busy ? 'Uploading…' : '+ Attach / paste'}
      </button>

      {files.length > 0 ? (
        <span className={styles.attachmentCount}>
          {files.length}/{MAX_FILES_PER_REQUEST} · {formatBytes(totalBytes(files))}
        </span>
      ) : null}

      {files.length > 0 ? (
        <ul className={styles.attachmentList}>
          {files.map((file) => {
            const isImage = file.category === 'image' || file.mime.startsWith('image/')
            const previewSrc = file.thumbnail || (file as any).localUrl

            if (isImage && previewSrc) {
              return (
                <li key={file.file_id} className={styles.imageCard}>
                  <div className={styles.imageThumbWrapper}>
                    <img
                      src={previewSrc}
                      alt={file.name}
                      className={styles.imageThumb}
                    />
                    <button
                      type="button"
                      className={styles.imageRemoveOverlay}
                      onClick={() => remove(file.file_id)}
                      aria-label={`Remove ${file.name}`}
                      title="Remove image"
                    >
                      ×
                    </button>
                  </div>
                  <div className={styles.imageMeta}>
                    <span className={styles.attachmentName} title={file.name}>
                      {file.name}
                    </span>
                    <span className={styles.attachmentSize}>
                      {formatBytes(file.size)}
                    </span>
                  </div>
                </li>
              )
            }

            return (
              <li key={file.file_id} className={styles.attachmentChip}>
                <TypeBadge file={file} />
                <span className={styles.attachmentName} title={file.name}>
                  {file.name}
                </span>
                <span className={styles.attachmentSize}>
                  {formatBytes(file.size)}
                </span>
                <button
                  type="button"
                  className={styles.attachmentRemove}
                  onClick={() => remove(file.file_id)}
                  aria-label={`Remove ${file.name}`}
                >
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      ) : null}

      {error ? (
        <span className={styles.attachmentError} role="alert">
          {error}
        </span>
      ) : null}
    </div>
  )
}
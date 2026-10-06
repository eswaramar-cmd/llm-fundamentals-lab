/**
 * MarkdownRenderer — renders agent responses with full GitHub-flavored Markdown.
 *
 * Provides ChatGPT-style structured, readable, and clean output:
 * - Code blocks with language badges & 1-click Copy button
 * - Clean responsive Markdown tables
 * - Blockquotes, bullet points, and numbered lists
 * - Inline code and links
 */

import { useState, useCallback, memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import styles from './MarkdownRenderer.module.css'

/** Code block component with copy-to-clipboard functionality */
const CodeBlock = memo(function CodeBlock({
  language,
  value,
}: {
  language?: string
  value: string
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback for older browsers or permission denials
      const textarea = document.createElement('textarea')
      textarea.value = value
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [value])

  return (
    <div className={styles.codeBlockContainer}>
      <div className={styles.codeBlockHeader}>
        <span className={styles.codeLang}>{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          className={styles.copyButton}
          aria-label="Copy code to clipboard"
          title="Copy code"
        >
          {copied ? (
            <>
              <svg className={styles.copyIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Copied!</span>
            </>
          ) : (
            <>
              <svg className={styles.copyIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className={styles.pre}>
        <code>{value}</code>
      </pre>
    </div>
  )
})

/* Custom renderers for markdown elements */
const components: Components = {
  h1: ({ children }) => <h1 className={styles.h1}>{children}</h1>,
  h2: ({ children }) => <h2 className={styles.h2}>{children}</h2>,
  h3: ({ children }) => <h3 className={styles.h3}>{children}</h3>,
  h4: ({ children }) => <h4 className={styles.h4}>{children}</h4>,

  p: ({ children }) => <p className={styles.p}>{children}</p>,

  ul: ({ children }) => <ul className={styles.ul}>{children}</ul>,
  ol: ({ children }) => <ol className={styles.ol}>{children}</ol>,
  li: ({ children }) => <li className={styles.li}>{children}</li>,

  // Handle both inline code and multiline fenced code blocks
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  code: ({ inline, className, children, ...props }: any) => {
    const rawContent = String(children || '').replace(/\n$/, '')
    if (inline) {
      return (
        <code className={styles.inlineCode} {...props}>
          {children}
        </code>
      )
    }

    const match = /language-(\w+)/.exec(className ?? '')
    const language = match ? match[1] : ''

    return <CodeBlock language={language} value={rawContent} />
  },

  blockquote: ({ children }) => (
    <blockquote className={styles.blockquote}>{children}</blockquote>
  ),

  table: ({ children }) => (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className={styles.thead}>{children}</thead>,
  tbody: ({ children }) => <tbody className={styles.tbody}>{children}</tbody>,
  tr: ({ children }) => <tr className={styles.tr}>{children}</tr>,
  th: ({ children }) => <th className={styles.th}>{children}</th>,
  td: ({ children }) => <td className={styles.td}>{children}</td>,

  hr: () => <hr className={styles.hr} />,

  a: ({ href, children }) => (
    <a
      className={styles.link}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),

  strong: ({ children }) => <strong className={styles.strong}>{children}</strong>,
  em: ({ children }) => <em className={styles.em}>{children}</em>,
}

interface Props {
  content: string
}

export function MarkdownRenderer({ content }: Props) {
  return (
    <div className={styles.root}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialMessages = [
  {
    role: "assistant",
    content: "Welcome to Codebase Intelligence. Ingest any GitHub repository or local directory path to begin asking natural language questions about architecture, classes, and call flows."
  }
];

function App() {
  const [source, setSource] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [loadingIngest, setLoadingIngest] = useState(false);
  const [loadingQuery, setLoadingQuery] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [lastIngestResult, setLastIngestResult] = useState(null);
  const [ingestJobId, setIngestJobId] = useState(null);
  const [ingestStatus, setIngestStatus] = useState(null);
  const [localIngestElapsed, setLocalIngestElapsed] = useState(0);
  const ingestPollTimeoutRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loadingQuery]);

  useEffect(() => {
    if (!loadingIngest) {
      setLocalIngestElapsed(0);
      return undefined;
    }

    const timer = window.setInterval(() => {
      setLocalIngestElapsed((current) => current + 1);
    }, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [loadingIngest]);

  useEffect(() => {
    if (!ingestJobId || !loadingIngest) {
      return undefined;
    }

    let cancelled = false;

    const stopPolling = () => {
      if (ingestPollTimeoutRef.current) {
        window.clearTimeout(ingestPollTimeoutRef.current);
        ingestPollTimeoutRef.current = null;
      }
    };

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ingest/status/${ingestJobId}`);
        const data = await response.json();
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error("Ingestion job expired or backend restarted. Please ingest again.");
          }
          throw new Error(data.detail || "Could not fetch ingestion status");
        }

        if (cancelled) return;
        setIngestStatus(data);

        if (data.status === "completed") {
          setLastIngestResult(data.result);
          setMessages((current) => [
            ...current,
            {
              role: "assistant",
              content:
                `Successfully indexed ${data.result.files_indexed} files and ${data.result.chunks_indexed} code chunks from ${data.result.source_root}.\n\n` +
                `Summary: ${data.result.repo_summary}`
            }
          ]);
          setLoadingIngest(false);
          setIngestJobId(null);
          stopPolling();
          return;
        }

        if (data.status === "failed") {
          setMessages((current) => [
            ...current,
            { role: "assistant", content: `Ingestion error: ${data.error || data.message}` }
          ]);
          setLoadingIngest(false);
          setIngestJobId(null);
          stopPolling();
          return;
        }

        ingestPollTimeoutRef.current = window.setTimeout(() => {
          pollStatus().catch((error) => {
            if (cancelled) return;
            setMessages((current) => [
              ...current,
              { role: "assistant", content: `Ingestion error: ${error.message}` }
            ]);
            setLoadingIngest(false);
            setIngestJobId(null);
            stopPolling();
          });
        }, 1000);
      } catch (error) {
        if (cancelled) return;
        setMessages((current) => [
          ...current,
          { role: "assistant", content: `Ingestion error: ${error.message}` }
        ]);
        setLoadingIngest(false);
        setIngestJobId(null);
        stopPolling();
      }
    };

    pollStatus();

    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [ingestJobId, loadingIngest]);

  const ingestRepository = async (e) => {
    e?.preventDefault();
    if (!source.trim()) return;
    setLoadingIngest(true);
    setIngestStatus({
      status: "running",
      progress: 1,
      message: "Initiating ingestion pipeline...",
      elapsed_seconds: 0
    });
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source.trim() })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Ingestion failed");
      }
      setIngestJobId(data.job_id);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Ingestion error: ${error.message}` }
      ]);
      setLoadingIngest(false);
    }
  };

  const submitQuery = async (e) => {
    e?.preventDefault();
    if (!query.trim() || loadingQuery) return;

    const userQuery = query.trim();
    setMessages((current) => [...current, { role: "user", content: userQuery }]);
    setQuery("");
    setLoadingQuery(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery, top_k: 5 })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Query failed");
      }
      setLastResult(data);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Query error: ${error.message}` }
      ]);
    } finally {
      setLoadingQuery(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  };

  return (
    <div className="app-container">
      {/* Top Header / Brand Bar */}
      <header className="top-navbar">
        <div className="brand-badge">
          <div className="brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M16 18l6-6-6-6M8 6l-6 6 6 6" />
            </svg>
          </div>
          <span className="brand-title">Codebase Intelligence</span>
        </div>
        <div className="nav-badges">
          <span className="pill-badge active">Hybrid RAG</span>
          <span className="pill-badge">FAISS + PostgreSQL</span>
        </div>
      </header>

      {/* Ingestion Hero Box */}
      <section className="ingest-card">
        <div className="hero-tag">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="10" />
          </svg>
          Semantic Code Understanding
        </div>
        <h1 className="hero-headline">Deep repository intelligence with grounded retrieval.</h1>
        <p className="hero-description">
          Parse Python & JavaScript codebases, extract AST structure and function references, and query your architecture with exact line citations.
        </p>

        <form className="ingest-form" onSubmit={ingestRepository}>
          <input
            className="ingest-input"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="Enter public GitHub URL or local repository path..."
            disabled={loadingIngest}
          />
          <button className="primary-btn" type="submit" disabled={loadingIngest || !source.trim()}>
            {loadingIngest ? (
              <>
                <span className="pulse-circle" style={{ background: '#fff' }} />
                <span>Ingesting...</span>
              </>
            ) : (
              <>
                <span>Ingest Repository</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </>
            )}
          </button>
        </form>

        {loadingIngest && (
          <div className="status-banner">
            <div className="status-row">
              <div className="status-label">
                <span className="pulse-circle" />
                <span>Processing Codebase</span>
              </div>
              <span style={{ fontSize: '0.86rem', color: 'var(--ink-secondary)', fontWeight: 600 }}>
                {Math.max(ingestStatus?.elapsed_seconds ?? 0, localIngestElapsed)}s • {ingestStatus?.progress ?? 0}%
              </span>
            </div>
            <div className="progress-track">
              <div className="progress-bar" style={{ width: `${ingestStatus?.progress ?? 5}%` }} />
            </div>
            <p className="status-detail">{ingestStatus?.message || "Parsing repository AST and building vector index..."}</p>
          </div>
        )}
      </section>

      {/* Main Workspace Split */}
      <main className="workspace-grid">
        {/* Left Column: Chat Assistant */}
        <section className="chat-container">
          <div className="panel-header">
            <h2>Query & Analysis</h2>
            <div className={`state-indicator ${loadingQuery ? "active" : ""}`}>
              {loadingQuery ? (
                <>
                  <span className="pulse-circle" />
                  <span>Searching index...</span>
                </>
              ) : (
                <span>Ready for queries</span>
              )}
            </div>
          </div>

          <div className="messages-scroll">
            {messages.map((msg, i) => (
              <div key={i} className={`msg-bubble ${msg.role}`}>
                <span className="msg-author">{msg.role}</span>
                <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className="query-box" onSubmit={submitQuery}>
            <textarea
              className="query-textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about the codebase (e.g., 'Where is authentication handled?')..."
              rows={2}
            />
            <div className="query-bottom-bar">
              <button className="primary-btn" type="submit" disabled={loadingQuery || !query.trim()}>
                {loadingQuery ? "Thinking..." : "Ask Assistant"}
              </button>
            </div>
          </form>
        </section>

        {/* Right Column: Retrieved Evidence & Citations */}
        <aside className="context-container">
          <div className="panel-header">
            <h2>Retrieved Evidence</h2>
            <span className="state-indicator">
              {lastResult ? `${lastResult.snippets?.length || 0} snippets` : "Awaiting search"}
            </span>
          </div>

          <div className="context-scroll">
            {!lastResult && !lastIngestResult && (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <p style={{ fontWeight: 600, color: 'var(--ink-secondary)' }}>No context retrieved yet</p>
                <p style={{ fontSize: '0.84rem' }}>Ingest a repository and ask a question to view grounded code snippets and citations here.</p>
              </div>
            )}

            {lastIngestResult && !lastResult && (
              <div className="overview-box">
                <h4>Repository Overview</h4>
                <p className="overview-text">{lastIngestResult.repo_summary}</p>
                <div style={{ marginTop: '0.9rem' }}>
                  <h5 style={{ fontSize: '0.82rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ink-tertiary)', marginBottom: '0.5rem' }}>Indexed Files</h5>
                  <div className="files-cluster">
                    {lastIngestResult.file_summaries?.slice(0, 12).map((item, idx) => (
                      <span className="file-chip" key={idx}>{item.file_path}</span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {lastResult && (
              <>
                {lastResult.relevant_files?.length > 0 && (
                  <div style={{ marginBottom: '1rem' }}>
                    <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--ink-tertiary)', fontWeight: 700, display: 'block', marginBottom: '0.45rem' }}>
                      Referenced Files
                    </span>
                    <div className="files-cluster">
                      {lastResult.relevant_files.map((file, idx) => (
                        <span className="file-chip" key={idx} style={{ background: 'var(--terracotta-soft)', color: 'var(--terracotta-dark)', borderColor: 'rgba(217,93,57,0.2)' }}>
                          {file}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {lastResult.snippets?.map((snip, idx) => (
                  <article className="code-card" key={idx}>
                    <div className="code-card-header">
                      <span className="code-title">{snip.name}</span>
                      <span className="code-lines">{snip.file_path} : L{snip.line_start || 1}-{snip.line_end || '?'}</span>
                    </div>
                    <pre className="code-block">
                      <code>{snip.code}</code>
                    </pre>
                  </article>
                ))}
              </>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;

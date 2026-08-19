import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialMessages = [
  {
    role: "assistant",
    content: "Welcome. Ingest any GitHub repository or local directory to explore architecture, understand modules, and ask natural language questions with grounded source citations."
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
          throw new Error(data.detail || "Could not fetch status");
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
                `Successfully indexed ${data.result.files_indexed} files and ${data.result.chunks_indexed} code chunks.\n\n` +
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
      message: "Starting ingestion pipeline...",
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
    <div className="app-wrapper">
      {/* 1. Header with Brand & Pill Nav Links */}
      <header className="brand-nav">
        <div className="brand-meta">
          <span className="brand-logo-text">CodebaseAI</span>
          <span className="brand-subline">CODEBASE TEACHING ASSISTANT</span>
        </div>
        <div className="nav-links">
          <a
            className="nav-pill-btn"
            href="https://github.com/devokdev/Codebase-explanator"
            target="_blank"
            rel="noreferrer"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
          <div className="nav-pill-btn" style={{ cursor: 'default' }}>
            Hybrid FAISS + PG
          </div>
        </div>
      </header>

      {/* 2. Hero Statement & Neo-Brutalist Input */}
      <section className="hero-section">
        <h1 className="hero-main-title">
          CodebaseAI teaches your codebase back to you.
        </h1>
        <p className="hero-paragraph">
          Explore repositories through guided architectural submaps, file roles, and semantic connection hints.
          Explains how a module works before you dive into implementation.
        </p>

        <span className="supports-tag">Supports Python, JavaScript, and multi-file projects.</span>

        <form className="ingest-form-container" onSubmit={ingestRepository}>
          <div className="ingest-input-box">
            <span className="input-icon-dot" />
            <input
              className="neo-input"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="https://github.com/owner/repo or local/path"
              disabled={loadingIngest}
            />
          </div>
          <button className="neo-submit-btn" type="submit" disabled={loadingIngest || !source.trim()}>
            {loadingIngest ? "Analyzing..." : "Analyze"}
          </button>
        </form>

        {loadingIngest && (
          <div className="neo-progress-banner">
            <div className="progress-header">
              <span>Ingesting Codebase AST...</span>
              <span>{ingestStatus?.progress ?? 0}% ({Math.max(ingestStatus?.elapsed_seconds ?? 0, localIngestElapsed)}s)</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${ingestStatus?.progress ?? 8}%` }} />
            </div>
            <p className="progress-msg">{ingestStatus?.message || "Parsing repository AST and building vector index..."}</p>
          </div>
        )}
      </section>

      {/* 3. Main Workspace: Chat & Evidence Columns */}
      <main className="workspace-section">
        {/* Left Column: Ask the codebase */}
        <section className="neo-panel">
          <div className="panel-title-bar">
            <h2 className="panel-heading">Ask the codebase</h2>
            <span className={`status-badge ${loadingQuery ? "active" : ""}`}>
              {loadingQuery ? "Searching..." : "Ready"}
            </span>
          </div>

          <div className="chat-scroll-area">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-card ${msg.role}`}>
                <span className="author-label">{msg.role}</span>
                <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className="query-compose" onSubmit={submitQuery}>
            <textarea
              className="compose-textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Where is authentication handled? Explain the flow..."
              rows={2}
            />
            <div className="compose-actions">
              <button className="neo-submit-btn" type="submit" disabled={loadingQuery || !query.trim()} style={{ padding: '0.6rem 1.4rem', fontSize: '0.9rem' }}>
                {loadingQuery ? "Thinking..." : "Ask Assistant"}
              </button>
            </div>
          </form>
        </section>

        {/* Right Column: Retrieved Evidence & Citations */}
        <aside className="neo-panel">
          <div className="panel-title-bar">
            <h2 className="panel-heading">Retrieved Evidence</h2>
            <span className="status-badge">
              {lastResult ? `${lastResult.snippets?.length || 0} citations` : "Empty"}
            </span>
          </div>

          <div className="evidence-scroll-area">
            {!lastResult && !lastIngestResult && (
              <div className="empty-state-card">
                <div className="empty-icon-circle">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <p style={{ fontWeight: 700, color: 'var(--ink-primary)' }}>No citations retrieved yet</p>
                <p style={{ fontSize: '0.88rem' }}>Ingest a repository above and ask questions to inspect extracted source functions and lines.</p>
              </div>
            )}

            {lastIngestResult && !lastResult && (
              <div className="summary-card">
                <h4>Repository Overview</h4>
                <p>{lastIngestResult.repo_summary}</p>
                <div className="chips-cluster">
                  {lastIngestResult.file_summaries?.slice(0, 10).map((item, idx) => (
                    <span className="chip-tag" key={idx}>{item.file_path}</span>
                  ))}
                </div>
              </div>
            )}

            {lastResult && (
              <>
                {lastResult.relevant_files?.length > 0 && (
                  <div style={{ marginBottom: '1.2rem' }}>
                    <span style={{ fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--ink-subtle)', fontWeight: 800, display: 'block', marginBottom: '0.5rem' }}>
                      Referenced Source Files
                    </span>
                    <div className="chips-cluster">
                      {lastResult.relevant_files.map((file, idx) => (
                        <span className="chip-tag" key={idx} style={{ background: 'var(--terracotta-tint)', color: 'var(--terracotta-dark)', borderColor: 'var(--terracotta)' }}>
                          {file}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {lastResult.snippets?.map((snip, idx) => (
                  <article className="snippet-card" key={idx}>
                    <div className="snippet-card-header">
                      <span className="snippet-func-name">{snip.name}</span>
                      <span className="snippet-loc">{snip.file_path} : L{snip.line_start || 1}-{snip.line_end || '?'}</span>
                    </div>
                    <pre className="snippet-pre">
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

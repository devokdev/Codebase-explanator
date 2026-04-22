import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialMessages = [
  {
    role: "assistant",
    content: "Ingest a GitHub repository URL or local folder, then ask questions about the codebase in natural language."
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
      const response = await fetch(`${API_BASE_URL}/ingest/status/${ingestJobId}`);
      const data = await response.json();
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Ingestion job expired or backend was restarted. Please ingest again.");
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
              `Indexed ${data.result.files_indexed} files and ${data.result.chunks_indexed} chunks from ${data.result.source_root}.\n\n` +
              `Repo summary: ${data.result.repo_summary}`
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
    };

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

    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [ingestJobId, loadingIngest]);

  const ingestRepository = async () => {
    if (!source.trim()) return;
    setLoadingIngest(true);
    setIngestStatus({
      status: "running",
      progress: 1,
      message: "Starting ingestion...",
      elapsed_seconds: 0
    });
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source })
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

  const submitQuery = async () => {
    if (!query.trim()) return;

    const userMessage = { role: "user", content: query };
    setMessages((current) => [...current, userMessage]);
    setLoadingQuery(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 })
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
      setQuery("");
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Query error: ${error.message}` }
      ]);
    } finally {
      setLoadingQuery(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="backdrop" />
      <main className="layout">
        <section className="hero-card">
          <p className="eyebrow">AI-Powered Codebase Understanding</p>
          <h1>Repository intelligence with grounded retrieval.</h1>
          <p className="hero-copy">
            Index Python and JavaScript code, retrieve the most relevant functions and classes, and ask natural language questions with file-grounded answers.
          </p>

          <div className="ingest-panel">
            <input
              value={source}
              onChange={(event) => setSource(event.target.value)}
              placeholder="GitHub URL or local folder path"
            />
            <button onClick={ingestRepository} disabled={loadingIngest}>
              {loadingIngest ? "Ingesting..." : "Ingest Repository"}
            </button>
          </div>

          {loadingIngest && (
            <div className="ingest-status-card" aria-live="polite">
              <div className="ingest-status-head">
                <span className="status-dot" />
                <strong>Repository ingestion in progress</strong>
                <span>{Math.max(ingestStatus?.elapsed_seconds ?? 0, localIngestElapsed)}s elapsed</span>
              </div>
              <div className="progress-row">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${ingestStatus?.progress ?? 0}%` }} />
                </div>
                <span>{ingestStatus?.progress ?? 0}%</span>
              </div>
              <p>{ingestStatus?.message || "Working..."}</p>
            </div>
          )}
        </section>

        <section className="workspace">
          <div className="chat-card">
            <div className="chat-header">
              <h2>Ask the codebase</h2>
              <span>{loadingQuery ? "Searching..." : "Ready"}</span>
            </div>

            <div className="messages">
              {messages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                  <span className="message-role">{message.role}</span>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>

            <div className="query-panel">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Where is authentication handled?"
                rows={4}
              />
              <button onClick={submitQuery} disabled={loadingQuery}>
                {loadingQuery ? "Thinking..." : "Ask"}
              </button>
            </div>
          </div>

          <aside className="results-card">
            <h2>Retrieved Context</h2>
            {!lastResult && !lastIngestResult && (
              <p className="muted">Query results and file references will appear here.</p>
            )}

            {lastIngestResult && (
              <section className="ingest-results">
                <h3>Repository Overview</h3>
                <p className="repo-summary">{lastIngestResult.repo_summary}</p>

                <div className="file-summary-list">
                  {lastIngestResult.file_summaries.map((item) => (
                    <article className="file-summary-card" key={item.file_path}>
                      <strong>{item.file_path}</strong>
                      <span>{item.summary}</span>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {lastResult && (
              <>
                <div className="references">
                  <h3>Files</h3>
                  {lastResult.relevant_files.map((file) => (
                    <span className="reference-pill" key={file}>
                      {file}
                    </span>
                  ))}
                </div>

                <div className="snippets">
                  {lastResult.snippets.map((snippet, index) => (
                    <article className="snippet-card" key={`${snippet.file_path}-${index}`}>
                      <div className="snippet-meta">
                        <strong>{snippet.name}</strong>
                        <span>
                          {snippet.file_path} ({snippet.line_start}-{snippet.line_end})
                        </span>
                      </div>
                      <pre>{snippet.code}</pre>
                    </article>
                  ))}
                </div>
              </>
            )}
          </aside>
        </section>
      </main>
    </div>
  );
}

export default App;

import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialMessages = [
  {
    role: "assistant",
    content: "Ingest a repository to explore architecture, dependencies, and code logic."
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
  const ingestPollTimeoutRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loadingQuery]);

  useEffect(() => {
    if (!ingestJobId || !loadingIngest) return undefined;

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
        if (!response.ok) throw new Error(data.detail || "Status check failed");

        if (cancelled) return;
        setIngestStatus(data);

        if (data.status === "completed") {
          setLastIngestResult(data.result);
          setMessages((current) => [
            ...current,
            {
              role: "assistant",
              content: `Indexed ${data.result.files_indexed} files (${data.result.chunks_indexed} snippets).\n\n${data.result.repo_summary}`
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
            { role: "assistant", content: `Ingestion failed: ${data.error || data.message}` }
          ]);
          setLoadingIngest(false);
          setIngestJobId(null);
          stopPolling();
          return;
        }

        ingestPollTimeoutRef.current = window.setTimeout(pollStatus, 1000);
      } catch (error) {
        if (cancelled) return;
        setMessages((current) => [
          ...current,
          { role: "assistant", content: `Connection error: ${error.message}` }
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
    setIngestStatus({ status: "running", progress: 5, message: "Analyzing codebase..." });
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source.trim() })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ingestion failed");
      setIngestJobId(data.job_id);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Error: ${error.message}` }
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
      if (!response.ok) throw new Error(data.detail || "Query failed");
      setLastResult(data);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Error: ${error.message}` }
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
    <div className="canvas">
      {/* Sleek Minimal Nav */}
      <header className="header">
        <div className="logo-lockup">
          <span className="logo-symbol">◇</span>
          <span className="logo-text">Explanator</span>
        </div>
      </header>

      {/* Main Container */}
      <main className="content-wrap">
        {/* Minimal Hero Bar */}
        <section className="input-hero">
          <form className="ingest-strip" onSubmit={ingestRepository}>
            <input
              className="ingest-input"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="Paste GitHub repository URL or folder path..."
              disabled={loadingIngest}
            />
            <button className="action-btn" type="submit" disabled={loadingIngest || !source.trim()}>
              {loadingIngest ? "Indexing..." : "Index"}
            </button>
          </form>

          {loadingIngest && (
            <div className="minimal-progress">
              <div className="progress-fill" style={{ width: `${ingestStatus?.progress ?? 10}%` }} />
            </div>
          )}
        </section>

        {/* 2-Column Clean Workspace */}
        <div className="workspace-duo">
          {/* Left Column: Conversation */}
          <section className="column-panel chat-panel">
            <div className="panel-bar">
              <span className="panel-title">Conversation</span>
              {loadingQuery && <span className="status-indicator">Retrieving...</span>}
            </div>

            <div className="dialogue-flow">
              {messages.map((msg, i) => (
                <div key={i} className={`bubble ${msg.role}`}>
                  <div className="bubble-text">{msg.content}</div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <form className="compose-box" onSubmit={submitQuery}>
              <textarea
                className="compose-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the codebase..."
                rows={1}
              />
              <button className="send-btn" type="submit" disabled={loadingQuery || !query.trim()}>
                ↑
              </button>
            </form>
          </section>

          {/* Right Column: Context & Evidence */}
          <aside className="column-panel context-panel">
            <div className="panel-bar">
              <span className="panel-title">Evidence</span>
              <span className="meta-count">
                {lastResult ? `${lastResult.snippets?.length || 0} snippets` : ""}
              </span>
            </div>

            <div className="context-flow">
              {!lastResult && !lastIngestResult && (
                <div className="blank-state">
                  <span>Context citations will appear here.</span>
                </div>
              )}

              {lastIngestResult && !lastResult && (
                <div className="summary-block">
                  <p className="summary-lead">{lastIngestResult.repo_summary}</p>
                  <div className="chips-wrapper">
                    {lastIngestResult.file_summaries?.slice(0, 10).map((f, idx) => (
                      <span className="file-tag" key={idx}>{f.file_path}</span>
                    ))}
                  </div>
                </div>
              )}

              {lastResult && (
                <>
                  {lastResult.relevant_files?.length > 0 && (
                    <div className="chips-wrapper">
                      {lastResult.relevant_files.map((file, idx) => (
                        <span className="file-tag highlight" key={idx}>{file}</span>
                      ))}
                    </div>
                  )}

                  {lastResult.snippets?.map((snip, idx) => (
                    <div className="snippet-block" key={idx}>
                      <div className="snippet-head">
                        <span className="snippet-symbol">{snip.name}</span>
                        <span className="snippet-loc">{snip.file_path} : {snip.line_start || 1}-{snip.line_end || '?'}</span>
                      </div>
                      <pre className="snippet-code">
                        <code>{snip.code}</code>
                      </pre>
                    </div>
                  ))}
                </>
              )}
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;

import { useEffect, useMemo, useState } from "react";
import {
  Plus, Search, Settings2, Server, Send, Copy, Check,
  ChevronDown, ChevronUp, BookOpen, Clock3, Database,
  Menu, X, Circle, Trash2
} from "lucide-react";
import { askRag, healthCheck } from "./services/api";

const QUESTIONS = [
  "What is a Python class?",
  "How does inheritance work in Python?",
  "What is a lambda expression?",
  "How does Python handle exceptions?",
  "How can a list be used as a stack?",
  "What is the difference between a list and a tuple?",
  "How are dictionaries used in Python?",
  "How does Python search for modules?",
  "How can data be written to a file in Python?",
  "What are generator expressions in Python?"
];

function App() {
  const [backend, setBackend] = useState("checking");
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [topK, setTopK] = useState(5);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chats, setChats] = useState([{ id: 1, title: "New conversation" }]);
  const [activeChat, setActiveChat] = useState(1);
  const [loading, setLoading] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const activeMessages = useMemo(
    () => messages.filter(m => m.chatId === activeChat),
    [messages, activeChat]
  );

  async function checkBackend() {
    setBackend("checking");
    try {
      await healthCheck(baseUrl);
      setBackend("online");
    } catch {
      setBackend("offline");
    }
  }

  useEffect(() => { checkBackend(); }, []);

  function newChat() {
    const id = Date.now();
    setChats(prev => [{ id, title: "New conversation" }, ...prev]);
    setActiveChat(id);
    setMessages(prev => prev.filter(m => m.chatId !== activeChat));
    setQuestion("");
    setMobileOpen(false);
  }

  function chooseQuestion(q) {
    setQuestion(q);
    setMobileOpen(false);
  }

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    const userMsg = { id: Date.now(), chatId: activeChat, role: "user", text: q };
    setMessages(prev => [...prev, userMsg]);

    setChats(prev => prev.map(c =>
      c.id === activeChat && c.title === "New conversation"
        ? { ...c, title: q.length > 34 ? q.slice(0, 34) + "…" : q }
        : c
    ));
    setQuestion("");

    try {
      const result = await askRag(q, topK, baseUrl);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        chatId: activeChat,
        role: "assistant",
        data: result
      }]);
      setBackend("online");
    } catch (e) {
      setBackend("offline");
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        chatId: activeChat,
        role: "error",
        text: e.message
      }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <strong>Cost-Efficient</strong>
            <span>RAG</span>
          </div>
          <button className="icon-btn mobile-close" onClick={() => setMobileOpen(false)}><X size={18}/></button>
        </div>

        <button className="new-chat" onClick={newChat}>
          <Plus size={17}/> New chat
        </button>

        <div className="side-label">Recent chats</div>
        <div className="chat-list">
          {chats.map(chat => (
            <button
              key={chat.id}
              className={`chat-item ${chat.id === activeChat ? "active" : ""}`}
              onClick={() => { setActiveChat(chat.id); setMobileOpen(false); }}
            >
              <span>{chat.title}</span>
            </button>
          ))}
        </div>

        <div className="side-divider"/>

        <div className="side-label">Connection</div>
        <div className="setting-block">
          <label>Backend URL</label>
          <input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} onBlur={checkBackend}/>
        </div>

        <div className="setting-block">
          <div className="range-head"><label>Top-K chunks</label><b>{topK}</b></div>
          <input className="range" type="range" min="1" max="10" value={topK} onChange={e => setTopK(e.target.value)}/>
        </div>

        <button className="backend-status" onClick={checkBackend}>
          <Circle size={9} fill="currentColor"/>
          <span>Backend {backend}</span>
          <Server size={15}/>
        </button>

        <div className="sidebar-footer">
          <div><Database size={14}/> ChromaDB</div>
          <div><span className="ollama-dot"/> Ollama local LLM</div>
          <small>No paid API required</small>
        </div>
      </aside>

      {mobileOpen && <div className="overlay" onClick={() => setMobileOpen(false)}/>}

      <main className="main">
        <header className="topbar">
          <button className="icon-btn mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20}/></button>
          <div className="top-title">
            <span className="eyebrow">LOCAL DOCUMENTATION ASSISTANT</span>
            <h1>Cost-Efficient RAG</h1>
          </div>
          <div className={`status-pill ${backend}`}>
            <Circle size={8} fill="currentColor"/>
            {backend === "online" ? "Connected" : backend === "checking" ? "Checking" : "Offline"}
          </div>
        </header>

        <section className="content">
          {activeMessages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-kicker"><BookOpen size={15}/> Python documentation</div>
              <h2>Ask something. <em>Get a grounded answer.</em></h2>
              <p className="welcome-copy">
                Search the indexed documentation through your local RAG pipeline.
                Answers are generated by Ollama and supported by retrieved source chunks.
              </p>

              <div className="suggestions">
                <div className="section-title">Try a question</div>
                <div className="question-grid">
                  {QUESTIONS.map(q => (
                    <button key={q} onClick={() => chooseQuestion(q)}>{q}</button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="conversation">
              {activeMessages.map(m => <Message key={m.id} message={m}/>)}
              {loading && (
                <div className="assistant-row">
                  <div className="avatar">R</div>
                  <div className="typing"><span/><span/><span/></div>
                </div>
              )}
            </div>
          )}

          <div className="composer-wrap">
            <div className="composer">
              <textarea
                value={question}
                onChange={e => setQuestion(e.target.value)}
                onKeyDown={e => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit();
                }}
                placeholder="Ask a question about the indexed documentation..."
                rows={1}
              />
              <button className="send-btn" onClick={submit} disabled={!question.trim() || loading}>
                <Send size={17}/>
              </button>
            </div>
            <div className="composer-meta">
              <span>Ctrl + Enter to submit</span>
              <span>Top-K: {topK}</span>
            </div>
          </div>
        </section>

        <footer className="footer">
          <span>FastAPI</span><i>·</i><span>ChromaDB</span><i>·</i><span>Ollama</span>
          <span className="footer-right">Local inference • No external LLM API</span>
        </footer>
      </main>
    </div>
  );
}

function Message({ message }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return <div className="user-row"><div className="user-bubble">{message.text}</div></div>;
  }

  if (message.role === "error") {
    return <div className="error-box"><strong>Request failed</strong><span>{message.text}</span></div>;
  }

  const d = message.data || {};
  const citations = d.citations || [];
  const answer = d.answer || "No answer returned.";

  async function copyAnswer() {
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="assistant-row">
      <div className="avatar">R</div>
      <div className="answer-card">
        <div className="answer-head">
          <span>RAG ANSWER</span>
          <button onClick={copyAnswer}>{copied ? <Check size={14}/> : <Copy size={14}/>} {copied ? "Copied" : "Copy"}</button>
        </div>
        <div className="answer-text">{answer}</div>

        <div className="metrics">
          <div><Clock3 size={13}/><span>Retrieval</span><b>{formatMs(d.retrieval_latency_ms)}</b></div>
          <div><Clock3 size={13}/><span>Generation</span><b>{formatMs(d.generation_latency_ms)}</b></div>
          <div><Database size={13}/><span>Chunks</span><b>{d.retrieved_chunks ?? citations.length}</b></div>
        </div>

        <button className="sources-toggle" onClick={() => setOpen(!open)}>
          <span>Sources ({citations.length})</span>{open ? <ChevronUp size={15}/> : <ChevronDown size={15}/>}
        </button>

        {open && (
          <div className="sources">
            {citations.map((c, i) => (
              <div className="source" key={i}>
                <span className="source-num">{c.citation ?? i + 1}</span>
                <div>
                  <b>{c.source || "Document"}</b>
                  <span>Chunk {c.chunk_index ?? "—"} · distance {typeof c.distance === "number" ? c.distance.toFixed(4) : "—"}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function formatMs(v) {
  if (v == null) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? `${n.toFixed(0)} ms` : "—";
}

export default App;
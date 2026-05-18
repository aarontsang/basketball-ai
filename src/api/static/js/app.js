const API_BASE = "/api/v1";

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusPill = document.getElementById("status-pill");

let isLoading = false;

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    if (data.agent_ready) {
      statusPill.textContent = `Online · ${data.ollama_model}`;
      statusPill.className = "status-pill ready";
    } else {
      statusPill.textContent = "Starting agent…";
      statusPill.className = "status-pill";
    }
  } catch {
    statusPill.textContent = "API offline";
    statusPill.className = "status-pill error";
  }
}

function appendMessage(role, text, extraClass = "") {
  const wrap = document.createElement("div");
  wrap.className = `message ${role} ${extraClass}`.trim();

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function removeMessage(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  input.disabled = loading;
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || isLoading) return;

  appendMessage("user", trimmed);
  input.value = "";
  setLoading(true);

  const loadingEl = appendMessage("assistant", "Thinking…", "loading");

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: trimmed, max_iterations: 25 }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      removeMessage(loadingEl);
      appendMessage("assistant", `Error: ${detail}`, "error");
      return;
    }

    removeMessage(loadingEl);
    appendMessage("assistant", data.response || "(No response)");
  } catch (err) {
    removeMessage(loadingEl);
    appendMessage(
      "assistant",
      `Could not reach the server. Is uvicorn running?\n\n${err.message}`,
      "error"
    );
  } finally {
    setLoading(false);
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const prompt = btn.getAttribute("data-prompt");
    if (prompt) {
      input.value = prompt;
      input.focus();
    }
  });
});

checkHealth();
setInterval(checkHealth, 30000);

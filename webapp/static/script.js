const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const settingsToggle = document.getElementById("settingsToggle");
const settingsPanel = document.getElementById("settingsPanel");
const temperatureEl = document.getElementById("temperature");
const tempValue = document.getElementById("tempValue");
const newCharsEl = document.getElementById("newChars");
const lenValue = document.getElementById("lenValue");
const modelInfoEl = document.getElementById("modelInfo");
const newChatBtn = document.getElementById("newChatBtn");

// Running plain-text transcript sent to the model as its generation prompt.
let transcript = "";
let busy = false;

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addRow(role, text, opts = {}) {
  emptyState.style.display = "none";

  const row = document.createElement("div");
  row.className = `row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "✦";

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (opts.error ? " error" : "");
  bubble.textContent = text;

  if (opts.note) {
    const note = document.createElement("span");
    note.className = "note";
    note.textContent = opts.note;
    bubble.appendChild(note);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return bubble;
}

function addTypingIndicator() {
  const bubble = addRow("assistant", "");
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  return bubble;
}

function typewriter(bubble, text) {
  bubble.textContent = "";
  let i = 0;
  const step = Math.max(1, Math.floor(text.length / 200)); // finish in ~200 ticks max
  return new Promise((resolve) => {
    const timer = setInterval(() => {
      i += step;
      bubble.textContent = text.slice(0, i);
      scrollToBottom();
      if (i >= text.length) {
        bubble.textContent = text;
        clearInterval(timer);
        resolve();
      }
    }, 12);
  });
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || busy) return;

  busy = true;
  sendBtn.disabled = true;
  inputEl.value = "";
  resizeInput();

  addRow("user", message);
  transcript += `You: ${message}\nGPT: `;

  const typingBubble = addTypingIndicator();

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: transcript,
        message: message,
        temperature: parseFloat(temperatureEl.value),
        new_chars: parseInt(newCharsEl.value, 10),
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      typingBubble.classList.add("error");
      typingBubble.textContent = data.error || "Something went wrong.";
      transcript = transcript.slice(0, -`GPT: `.length); // don't leave a dangling "GPT: "
      return;
    }

    const reply = data.reply.trim() || "(model produced no output — try a lower temperature)";
    await typewriter(typingBubble, reply);
    transcript += data.reply + "\n";

    if (data.dropped_chars && data.dropped_chars.length) {
      const note = document.createElement("span");
      note.className = "note";
      note.textContent = `Ignored characters not in the training vocabulary: ${data.dropped_chars.join(" ")}`;
      typingBubble.appendChild(note);
    }
  } catch (err) {
    typingBubble.classList.add("error");
    typingBubble.textContent = "Could not reach the server. Is webapp/app.py running?";
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener("input", resizeInput);

settingsToggle.addEventListener("click", () => {
  settingsPanel.classList.toggle("open");
});

temperatureEl.addEventListener("input", () => {
  tempValue.textContent = temperatureEl.value;
});

newCharsEl.addEventListener("input", () => {
  lenValue.textContent = newCharsEl.value;
});

newChatBtn.addEventListener("click", () => {
  transcript = "";
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  emptyState.style.display = "block";
});

async function loadModelInfo() {
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    if (!data.ready) {
      modelInfoEl.textContent = "No checkpoint found.\nRun: python train.py";
      sendBtn.disabled = true;
      return;
    }
    modelInfoEl.textContent =
      `device: ${data.device}\n` +
      `params: ${data.num_params.toLocaleString()}\n` +
      `vocab: ${data.vocab_size}  ctx: ${data.context_length}\n` +
      `blocks: ${data.num_blocks}  heads: ${data.num_heads}`;
  } catch (err) {
    modelInfoEl.textContent = "Could not reach server.";
  }
}

loadModelInfo();
resizeInput();
inputEl.focus();

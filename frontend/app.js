const API = "/api/v1";
const state = {
  documents: [],
  activeDocument: localStorage.getItem("paperlens.activeDocument") || "",
  sessionId: localStorage.getItem("paperlens.sessionId") || crypto.randomUUID(),
  busy: false,
};
localStorage.setItem("paperlens.sessionId", state.sessionId);

const $ = (selector) => document.querySelector(selector);
const els = {
  status: $("#system-status"), statusDot: $("#status-dot"), paperCount: $("#paper-count"),
  documentList: $("#document-list"), refreshDocs: $("#refresh-docs"), pdfInput: $("#pdf-input"),
  uploadZone: $("#upload-zone"), activeTitle: $("#active-title"), activeBadge: $("#active-badge"),
  scope: $("#scope-label"), question: $("#question-input"), form: $("#query-form"),
  send: $("#send-button"), conversation: $("#conversation"), empty: $("#empty-state"),
  evidenceList: $("#evidence-list"), evidenceSummary: $("#evidence-summary"),
  pipeline: $(".pipeline-card"), pipelineTime: $("#pipeline-time"), topK: $("#top-k"),
  overlay: $("#processing-overlay"), processingFile: $("#processing-file"),
  progress: $("#progress-bar"), processingSteps: [...document.querySelectorAll("#processing-steps li")],
  toast: $("#toast"), theme: $("#theme-toggle"),
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("show"), 3200);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
  return data;
}

async function checkHealth() {
  try {
    const health = await request(`${API}/health`);
    els.status.textContent = health.status === "healthy" ? "Research index connected" : "Service needs attention";
    els.statusDot.className = `status-dot ${health.status === "healthy" ? "ready" : "error"}`;
  } catch {
    els.status.textContent = "Service unavailable";
    els.statusDot.className = "status-dot error";
  }
}

async function loadDocuments() {
  els.documentList.innerHTML = '<div class="skeleton-card"></div><div class="skeleton-card"></div>';
  try {
    const data = await request(`${API}/documents`);
    state.documents = data.documents || [];
    if (state.activeDocument && !state.documents.some(d => d.source_file === state.activeDocument)) state.activeDocument = "";
    renderDocuments();
  } catch (error) {
    els.documentList.innerHTML = `<div class="empty-library">${escapeHtml(error.message)}</div>`;
  }
}

function renderDocuments() {
  els.paperCount.textContent = state.documents.length;
  if (!state.documents.length) {
    els.documentList.innerHTML = '<div class="empty-library">No papers yet. Import a PDF to begin.</div>';
    selectDocument("");
    return;
  }
  els.documentList.innerHTML = state.documents.map(doc => `
    <button class="document-card ${doc.source_file === state.activeDocument ? "active" : ""}" type="button" data-document="${escapeHtml(doc.source_file)}">
      <span class="doc-icon">PDF</span>
      <span><strong title="${escapeHtml(doc.source_file)}">${escapeHtml(doc.source_file.replace(/\.pdf$/i,""))}</strong><small>${doc.chunk_count} indexed chunks</small></span>
    </button>`).join("");
  document.querySelectorAll("[data-document]").forEach(button => button.addEventListener("click", () => selectDocument(button.dataset.document)));
  if (state.activeDocument) applyActiveDocument();
}

function selectDocument(name) {
  state.activeDocument = name;
  localStorage.setItem("paperlens.activeDocument", name);
  renderDocuments();
  applyActiveDocument();
}

function applyActiveDocument() {
  const name = state.activeDocument;
  els.activeTitle.textContent = name ? name.replace(/\.pdf$/i, "") : "Select a paper to begin";
  els.activeBadge.classList.toggle("active", Boolean(name));
  els.activeBadge.innerHTML = `<span></span>${name ? "Scoped to this paper" : "No paper selected"}`;
  els.scope.textContent = name ? `Scope: ${name}` : "Scope: no paper selected";
  els.question.disabled = !name;
  els.send.disabled = !name || state.busy;
  els.question.placeholder = name ? "Ask about methods, equations, figures, or findings…" : "Select a paper, then ask a question…";
}

function addMessage(role, content, meta = "") {
  if (els.empty) { els.empty.remove(); els.empty = null; }
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `<div class="message-bubble"><span class="message-label">${role === "user" ? "You" : "PaperLens"}</span><div class="answer-text">${escapeHtml(content)}</div>${meta ? `<div class="answer-meta">${meta}</div>` : ""}</div>`;
  els.conversation.appendChild(wrapper);
  els.conversation.scrollTop = els.conversation.scrollHeight;
  return wrapper;
}

function addThinking() {
  const item = addMessage("assistant", "");
  item.querySelector(".answer-text").innerHTML = '<div class="thinking"><i></i><i></i><i></i></div>';
  return item;
}

function renderEvidence(chunks = []) {
  els.evidenceSummary.hidden = chunks.length > 0;
  els.evidenceList.innerHTML = chunks.map((chunk, index) => {
    const type = chunk.content_type || "text";
    const labels = { text: "Text", formula: "Equation", figure: "Figure" };
    const score = Number(chunk.score || 0);
    return `<details class="evidence-card" ${index === 0 ? "open" : ""}>
      <summary>
        <div class="evidence-card-head"><span class="type-chip ${type}">${labels[type] || type}</span><span class="score">Score ${score.toFixed(4)}</span></div>
        <h3>${escapeHtml(chunk.source_file)}${chunk.page_number ? ` · Page ${chunk.page_number}` : ""}</h3>
        <p>${escapeHtml(chunk.content_preview)}</p>
      </summary>
      <div class="evidence-full">${escapeHtml(chunk.full_content)}</div>
    </details>`;
  }).join("");
}

async function askQuestion(question) {
  if (!state.activeDocument || !question.trim() || state.busy) return;
  state.busy = true; applyActiveDocument();
  addMessage("user", question.trim());
  els.question.value = ""; autoResize();
  const thinking = addThinking();
  els.pipeline.classList.add("running");
    els.pipelineTime.textContent = "Running";
  try {
    const data = await request(`${API}/query`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ question: question.trim(), session_id: state.sessionId, top_k: Number(els.topK.value), source_file: state.activeDocument, metadata: {source:"paperlens_ui"} }),
    });
    thinking.remove();
    addMessage("assistant", data.answer, `<span>${data.chunks_retrieved} sources</span><span>${data.processing_time_seconds}s</span><span>${escapeHtml(data.intent)}</span>`);
    renderEvidence(data.retrieved_chunks);
    els.pipelineTime.textContent = `${data.processing_time_seconds}s · ${data.chunks_retrieved} sources`;
  } catch (error) {
    thinking.remove(); addMessage("assistant", `I couldn't complete that request: ${error.message}`); showToast(error.message);
    els.pipelineTime.textContent = "Request failed";
  } finally {
    state.busy = false; applyActiveDocument(); els.pipeline.classList.remove("running"); els.question.focus();
  }
}

function autoResize() { els.question.style.height = "auto"; els.question.style.height = `${Math.min(els.question.scrollHeight,150)}px`; }

function updateUploadProgress(step) {
  const widths = [18, 42, 72, 92];
  els.progress.style.width = `${widths[Math.min(step,3)]}%`;
  els.processingSteps.forEach((item, index) => item.classList.toggle("active", index <= step));
}

async function uploadPdf(file) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) return showToast("Please choose a PDF file");
  els.overlay.hidden = false; els.processingFile.textContent = file.name; updateUploadProgress(0);
  let step = 0;
  const timer = setInterval(() => { step = Math.min(step + 1, 3); updateUploadProgress(step); }, 18000);
  const body = new FormData(); body.append("file", file);
  try {
    const result = await request(`${API}/upload`, {method:"POST", body});
    updateUploadProgress(3);
    showToast(`Indexed ${result.chunks_indexed} chunks · ${result.formulas_detected} equations · ${result.figures_analyzed} figures analyzed`);
    state.activeDocument = result.file_name; localStorage.setItem("paperlens.activeDocument", result.file_name);
    await loadDocuments(); applyActiveDocument();
  } catch (error) { showToast(error.message); }
  finally { clearInterval(timer); els.overlay.hidden = true; els.pdfInput.value = ""; }
}

els.form.addEventListener("submit", event => { event.preventDefault(); askQuestion(els.question.value); });
els.question.addEventListener("input", autoResize);
els.question.addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); els.form.requestSubmit(); } });
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => { if (!state.activeDocument) return showToast("Select a paper from the library first"); askQuestion(button.dataset.question); }));
els.pdfInput.addEventListener("change", () => uploadPdf(els.pdfInput.files[0]));
els.refreshDocs.addEventListener("click", loadDocuments);
["dragenter","dragover"].forEach(name => els.uploadZone.addEventListener(name, event => { event.preventDefault(); els.uploadZone.classList.add("dragging"); }));
["dragleave","drop"].forEach(name => els.uploadZone.addEventListener(name, event => { event.preventDefault(); els.uploadZone.classList.remove("dragging"); }));
els.uploadZone.addEventListener("drop", event => uploadPdf(event.dataTransfer.files[0]));
els.theme.addEventListener("click", () => { document.body.classList.toggle("dark"); localStorage.setItem("paperlens.theme", document.body.classList.contains("dark") ? "dark" : "light"); });
if (localStorage.getItem("paperlens.theme") === "dark") document.body.classList.add("dark");

checkHealth(); loadDocuments();

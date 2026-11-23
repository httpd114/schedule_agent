"use strict";

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const statusDot = document.querySelector(".status-dot");
const toolList = document.getElementById("toolList");
const toolCount = document.getElementById("toolCount");

const TOOL_LABELS = {
  create_schedule: "创建日程",
  query_schedules: "查询日程",
  create_todo: "创建待办",
  query_todos: "查询待办",
  search_documents: "文档检索",
  delete_schedule: "取消日程（软删除）",
  aggregate_context: "上下文聚合",
};

let history = [];
let threadId = null;
let busy = false;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(text) {
  const lines = String(text || "").split(/\r?\n/);
  let html = "";
  let listType = null;
  let items = [];

  const flush = () => {
    if (!items.length) return;
    html += `<${listType}>` + items.map((item) => `<li>${item}</li>`).join("") + `</${listType}>`;
    items = [];
    listType = null;
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flush(); continue; }

    let match = line.match(/^(#{1,4})\s+(.*)$/);
    if (match) {
      flush();
      const level = Math.min(match[1].length + 2, 5);
      html += `<h${level}>${inline(match[2])}</h${level}>`;
      continue;
    }
    match = line.match(/^[-*]\s+(.*)$/);
    if (match) {
      if (listType && listType !== "ul") flush();
      listType = "ul";
      items.push(inline(match[1]));
      continue;
    }
    match = line.match(/^\d+\.\s+(.*)$/);
    if (match) {
      if (listType && listType !== "ol") flush();
      listType = "ol";
      items.push(inline(match[1]));
      continue;
    }
    flush();
    html += `<p>${inline(line)}</p>`;
  }
  flush();
  return html;
}

function scrollBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "A";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  scrollBottom();
  return bubble;
}

function hideWelcome() {
  const welcome = chatEl.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function plansHtml(result) {
  const plans = result.plans || [];
  const cards = plans
    .map((plan) => {
      const steps = (plan.steps || []).map((step) => escapeHtml(step.tool)).join(" → ");
      const risks = (plan.risks || []).map((risk) => `<li>${escapeHtml(risk)}</li>`).join("");
      const recommended = plan.recommended ? '<span class="badge">推荐</span>' : "";
      const advice = plan.recommendation
        ? `<div class="plan-advice">${escapeHtml(plan.recommendation)}</div>`
        : "";
      // 打分/冲突数属于调试信息，界面不再展示；只呈现方案价值与代价
      return `<div class="plan-card">
        <div class="plan-title">${escapeHtml(plan.title)} ${recommended}</div>
        ${advice}
        <div class="plan-meta">代价 ${escapeHtml(plan.cost || "—")} · 工具链：${steps || "无"}</div>
        ${risks ? `<ul class="plan-risks">${risks}</ul>` : ""}
        <button class="plan-btn" data-action="approve" data-plan-id="${escapeHtml(plan.id)}">执行此方案</button>
      </div>`;
    })
    .join("");
  return `<div class="plan-box">
    <div class="plan-header"><strong>方案确认</strong><span>${escapeHtml(result.summary || "高风险操作需确认")}</span></div>
    <div class="plans">${cards}</div>
    <div class="plan-actions">
      <button class="plan-btn ghost" data-action="reject">否决并重新规划</button>
      <button class="plan-btn ghost" data-action="modify">提出修改意见</button>
    </div>
  </div>`;
}

function degradedHtml(result) {
  const steps = (result.pending_steps || []).map((name) => escapeHtml(name)).join("、") || "无";
  return `<div class="plan-box degraded">
    <div class="plan-header"><strong>需要二次确认</strong><span>${escapeHtml(result.question || "")}</span></div>
    <p class="plan-advice">${escapeHtml(result.warning || "")}</p>
    <div class="plan-meta">待执行步骤：${steps}</div>
    <div class="plan-actions">
      <button class="plan-btn" data-action="continue">继续（已知校验不完整）</button>
      <button class="plan-btn ghost" data-action="abort">取消，不写入</button>
    </div>
  </div>`;
}

function bindPlanButtons(bubble) {
  bubble.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (busy) return;
      if (action === "modify") {
        const comment = window.prompt("请输入修改意见：");
        if (!comment) return;
        send("", { action: "modify", comment });
      } else if (action === "reject") {
        send("", { action: "reject", comment: "用户否决，请重新推演方案" });
      } else if (action === "continue") {
        send("", { action: "continue" });
      } else if (action === "abort") {
        send("", { action: "abort" });
      } else {
        send("", { action: "approve", plan_id: button.dataset.planId || null });
      }
    });
  });
}

async function send(rawText, decision = null) {
  if (busy) return;
  const isResume = decision !== null;
  const message = isResume ? "" : String(rawText || inputEl.value || "").trim();
  if (!isResume && !message) return;

  busy = true;
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;
  hideWelcome();

  if (!isResume) addMessage("user", escapeHtml(message).replace(/\n/g, "<br>"));
  const bubble = addMessage(
    "assistant",
    '<div class="thinking"><span class="dots"><i></i><i></i><i></i></span>正在识别意图、推演并校验方案...</div>'
  );

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history, thread_id: threadId, decision }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

    threadId = data.thread_id || threadId;
    if (data.status === "awaiting_confirmation") {
      bubble.innerHTML = data.kind === "degraded_confirm" ? degradedHtml(data) : plansHtml(data);
      bindPlanButtons(bubble);
    } else {
      const chosen = data.selected_plan?.recommendation
        ? `<div class="selected-plan">已执行：${escapeHtml(data.selected_plan.title)} · ${escapeHtml(data.selected_plan.recommendation)}</div>`
        : "";
      // 工具 trace、文档权重、打分明细均不渲染，用户侧只看结论
      bubble.innerHTML = chosen + renderMarkdown(data.answer || "（未生成回答）");
      if (!isResume) {
        history.push({ role: "user", content: message });
        history.push({ role: "assistant", content: data.answer || "" });
        if (history.length > 20) history = history.slice(-20);
      }
    }
  } catch (error) {
    bubble.innerHTML = `<p class="error">请求失败：${escapeHtml(error.message)}</p>`;
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

async function loadMeta() {
  try {
    const health = await (await fetch("/api/health")).json();
    statusText.textContent = `智谱 GLM 已连接 · ${health.tools.length} 个工具可用`;
    statusDot.classList.add("online");
    toolCount.textContent = String(health.tools.length);
    toolList.innerHTML = health.tools
      .map((name) => `<li><code>${escapeHtml(name)}</code> · ${escapeHtml(TOOL_LABELS[name] || name)}</li>`)
      .join("");
  } catch (error) {
    statusText.textContent = "服务连接失败";
  }
}

sendBtn.addEventListener("click", () => send());
clearBtn.addEventListener("click", () => { threadId = null; history = []; location.reload(); });

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

document.querySelectorAll("[data-q]").forEach((button) => {
  button.addEventListener("click", () => send(button.dataset.q));
});

loadMeta();
inputEl.focus();

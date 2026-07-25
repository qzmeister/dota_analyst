"use strict";

const API = "";
const LS_KEY = "dota_analyst_leagues";
const LS_WATCH = "dota_analyst_watchlist";
const LS_FAVORITES = "dota_analyst_favorite_teams";
let LEAGUES = [];
let SELECTED = new Set(JSON.parse(localStorage.getItem(LS_KEY) || "[]"));
let WATCHLIST = JSON.parse(localStorage.getItem(LS_WATCH) || "[]"); // [{id, title?}]
let FAVORITE_TEAMS = new Set(JSON.parse(localStorage.getItem(LS_FAVORITES) || "[]"));
let refreshTimer = null;
const COLUMN_LIMIT = 10;
const EXPANDED_COLUMNS = new Set();
const OPEN_POSTMATCH_DETAILS = new Set();
const SEEN_LIVE = new Set(JSON.parse(sessionStorage.getItem("dota_analyst_seen_live") || "[]"));

const $ = (id) => document.getElementById(id);

// ---------------- League picker ----------------
async function loadLeagues() {
  try {
    const r = await fetch(`${API}/api/leagues`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    LEAGUES = Array.isArray(d.leagues) ? d.leagues : [];
    renderLeagueList();
    updateLeagueCount();
  } catch (e) {
    setStatus("Не удалось загрузить список лиг: " + e.message);
  }
}

const REFRESH_FAST = 15000;   // 15s when live cards present
const REFRESH_SLOW = 60000;   // 60s otherwise

function renderLeagueList(filter = "") {
  const list = $("leagueList");
  const f = filter.toLowerCase();
  list.innerHTML = "";

  // Backend now returns ONLY active leagues (those with upcoming or live matches),
  // so we render a single flat list without status groups.
  const items = LEAGUES.filter((l) => l.title.toLowerCase().includes(f));

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "league-group-header";
    empty.textContent = "Нет активных лиг";
    list.appendChild(empty);
    return;
  }

  const header = document.createElement("div");
  header.className = "league-group-header league-group-live";
  header.textContent = `🟢 Активные лиги (${items.length})`;
  list.appendChild(header);

  items.forEach((l) => {
    const item = document.createElement("label");
    item.className = "league-item league-live";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = SELECTED.has(l.id);
    cb.onchange = () => {
      cb.checked ? SELECTED.add(l.id) : SELECTED.delete(l.id);
      persist();
      updateLeagueCount();
      refresh();
    };
    const span = document.createElement("span");
    span.textContent = l.title;
    item.appendChild(cb);
    item.appendChild(span);
    list.appendChild(item);
  });
}

function persist() {
  localStorage.setItem(LS_KEY, JSON.stringify([...SELECTED]));
  localStorage.setItem(LS_WATCH, JSON.stringify(WATCHLIST));
  localStorage.setItem(LS_FAVORITES, JSON.stringify([...FAVORITE_TEAMS]));
}
function updateLeagueCount() {
  $("leagueCount").textContent = SELECTED.size;
  $("watchCount").textContent = WATCHLIST.length;
}

// extract steam id from "https://dltv.org/live/8910670427.json", "/live/8910670427.json", "8910670427", or bare digits
function extractSteamId(input) {
  const s = (input || "").trim();
  const m = s.match(/(\d{9,11})/);
  return m ? m[1] : null;
}
function addWatch(raw) {
  const id = extractSteamId(raw);
  if (!id) { alert("Не удалось распознать steam_id"); return; }
  if (WATCHLIST.some(w => String(w.id) === id)) return;
  WATCHLIST.push({ id });
  persist(); updateLeagueCount(); renderWatchList(); refresh();
}
function removeWatch(id) {
  WATCHLIST = WATCHLIST.filter(w => String(w.id) !== String(id));
  persist(); updateLeagueCount(); renderWatchList(); refresh();
}
function renderWatchList() {
  const list = $("watchList");
  list.innerHTML = "";
  WATCHLIST.forEach(w => {
    const item = document.createElement("div");
    item.className = "league-item watch-item";
    item.innerHTML = `<span style="flex:1;overflow:hidden;text-overflow:ellipsis">${w.title || ("match " + w.id)}</span>`;
    const x = document.createElement("button");
    x.className = "btn-ghost";
    x.textContent = "×";
    x.onclick = () => removeWatch(w.id);
    item.appendChild(x);
    list.appendChild(item);
  });
}

// ---------------- Board ----------------
async function refresh() {
  setStatus("Загрузка…");
  try {
    const ids = [...SELECTED].join(",");
    const watch = WATCHLIST.map(w => w.id).join(",");
    const url = `${API}/api/board?events=${ids}&watch=${watch}`;
    const r = await fetch(url);
    const d = await r.json();
    renderColumn("prematch", d.prematch, prematchCard);
    renderColumn("live", d.live, liveCard);
    renderColumn("postmatch", d.postmatch, postmatchCard);
    notifyAboutLiveMatches(d.live || []);
    loadModelStatus();
    loadAnalytics();
    loadDataQuality();
    // enrich watchlist titles from fetched live cards (best-effort)
    const byMatchId = {};
    (d.live || []).forEach(c => { if (c.match_id) byMatchId[String(c.match_id)] = c; });
    let changed = false;
    WATCHLIST.forEach(w => {
      const c = byMatchId[String(w.id)];
      if (c && !w.title) {
        w.title = `${c.radiant_team?.name || "?"} vs ${c.dire_team?.name || "?"}`;
        changed = true;
      }
    });
    if (changed) { persist(); renderWatchList(); }
    setStatus(`Обновлено ${new Date().toLocaleTimeString()} · лиг: ${SELECTED.size} · лайв: ${(d.live||[]).length} · прематч: ${(d.prematch||[]).length}`);
    // adaptive refresh: faster when there are live matches
    const wantFast = (d.live || []).length > 0;
    if (wantFast !== _lastFast) {
      _lastFast = wantFast;
      if ($("autoRefresh").checked) setupAutoRefresh();
    }
  } catch (e) {
    setStatus("Ошибка загрузки: " + e.message);
  }
}
let _lastFast = false;

function renderColumn(name, items, builder) {
  const col = $("col-" + name);
  col.classList.add("is-refreshing");
  $("cnt-" + name).textContent = (items || []).length;
  col.innerHTML = "";
  if (!items || items.length === 0) {
    col.innerHTML = `<div class="empty">Нет матчей</div>`;
    requestAnimationFrame(() => col.classList.remove("is-refreshing"));
    return;
  }
  const expanded = EXPANDED_COLUMNS.has(name);
  const visibleItems = expanded ? items : items.slice(0, COLUMN_LIMIT);
  visibleItems.forEach((it) => col.appendChild(builder(it)));

  if (items.length > COLUMN_LIMIT) {
    const toggle = document.createElement("button");
    toggle.className = "show-more";
    toggle.textContent = expanded
      ? "Свернуть список"
      : `Показать ещё (${items.length - COLUMN_LIMIT})`;
    toggle.onclick = () => {
      expanded ? EXPANDED_COLUMNS.delete(name) : EXPANDED_COLUMNS.add(name);
      renderColumn(name, items, builder);
    };
    col.appendChild(toggle);
  }
  requestAnimationFrame(() => col.classList.remove("is-refreshing"));
}

function togglePostmatchDetails(element) {
  const key = element.dataset.detailKey;
  const opened = element.classList.toggle("open");
  if (key) {
    opened ? OPEN_POSTMATCH_DETAILS.add(key) : OPEN_POSTMATCH_DETAILS.delete(key);
  }
}

function notifyAboutLiveMatches(liveCards) {
  const isAllowed = "Notification" in window && Notification.permission === "granted";
  liveCards.forEach((card) => {
    const key = String(card.match_id || card.series_id || "");
    if (!key || SEEN_LIVE.has(key)) return;
    SEEN_LIVE.add(key);
    if (isAllowed) {
      new Notification("Матч начался", {
        body: `${card.radiant_team?.name || "?"} vs ${card.dire_team?.name || "?"} · ${card.event || ""}`,
      });
    }
  });
  sessionStorage.setItem("dota_analyst_seen_live", JSON.stringify([...SEEN_LIVE].slice(-200)));
}

async function loadModelStatus() {
  try {
    const response = await fetch(`${API}/api/model-status`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const model = data.model || {}, learning = data.learning || {}, collection = data.collection || {};
    const accuracy = model.accuracy == null ? "—" : `${(model.accuracy * 100).toFixed(1)}%`;
    const total = collection.total || 0;
    const fetched = collection.fetched || 0;
    const remaining = collection.remaining ?? Math.max(0, total - fetched);
    let collectionText = "";
    if (collection.state === "training") {
      collectionText = " · дообучение модели…";
    } else if (collection.state === "complete") {
      collectionText = ` · исторический сбор завершён: ${fetched}/${total}`;
    } else if (collection.state === "blocked_by_failed_maps") {
      collectionText = ` · сбор остановлен: ${fetched}/${total}, ошибок: ${collection.failed || 0}`;
    } else if (total) {
      collectionText = ` · сбор карт: ${fetched}/${total} · осталось ${remaining}`;
    }
    const liveQueue = learning.queued_maps ? ` · live-очередь: ${learning.queued_maps}` : "";
    $("modelStatus").textContent = `ML: ${model.samples || 0} карт · accuracy ${accuracy}${collectionText}${liveQueue}`;
  } catch (_) {
    $("modelStatus").textContent = "ML: статус временно недоступен";
  }
}

async function loadAnalytics() {
  const panel = $("analyticsPanel");
  try {
    const response = await fetch(API + "/api/analytics", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const audit = (await response.json()).prediction_audit || {};
    if (!audit.settled) {
      panel.innerHTML = '<div class="analytics-empty">Проверка прогнозов начнётся после завершения первой live-карты. Журнал фиксирует только прогнозы, которые были показаны на доске.</div>';
      return;
    }
    const accuracy = (audit.accuracy * 100).toFixed(1) + "%";
    const brier = audit.brier_score.toFixed(3);
    const sampleWarning = audit.settled < 30
      ? '<div class="analytics-warning">Недостаточно данных для оценки: нужно минимум 30 завершённых live-карт.</div>'
      : "";
    const rows = (audit.calibration || []).map((bucket) =>
      '<div class="cal-row"><span>' + bucket.range + '</span><span>' + bucket.samples + '</span><span>' +
      (bucket.predicted * 100).toFixed(0) + '%</span><span>' + (bucket.actual * 100).toFixed(0) + '%</span></div>'
    ).join("");
    panel.innerHTML =
      '<div class="analytics-head"><span>Качество live-прогнозов</span><span>' + audit.settled + ' завершено / ' + audit.shown + ' показано</span></div>' +
      '<div class="analytics-metrics"><div><small>Accuracy</small><b>' + accuracy + '</b></div><div><small>Brier score</small><b>' + brier + '</b></div></div>' +
      (rows ? '<div class="calibration"><div class="cal-row cal-title"><span>Прогноз</span><span>Карт</span><span>Модель</span><span>Факт</span></div>' + rows + '</div>' : "") + sampleWarning;
  } catch (_) {
    panel.innerHTML = "";
  }
}

async function loadDataQuality() {
  try {
    const response = await fetch(API + "/api/data-quality", { cache: "no-store" });
    if (!response.ok) return;
    const quality = await response.json();
    const health = $("dataHealth");
    health.textContent = "Данные: " + quality.full_maps + " полных карт · " +
      quality.target_fetched + " загружено из целевого набора · ошибок: " +
      quality.target_failed + " · проверенных прогнозов: " + quality.audited_predictions;
  } catch (_) {}
}

// ---------------- Card builders ----------------
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}
function teamLogo(t) {
  return t.logo
    ? `<img src="${t.logo}" alt="" onerror="this.style.display='none'"/>`
    : `<div style="width:26px"></div>`;
}
function teamBlock(t, side) {
  const rank = t.rank ? `<span class="trank">#${t.rank}</span>` : "";
  const favorite = FAVORITE_TEAMS.has(t.name);
  return `<div class="team ${side}">${teamLogo(t)}<div style="min-width:0"><div class="tname">${t.name}</div>${rank}</div><button class="fav-btn ${favorite ? "active" : ""}" data-team="${t.name}" title="Избранная команда">${favorite ? "★" : "☆"}</button></div>`;
}
function heroIcon(h, cls = "") {
  const inner = h.image
    ? `<img src="${h.image}" title="${h.name}" onerror="this.parentNode.innerHTML='<div class=ph>${h.name}</div>'"/>`
    : `<div class="ph">${h.name}</div>`;
  return `<div class="hero ${cls}">${inner}</div>`;
}
function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function prematchCard(c) {
  const p = c.predictions || {};
  const form = c.form_context || {};
  const maps = p.total_maps || {};
  const prematchMarkets = p.winner ? `
    <div class="prematch-markets">
      <div class="market-head">Прематч · ${p.source === "prematch_team_form" ? "форма команд" : "—"}</div>
      <div class="market-grid">
        <div><span>Победитель серии</span><b>${p.winner.team} · ${p.winner.probability}%</b></div>
        <div><span>Счёт серии</span><b>${p.series_score?.favourite || "—"} ${p.series_score?.score || "—"}</b></div>
      </div>
    </div>` : "";
  const formHtml = form.team_a?.maps || form.team_b?.maps ? `
    <div class="form-context">
      <span>${c.team_a.name}: Elo ${form.team_a?.elo ?? "—"} · форма ${form.team_a?.recent_win_rate ?? "—"}%</span>
      <span>${c.team_b.name}: Elo ${form.team_b?.elo ?? "—"} · форма ${form.team_b?.recent_win_rate ?? "—"}%</span>
      ${form.h2h_maps ? `<small>Очные карты: ${form.h2h_maps}</small>` : ""}
    </div>` : "";
  return el(`
    <div class="card card--prematch">
      <div class="event"><span>${c.event}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="teams-row">
        ${teamBlock(c.team_a, "")}
        <span class="vs">VS</span>
        ${teamBlock(c.team_b, "right")}
      </div>
      <div class="starttime">🕓 ${fmtDate(c.start_time) || "Время уточняется"}</div>
      ${prematchMarkets}
      ${formHtml}
    </div>`);
}

function postmatchCard(c) {
  const aWin = c.score_a >= c.score_b;
  const detailed = c.games_detailed || [];

  // Collapsible per-game detailed blocks
  const gamesHtml = detailed.map((g) => gameDetailedHtml(g, c)).join("");
  const hasDetailed = detailed.length > 0;
  const detailKey = String(c.series_id || [c.event, c.team_a?.name, c.team_b?.name, c.ended_at].join("|"));
  const isOpen = OPEN_POSTMATCH_DETAILS.has(detailKey);

  return el(`
    <div class="card postmatch-card card--postmatch">
      <div class="event"><span>${c.event}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="teams-row">
        ${teamBlock(c.team_a, "")}
        <span class="score-badge"><span class="${aWin ? "win" : ""}">${c.score_a}</span> : <span class="${!aWin ? "win" : ""}">${c.score_b}</span></span>
        ${teamBlock(c.team_b, "right")}
      </div>
      ${hasDetailed ? `
        <div class="pm-games ${isOpen ? "open" : ""}" data-detail-key="${detailKey}">
          <div class="pm-games-toggle" onclick="togglePostmatchDetails(this.parentNode)">
            <span>📋 Детали по ${detailed.length} ${detailed.length === 1 ? 'карте' : 'картам'}</span>
            <span class="pm-arrow">▾</span>
          </div>
          <div class="pm-games-body">
            ${gamesHtml}
          </div>
        </div>
      ` : ""}
    </div>`);
}

function _sideLabel(side, teamA, teamB) {
  if (!side) return "—";
  return side === "radiant" ? `☀ ${teamA}` : `🌙 ${teamB}`;
}
function _verdictBadge(v) {
  if (v === true)  return `<span class="vbadge v-ok">✓</span>`;
  if (v === false) return `<span class="vbadge v-no">✗</span>`;
  return `<span class="vbadge v-na">—</span>`;
}
function _valOr(v, fallback = "—") {
  return (v === null || v === undefined) ? fallback : v;
}
function _fmtDur(sec) {
  if (!sec && sec !== 0) return "—";
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function gameDetailedHtml(g, c) {
  const teamA = c.team_a?.name || "A";
  const teamB = c.team_b?.name || "B";
  const p = g.prediction || {};
  const v = g.verdict || {};

  // Header row for this game
  const winnerBadge = g.winner
    ? `<span class="pm-winner">${g.winner} победил</span>`
    : `<span class="pm-winner">—</span>`;

  // Left: actual stats
  const actualHtml = `
    <div class="pm-col pm-actual">
      <div class="pm-col-head">Факт</div>
      <div class="pm-stat-row"><span class="pm-label">Длительность</span><span>${_fmtDur(g.duration_sec)} (${g.duration_min ?? "—"} мин)</span></div>
      <div class="pm-stat-row">
        <span class="pm-label">Фраги</span>
        <span>${teamA.slice(0,6)}: ${_valOr(g.team_a_score)} | ${teamB.slice(0,6)}: ${_valOr(g.team_b_score)}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Общий счёт</span>
        <span>${_valOr(g.team_a_score) + _valOr(g.team_b_score)}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Вышки снесены</span>
        <span>${teamA.slice(0,6)}: ${_valOr(g.team_a_towers)} · ${teamB.slice(0,6)}: ${_valOr(g.team_b_towers)}</span>
      </div>
      <div class="pm-stat-row"><span class="pm-label">First Blood</span><span>${_sideLabel(g.fb_side, teamA, teamB)}</span></div>
      <div class="pm-stat-row"><span class="pm-label">Первые 15 киллов</span><span>${_sideLabel(g.f15_side, teamA, teamB)}</span></div>
    </div>`;

  // Right: predictions + verdicts
  const total = p.total_over_under || {};
  const kills_total = p.kills_total_over_under || {};
  const predHtml = `
    <div class="pm-col pm-pred">
      <div class="pm-col-head">Прогноз</div>
      <div class="pm-stat-row">
        <span class="pm-label">Победитель ${_verdictBadge(v.winner)}</span>
        <span>${_valOr((p.winner||{}).team)} · ${_valOr((p.winner||{}).probability)}%</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Длительность ${_verdictBadge(v.duration)}</span>
        <span>${total.side ? (total.side === "over" ? "ТБ" : "ТМ") : "—"} ${total.threshold !== undefined ? total.threshold : "—"} мин</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Киллы (всего) ${_verdictBadge(v.kills_total)}</span>
        <span>${kills_total.side ? (kills_total.side === "over" ? "ТБ" : "ТМ") : "—"} ${kills_total.threshold !== undefined ? kills_total.threshold : "—"}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Вышки (всего) ${_verdictBadge(v.towers_total)}</span>
        <span>${_valOr((p.towers||{}).total)}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">First Blood ${_verdictBadge(v.first_blood)}</span>
        <span>${_valOr((p.first_blood||{}).team)}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Первые 15 ${_verdictBadge(v.first_to_15)}</span>
        <span>${_valOr((p.first_to_15||{}).team)}</span>
      </div>
    </div>`;

  return `
    <div class="pm-game">
      <div class="pm-game-head">
        <span class="pm-game-no">Карта ${g.game}</span>
        <span class="pm-game-dur">${_fmtDur(g.duration_sec)}</span>
        ${winnerBadge}
      </div>
      <div class="pm-grid">
        ${actualHtml}
        ${predHtml}
      </div>
    </div>`;
}

function heroesRow(list, cls = "") {
  return `<div class="heroes ${cls}">${(list || []).map((h) => heroIcon(h, cls === "bans" ? "" : "")).join("")}</div>`;
}

function liveCard(c) {
  const p = c.predictions || {};
  const w = p.winner || {};
  const pr = w.prob_radiant ?? 50;
  const draft = c.draft || {};
  const completedMaps = c.completed_maps || [];
  const completedHtml = completedMaps.length ? `
    <div class="completed-maps">
      <div class="completed-head">Сыгранные карты</div>
      ${completedMaps.map((game) => `
        <div class="completed-map">
          <span>Карта ${game.game}</span>
          <b>${game.winner}</b>
          <span>${game.radiant_score} : ${game.dire_score} · ${_fmtDur(game.duration_sec)}</span>
        </div>`).join("")}
    </div>` : "";
  if (c.waiting_for_next_map) {
    return el(`
      <div class="card card--live waiting-card">
        <div class="event"><span>${c.event} · Серия продолжается</span><span class="bo-tag">${c.bo}</span></div>
        <div class="teams-row">
          ${teamBlock(c.radiant_team, "")}
          <span class="vs">${c.series_score_a}–${c.series_score_b}</span>
          ${teamBlock(c.dire_team, "right")}
        </div>
        <div class="waiting-next-map">Ожидание следующей карты</div>
        ${completedHtml}
      </div>`);
  }
  const series = c.series_outlook || {};
  const draftContext = c.draft_context || {};
  const signalNames = { skip: "Пропустить", watch: "Наблюдать", strong: "Сильный сигнал" };
  const insightHtml = `
    <div class="live-insights">
      <div class="signal signal-${c.signal || "skip"}"><span>Сигнал</span><b>${signalNames[c.signal] || "Пропустить"}</b></div>
      ${series.favourite ? `<div><span>${series.mode === "series" ? "Серия" : "Следующая карта"}</span><b>${series.favourite} · ${series.probability}%</b></div>` : ""}
      ${draftContext.radiant?.win_rate != null ? `<div><span>Драфт · история</span><b>${c.radiant_team.name} ${draftContext.radiant.win_rate}% / ${c.dire_team.name} ${draftContext.dire.win_rate ?? "—"}%</b></div>` : ""}
    </div>`;
  return el(`
    <div class="card card--live">
      <div class="event"><span>${c.event} · Игра ${c.game_no}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="teams-row">
        ${teamBlock(c.radiant_team, "")}
        <span class="vs">${c.series_score_a}–${c.series_score_b}</span>
        ${teamBlock(c.dire_team, "right")}
      </div>
      <div class="live-score"><span class="r">${c.live_score.radiant}</span><span class="sep">kills</span><span class="d">${c.live_score.dire}</span></div>
      ${completedHtml}

      <div class="draft">
        <div class="side-label radiant">☀ ${c.radiant_team.name}</div>
        ${heroesRow(draft.radiant_picks)}
        <div class="side-label dire">🌙 ${c.dire_team.name}</div>
        ${heroesRow(draft.dire_picks)}
      </div>

      <div class="pred">
        <div class="pbox full">
          <div class="plabel">Победитель${p.ml_winner ? " · ML" : ""}</div>
          <div class="pval">${w.team || "—"} · ${w.probability ?? "—"}%</div>
          <div class="winbar"><div style="width:${pr}%"></div></div>
          <div class="winrow"><span class="r">${c.radiant_team.name} ${pr}%</span><span class="d">${100 - pr}% ${c.dire_team.name}</span></div>
        </div>
        <div class="pbox">
          <div class="plabel">Тотал фрагов</div>
          <div class="pval">${p.kills_total_over_under?.side === "over" ? "ТБ" : p.kills_total_over_under?.side === "under" ? "ТМ" : "—"} ${p.kills_total_over_under?.threshold ?? "—"}</div>
          <div class="psub">линия без возврата</div>
        </div>
        <div class="pbox">
          <div class="plabel">Длительность</div>
          <div class="pval">${p.duration_min ?? "—"} мин</div>
        </div>
        <div class="pbox">
          <div class="plabel">Вышки (потенциал)</div>
          <div class="pval">${p.towers?.total ?? "—"}</div>
          <div class="psub">${p.towers?.radiant ?? "—"} ☀ / 🌙 ${p.towers?.dire ?? "—"}</div>
        </div>
        <div class="pbox">
          <div class="plabel">Первым 15 киллов</div>
          <div class="pval">${p.first_to_15?.team || "—"}</div>
          <div class="psub">${p.first_to_15?.probability ?? "—"}% уверенность</div>
        </div>
        <div class="pbox full">
          <div class="plabel">Ultra Kill / Rampage</div>
          <div class="pval"><span class="tag-level lvl-${p.multikill?.level || "Low"}">${p.multikill?.level || "—"}</span>
            <span class="psub">чаще у ${p.multikill?.likely_side || "—"}</span></div>
        </div>
      </div>
      ${insightHtml}
      <div class="conf">достоверность данных: ${Math.round((p.confidence || 0) * 100)}%</div>
    </div>`);
}

// ---------------- Status / events ----------------
function setStatus(msg) {
  $("status").textContent = msg;
}
function setupAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  if ($("autoRefresh").checked) {
    const interval = _lastFast ? REFRESH_FAST : REFRESH_SLOW;
    refreshTimer = setInterval(refresh, interval);
  }
}

function init() {
  $("leagueBtn").onclick = () => $("leaguePanel").classList.toggle("hidden");
  $("watchBtn").onclick = () => $("watchPanel").classList.toggle("hidden");
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".league-picker")) {
      $("leaguePanel").classList.add("hidden");
      $("watchPanel").classList.add("hidden");
    }
  });
  document.addEventListener("click", (e) => {
    const button = e.target.closest(".fav-btn");
    if (!button) return;
    e.preventDefault();
    e.stopPropagation();
    const team = button.dataset.team;
    if (!team) return;
    FAVORITE_TEAMS.has(team) ? FAVORITE_TEAMS.delete(team) : FAVORITE_TEAMS.add(team);
    persist();
    button.classList.toggle("active", FAVORITE_TEAMS.has(team));
    button.textContent = FAVORITE_TEAMS.has(team) ? "★" : "☆";
  });
  $("leagueSearch").oninput = (e) => renderLeagueList(e.target.value);
  $("watchAddBtn").onclick = () => {
    const v = $("watchInput").value;
    if (v) { addWatch(v); $("watchInput").value = ""; }
  };
  $("watchInput").onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); addWatch($("watchInput").value); $("watchInput").value = ""; }
  };
  $("refreshBtn").onclick = refresh;
  $("notifyBtn").onclick = async () => {
    if (!("Notification" in window)) { alert("Этот браузер не поддерживает уведомления"); return; }
    const permission = await Notification.requestPermission();
    $("notifyBtn").classList.toggle("is-active", permission === "granted");
  };
  if ("Notification" in window) $("notifyBtn").classList.toggle("is-active", Notification.permission === "granted");
  $("autoRefresh").onchange = setupAutoRefresh;

  renderWatchList();
  loadLeagues().then(refresh);
  setupAutoRefresh();
}

init();

"use strict";

// v0.7.46: when running without Docker (python -m uvicorn business.app:app
// on :8000 + python -m http.server for the frontend on :8080), the
// API lives on a different origin, so we point at it explicitly.
// In Docker compose the nginx reverse proxy serves both on :80
// and `const API = ""` was correct.  Keep the explicit host for
// the no-Docker dev path; switch back to "" for compose if you
// want nginx passthrough.
const API = "http://localhost:8000";
const LS_KEY = "dota_analyst_leagues";
const LS_WATCH = "dota_analyst_watchlist";
const LS_THEME = "dota_analyst_theme";
// v0.3.25f: auto-refresh became a radio (Вкл/Выкл).  The old
// checkbox was #autoRefresh; we keep a single helper so the rest
// of the file doesn't care how the UI exposes the toggle.
function isAutoRefreshOn() {
  const el = document.querySelector('input[name="autoRefresh"]:checked');
  return !el || el.value !== "off";  // default to "on" if the radio is missing
}
let LEAGUES = [];
let SELECTED = new Set(JSON.parse(localStorage.getItem(LS_KEY) || "[]"));
let WATCHLIST = JSON.parse(localStorage.getItem(LS_WATCH) || "[]"); // [{id, title?}]
let refreshTimer = null;

const $ = (id) => document.getElementById(id);

// ---------------- League picker ----------------
async function loadLeagues() {
  try {
    const r = await fetch(`${API}/api/leagues`, { credentials: "same-origin" });
    const d = await r.json();
    // v0.3.25f: drop leagues with no scheduled matches — the old
    // picker listed every DatDota league (60+), most of them empty
    // for our use case, which just made the dropdown noisy.
    // match_count is the "would appear in /api/board right now" count.
    //
    // v0.4.0.2: include any league that is `is_active` on DLTV, even
    // if our internal `_latest_auto_board.match_count` is still 0.
    // The user-reported symptom: "1win has live matches right now but
    // it's missing from the picker".  Cause: the SSE publisher hasn't
    // finished its first build yet (DLTV v1 events call is slow,
    // 17-36s), so `_latest_auto_board` is empty and `match_count` is
    // 0 for every league — even though DLTV's own `is_active` flag
    // already says "1win Essence 2 is currently running".  The
    // previous code's `match_count>0` strict filter dropped 1win from
    // the picker for the first 20-30s of every page load, then the
    // `is_active` fallback *should* have re-added it but only fires
    // when strictActive is empty (and 1win eventually appears there
    // as soon as one match lands in the auto-board).  Effectively
    // 1win flickers in and out of the picker for the first minute.
    //
    // Fix: trust DLTV's `is_active` as the primary signal and ALSO
    // keep any league that has at least one card in our auto-board
    // (so a postmatch league with is_active=false still appears).
    // This matches the user-stated rule: "hide only leagues that
    // have no scheduled DLTV matches — the rest belong in the list".
    const allLeagues = d.leagues || [];
    LEAGUES = allLeagues.filter(
      (l) => l.is_active || (l.match_count || 0) > 0
    );
    // v0.3.21: first-load UX.  An empty SELECTED set used to leave
    // the user staring at an empty board even though every league
    // has cards.  Auto-pick everything so the first /api/board
    // comes back populated.  We still persist the choice, so
    // a returning user with a saved selection keeps their
    // custom filter.
    // v0.3.25g: auto-select must also fire when the saved SELECTED
    // has STALE league IDs — leagues that no longer appear in
    // /api/leagues (or were dropped by the v0.3.25f empty-league
    // filter).  Without this, a returning user with old localStorage
    // sees a fully empty board with "Лиги 30" — looks broken.
    // v0.4.0.1: re-fire the auto-select on EVERY call, not just init.
    // If the first call landed during a /api/leagues empty-window
    // (DLTV v1 join lag, server hiccup, network blip), the user was
    // stuck with `SELECTED.size === 0` and an empty board until a
    // full page reload.  Now any subsequent loadLeagues() — which
    // the auto-refresh loop already calls every 5s via the
    // `refreshAll()` chain in some code paths — gets a second /
    // third / N-th chance to populate the picker.
    const leagueIds = new Set(LEAGUES.map((l) => l.id));
    const hadNone = SELECTED.size === 0;
    const hasStaleOnly = SELECTED.size > 0
      && ![...SELECTED].some((id) => leagueIds.has(id));
    if ((hadNone || hasStaleOnly) && LEAGUES.length > 0) {
      SELECTED = new Set(LEAGUES.map((l) => l.id));
      persist();
      // v0.4.0.1: if we transitioned from "nothing selected" to
      // "everything selected", the next /api/board will actually
      // have data — kick a refresh so the user doesn't have to wait
      // for the next auto-refresh tick.  `refresh` is defined later
      // in this file; safe to call here because `loadLeagues`
      // itself is only invoked from already-initialised code paths.
      if (hadNone) {
        try { refresh(); } catch (e) { /* refresh not yet bound — ignore */ }
      }
    }
    renderLeagueList();
    renderLeagueChips();
    updateLeagueCount();
    // v0.3.25g: surface a hint when the /api/leagues filter dropped
    // every league — user otherwise sees a totally empty board and
    // has no idea why.  Could be a backend hiccup or a quiet day.
    if (LEAGUES.length === 0 && (d.leagues || []).length > 0) {
      setStatus("Все лиги пустые (нет запланированных матчей). Попробуй позже.");
    }
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

  // Sort by match_count desc, then by title.  This puts the
  // busy leagues at the top of the picker.
  const items = LEAGUES
    .filter((l) => l.title.toLowerCase().includes(f))
    .sort((a, b) => (b.match_count || 0) - (a.match_count || 0)
                    || (a.title || "").localeCompare(b.title || ""));

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "league-group-header";
    empty.textContent = "Нет активных лиг";
    list.appendChild(empty);
    return;
  }

  // Action bar — "All" / "None" / count
  const actions = document.createElement("div");
  actions.className = "league-actions";
  const btnAll = document.createElement("button");
  btnAll.className = "btn-ghost";
  btnAll.textContent = "Все";
  btnAll.onclick = (e) => { e.preventDefault(); selectAllLeagues(); };
  const btnNone = document.createElement("button");
  btnNone.className = "btn-ghost";
  btnNone.textContent = "Очистить";
  btnNone.onclick = (e) => { e.preventDefault(); clearAllLeagues(); };
  const summary = document.createElement("span");
  summary.className = "league-summary";
  const totalMatches = items.reduce((s, l) => s + (l.match_count || 0), 0);
  summary.textContent = `Выбрано ${SELECTED.size}/${items.length} · матчей: ${totalMatches}`;
  actions.appendChild(btnAll);
  actions.appendChild(btnNone);
  actions.appendChild(summary);
  list.appendChild(actions);

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
      renderLeagueChips();
      refresh();
    };
    const span = document.createElement("span");
    span.textContent = l.title;
    const count = document.createElement("span");
    count.className = "league-count";
    count.textContent = (l.match_count != null) ? `${l.match_count}` : "—";
    item.appendChild(cb);
    item.appendChild(span);
    item.appendChild(count);
    list.appendChild(item);
  });
}

function selectAllLeagues() {
  SELECTED = new Set(LEAGUES.map((l) => l.id));
  persist();
  renderLeagueList($("leagueSearch")?.value || "");
  renderLeagueChips();
  updateLeagueCount();
  refresh();
}
function clearAllLeagues() {
  SELECTED = new Set();
  persist();
  renderLeagueList($("leagueSearch")?.value || "");
  renderLeagueChips();
  updateLeagueCount();
  refresh();
}

function renderLeagueChips() {
  // v0.3.21: a single-row strip of the top-5 leagues so the
  // most relevant filter chips are always one click away,
  // without opening the picker.  Clicking a chip toggles that
  // league.  Clicking the "all" chip selects everything.
  const host = $("leagueChips");
  if (!host) return;
  host.innerHTML = "";
  // Top 5 by match_count
  const top = LEAGUES
    .filter((l) => (l.match_count || 0) > 0)
    .sort((a, b) => (b.match_count || 0) - (a.match_count || 0))
    .slice(0, 5);
  if (top.length === 0) return;
  // "All" chip
  const all = document.createElement("button");
  all.className = "chip chip-all " + (SELECTED.size === LEAGUES.length ? "chip-on" : "");
  all.textContent = "Все лиги";
  all.title = "Включить все лиги";
  all.onclick = () => selectAllLeagues();
  host.appendChild(all);
  top.forEach((l) => {
    const btn = document.createElement("button");
    const on = SELECTED.has(l.id);
    btn.className = "chip " + (on ? "chip-on" : "");
    btn.textContent = `${l.title} · ${l.match_count || 0}`;
    btn.title = on ? `Скрыть ${l.title}` : `Показать только ${l.title}`;
    btn.onclick = () => {
      if (SELECTED.has(l.id)) SELECTED.delete(l.id);
      else SELECTED.add(l.id);
      persist();
      renderLeagueChips();
      renderLeagueList($("leagueSearch")?.value || "");
      updateLeagueCount();
      refresh();
    };
    host.appendChild(btn);
  });
}

function persist() {
  localStorage.setItem(LS_KEY, JSON.stringify([...SELECTED]));
  localStorage.setItem(LS_WATCH, JSON.stringify(WATCHLIST));
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
    const r = await fetch(url, { credentials: "same-origin" });
    const d = await r.json();
    renderColumn("prematch", d.prematch, prematchCard);
    renderColumn("live", d.live, liveCard);
    renderColumn("postmatch", d.postmatch, postmatchCard);
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
      if (isAutoRefreshOn()) setupAutoRefresh();
    }
  } catch (e) {
    setStatus("Ошибка загрузки: " + e.message);
  }
}
let _lastFast = false;

function renderColumn(name, items, builder) {
  const col = $("col-" + name);
  $("cnt-" + name).textContent = (items || []).length;
  col.innerHTML = "";
  if (!items || items.length === 0) {
    col.innerHTML = `<div class="empty">Нет матчей</div>`;
    return;
  }
  items.forEach((it) => col.appendChild(builder(it)));
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
  return `<div class="team ${side}">${teamLogo(t)}<div style="min-width:0"><div class="tname">${t.name}</div>${rank}</div></div>`;
}
function heroIcon(h, cls = "") {
  const inner = h.image
    ? `<img src="${h.image}" title="${h.name}" onerror="this.parentNode.innerHTML='<div class=ph>${h.name}</div>'"/>`
    : `<div class="ph">${h.name}</div>`;
  // v0.4.0-players: surface the actual player name under the
  // hero icon when the backend sends `player_name` (DLTV-style:
  // "Rylai" under a Puck portrait).  We use a separate
  // `.hero-player` div so CSS can size/colour it without
  // touching the icon itself.  Country code is kept in
  // `data-country` for future flag-overlay work.
  const pn = h.player_name;
  const pc = h.player_country;
  const playerLine = pn
    ? `<div class="hero-player" data-country="${pc || ""}">${pn}</div>`
    : "";
  return `<div class="hero ${cls}">${inner}${playerLine}</div>`;
}
function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function prematchCard(c) {
  return el(`
    <div class="card">
      <div class="event"><span>${c.event}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="teams-row">
        ${teamBlock(c.team_a, "")}
        <span class="vs">VS</span>
        ${teamBlock(c.team_b, "right")}
      </div>
      <div class="starttime">🕓 ${fmtDate(c.start_time) || "Время уточняется"}</div>
    </div>`);
}

function postmatchCard(c) {
  const aWin = c.score_a >= c.score_b;
  const pred = c.prediction || {};
  const detailed = c.games_detailed || [];

  // Header: series-level prediction summary (only if available)
  let predSummaryHtml = "";
  if (pred.winner_team) {
    const correct = pred.winner_team === c.winner;
    predSummaryHtml = `
      <div class="postmatch-pred ${correct ? "correct" : "wrong"}">
        <div class="pred-label">🤖 Итог серии</div>
        <div class="pred-row">
          <span class="pred-winner">${pred.winner_team}</span>
          <span class="pred-prob">${pred.winner_probability ?? "—"}%</span>
        </div>
        <div class="pred-actual">
          <span class="pred-actual-label">Факт:</span>
          <span class="pred-actual-winner">${c.winner} (${c.score_a}–${c.score_b})</span>
          <span class="pred-verdict">${correct ? "✓ точно" : "✗ мимо"}</span>
        </div>
      </div>`;
  }

  // Collapsible per-game detailed blocks
  const gamesHtml = detailed.map((g) => gameDetailedHtml(g, c)).join("");
  const hasDetailed = detailed.length > 0;

  return el(`
    <div class="card postmatch-card">
      <div class="event"><span>${c.event}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="teams-row">
        ${teamBlock(c.team_a, "")}
        <span class="score-badge"><span class="${aWin ? "win" : ""}">${c.score_a}</span> : <span class="${!aWin ? "win" : ""}">${c.score_b}</span></span>
        ${teamBlock(c.team_b, "right")}
      </div>
      ${predSummaryHtml}
      ${hasDetailed ? `
        <div class="pm-games">
          <div class="pm-games-toggle" onclick="this.parentNode.classList.toggle('open')">
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
        <span class="pm-label">Команды ( frags )</span>
        <span>${teamA.slice(0,6)}: ${_valOr(g.team_a_score)} | ${teamB.slice(0,6)}: ${_valOr(g.team_b_score)}</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Общий счёт</span>
        <span>${_valOr(g.team_a_score) + _valOr(g.team_b_score)}</span>
      </div>
      <!-- v0.3.25e: TOWERS HIDDEN — no reliable data source yet (full_matches
           lacks the per-side tower bitmask; the heuristic number is
           made up and was misleading the user).  Uncomment when
           we have a real per-side tower count. -->
      <!--
      <div class="pm-stat-row">
        <span class="pm-label">Вышки снесены</span>
        <span>${teamA.slice(0,6)}: ${_valOr(g.team_a_towers)} · ${teamB.slice(0,6)}: ${_valOr(g.team_b_towers)}</span>
      </div>
      -->
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
        <span>${total.side ? (total.side === "over" ? "ТБ" : "ТМ") : "—"} ${total.threshold !== undefined ? total.threshold : "—"} мин (${total.formatted || "—"})</span>
      </div>
      <div class="pm-stat-row">
        <span class="pm-label">Киллы (всего) ${_verdictBadge(v.kills_total)}</span>
        <span>${kills_total.side ? (kills_total.side === "over" ? "ТБ" : "ТМ") : "—"} ${kills_total.threshold !== undefined ? kills_total.threshold : "—"}</span>
      </div>
      <!-- v0.3.25e: TOWERS PREDICTION HIDDEN — see note above. -->
      <!--
      <div class="pm-stat-row">
        <span class="pm-label">Вышки (всего) ${_verdictBadge(v.towers_total)}</span>
        <span>${_valOr((p.towers||{}).total)}</span>
      </div>
      -->
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
  // v0.3.24g: TM/ТБ helpers — same shape as the postmatch card.
  const overUnder = (ou) => {
    if (!ou || !ou.side) return "—";
    const side = ou.side === "over" ? "ТБ" : "ТМ";
    return `${side} ${ou.threshold ?? "—"}`;
  };
  // v0.3.24g: format the game clock (raw seconds) as MM:SS.
  const fmtClock = (s) => {
    if (typeof s !== "number" || !Number.isFinite(s) || s < 0) return "—";
    const mm = Math.floor(s / 60);
    const ss = Math.floor(s % 60);
    return `${mm}:${String(ss).padStart(2, "0")}`;
  };
  // v0.3.24g: format networth (an int) as e.g. "23.9k" — DLTV style.
  const fmtGold = (g) => {
    if (typeof g !== "number" || !Number.isFinite(g)) return "—";
    const sign = g < 0 ? "-" : "";
    const abs = Math.abs(g);
    return `${sign}${(abs / 1000).toFixed(1)}k`;
  };
  // v0.3.24h: gold block — handle partial data.  The lead only
  // exists when both sides are known; when only one is, show what
  // we have and leave the lead / missing side as "—".
  let goldLine = "";
  if (c.live_gold && (typeof c.live_gold.radiant === "number" || typeof c.live_gold.dire === "number")) {
    const g = c.live_gold;
    const rn = g.radiant, dn = g.dire, lead = g.lead_radiant;
    if (typeof lead === "number") {
      const aheadIsRadiant = lead >= 0;
      const aheadName = aheadIsRadiant ? c.radiant_team.name : c.dire_team.name;
      const arrow = aheadIsRadiant ? "▲" : "▼";
      const cls = aheadIsRadiant ? "gold-pos" : "gold-neg";
      goldLine = `<div class="live-gold"><span class="${cls}">${arrow} ${fmtGold(lead)}</span><span class="gold-team">${aheadName}</span><span class="gold-split">${fmtGold(rn)} ☀ / 🌙 ${fmtGold(dn)}</span></div>`;
    } else {
      // Partial: only one side known.
      const which = typeof rn === "number" ? "radiant" : "dire";
      const val = which === "radiant" ? rn : dn;
      const sym = which === "radiant" ? "☀" : "🌙";
      const name = which === "radiant" ? c.radiant_team.name : c.dire_team.name;
      goldLine = `<div class="live-gold"><span class="gold-split">${sym} ${fmtGold(val)} ${name}</span></div>`;
    }
  }
  // v0.4.0.1: destroyed-tower counts from the dltv_browser
  // mini-map scrape.  DLTV doesn't expose this in the socket
  // payload; the only source is the rendered mini-map icons
  // (standing = full color, destroyed = greyscale).  We render
  // a small "Башни: 3 / 11 ☀ / 🌙 5 / 11" line under the gold
  // block — when the cache didn't have a mini-map to read, the
  // whole line is omitted (not "—").  Two shapes:
  //   1. side split:
  //      {radiant_destroyed, dire_destroyed,
  //       radiant_standing,  dire_standing}
  //   2. no side split:
  //      {standing, total, note}
  let towerLine = "";
  const dt = c.destroyed_towers;
  if (dt && (typeof dt.radiant_destroyed === "number" || typeof dt.dire_destroyed === "number")) {
    // Side split.  Convention: each side has 11 towers in a
    // standard map (3 T1 + 2 mid T1 + 2 T2 + 4 barracks = 11;
    // T3 = 2 ancient towers but DLTV only counts the 11
    // "lane/barracks" towers for the destroyed stat).
    const rD = dt.radiant_destroyed;
    const dD = dt.dire_destroyed;
    const rS = dt.radiant_standing;
    const dS = dt.dire_standing;
    const fmt = (destroyed, standing) => {
      const tot = (destroyed != null && standing != null)
        ? destroyed + standing : 11;
      return `${destroyed ?? "—"}/${tot}`;
    };
    towerLine = `<div class="live-towers" title="Снесено башен: radiant ${rD ?? "—"}, dire ${dD ?? "—"}">🏰 Башни: <span class="r">${fmt(rD, rS)}</span> ☀ / 🌙 <span class="d">${fmt(dD, dS)}</span></div>`;
  } else if (dt && (typeof dt.standing === "number" || typeof dt.total === "number")) {
    const tot = dt.total ?? 22;
    const std = dt.standing ?? 0;
    const dst = Math.max(0, tot - std);
    towerLine = `<div class="live-towers" title="Всего: ${tot}, стоит: ${std}">🏰 Башни: ${dst}/${tot} (всего)</div>`;
  }
  // v0.3.24h: per-team side block — team name + row of 5 big hero
  // icons.  This is the DLTV layout: name on top, hero icons
  // underneath, player names below the icons.  We don't have
  // player names in the cache (the picks carry hero names, not
  // player names), so the label under each icon is the hero name.
  // v0.3.25i: also render a static BANS row below the picks (no
  // collapse — always visible) so the user can see what was taken
  // away without clicking a toggle.  Smaller + desaturated to keep
  // the visual hierarchy clear (picks > bans).
  // v0.7.50: highlight the team that's leading the series.  The
  // card has `series_score_a` (radiant's wins) and `series_score_b`
  // (dire's wins).  When one side is ahead, colour that side's
  // name green and add a "ведёт" badge — DLTV's colour-coded dots
  // are easier to read at a glance than the bare "серия 0-1" text
  // (user feedback 2026-08-03: "надо смотреть на предыдущие карты,
  // чтобы понимать какая команда ведёт" — fix that friction).
  const sa = Number(c.series_score_a || 0);
  const sb = Number(c.series_score_b || 0);
  const radiantLeads = sa > sb;
  const direLeads    = sb > sa;
  const seriesTied   = sa === sb;
  const sideBlock = (team, picks, bans, sideClass, isLeading) => {
    const heroCells = (picks || []).map((h) => {
      const inner = h.image
        ? `<img src="${h.image}" title="${h.name || ""}" onerror="this.parentNode.innerHTML='<div class=ph>${h.name || "?"}</div>'"/>`
        : `<div class="ph">${h.name || "?"}</div>`;
      return `<div class="hero hero-lg">${inner}<div class="hero-name">${h.name || "—"}</div></div>`;
    }).join("");
    const ph = (picks || []).length === 0
      ? `<div class="heroes-empty">— нет пиков —</div>`
      : "";
    const banCells = (bans || []).map((h) => {
      const inner = h.image
        ? `<img src="${h.image}" title="${h.name || ""}" onerror="this.parentNode.innerHTML='<div class=ph>${h.name || "?"}</div>'"/>`
        : `<div class="ph">${h.name || "?"}</div>`;
      return `<div class="hero hero-ban" title="${h.name || "—"}">${inner}</div>`;
    }).join("");
    const banRow = (bans || []).length === 0
      ? ""
      : `<div class="side-block-bans-label">BANS</div><div class="side-block-bans">${banCells}</div>`;
    // Mark the leading side with a `leading` class (CSS paints the
    // name in green) and append a small "ведёт" badge.  For tied
    // series we don't paint anything — both names stay neutral.
    const leadingCls = isLeading ? " leading" : "";
    const leadingBadge = isLeading
      ? `<span class="series-lead-badge" title="Лидирует в серии ${sa}–${sb}">ведёт</span>`
      : "";
    return `<div class="side-block ${sideClass}">
      <div class="side-block-name${leadingCls}">${teamLogo(team)} <span>${team.name}</span>${leadingBadge}</div>
      <div class="side-block-heroes">${heroCells}${ph}</div>
      ${banRow}
    </div>`;
  };
  return el(`
    <div class="card live-card">
      <div class="event"><span>${c.event} · Игра ${c.game_no}</span><span class="bo-tag">${c.bo}</span></div>
      <div class="live-header">
        ${sideBlock(c.radiant_team, draft.radiant_picks, draft.radiant_bans, "radiant", radiantLeads)}
        <div class="live-center">
          <div class="live-score"><span class="r">${c.live_score.radiant}</span><span class="sep">:</span><span class="d">${c.live_score.dire}</span></div>
          <div class="live-clock">⏱ ${fmtClock(c.game_time)}</div>
          ${goldLine}
          ${towerLine}
          <div class="live-series-score ${seriesTied ? "tied" : (radiantLeads ? "r-leads" : "d-leads")}">серия ${c.series_score_a}–${c.series_score_b}${seriesTied ? "" : " · " + (radiantLeads ? c.radiant_team.name : c.dire_team.name) + " ведёт"}</div>
        </div>
        ${sideBlock(c.dire_team, draft.dire_picks, draft.dire_bans, "dire", direLeads)}
      </div>

      <div class="pred">
        <div class="pbox full">
          <div class="plabel">Победитель</div>
          <div class="pval">${w.team || "—"} · ${w.probability ?? "—"}%</div>
          <div class="winbar"><div style="width:${pr}%"></div></div>
          <div class="winrow"><span class="r">${c.radiant_team.name} ${pr}%</span><span class="d">${100 - pr}% ${c.dire_team.name}</span></div>
        </div>
        <div class="pbox">
          <div class="plabel">Киллы (потенциал)</div>
          <div class="pval">${overUnder(p.kills_total_over_under)}</div>
          <div class="psub">${p.kills?.radiant ?? "—"} ☀ / 🌙 ${p.kills?.dire ?? "—"}</div>
        </div>
        <div class="pbox">
          <div class="plabel">Длительность</div>
          <div class="pval">${overUnder(p.total_over_under)} мин</div>
          <div class="psub">${p.total_over_under?.formatted ?? "—"}</div>
        </div>
        <!-- v0.3.25e: TOWERS PREDICTION HIDDEN — see note above. -->
        <!--
        <div class="pbox">
          <div class="plabel">Вышки (потенциал)</div>
          <div class="pval">${overUnder(p.towers_over_under)}</div>
          <div class="psub">${p.towers?.radiant ?? "—"} ☀ / 🌙 ${p.towers?.dire ?? "—"}</div>
        </div>
        -->
        <div class="pbox">
          <div class="plabel">Первым 15 киллов</div>
          <div class="pval">${p.first_to_15?.team || "—"}</div>
          <div class="psub">${p.first_to_15?.probability ?? "—"}% уверенность</div>
        </div>
        <!-- v0.3.25f: ULTRA KILL / RAMPAGE HIDDEN — multikill classifier
             degenerated to "always High" on the pro corpus (notes in
             ml/train.py HEAD_REGISTRY[multikill]).  The heuristic
             that fills p.multikill is a guess; we'd rather not
             surface a number that has no real signal.  Uncomment when
             we have a proper per-slot multikill data source. -->
        <!--
        <div class="pbox full">
          <div class="plabel">Ultra Kill / Rampage</div>
          <div class="pval"><span class="tag-level lvl-${p.multikill?.level || "Low"}">${p.multikill?.level || "—"}</span>
            <span class="psub">чаще у ${p.multikill?.likely_side || "—"}</span></div>
        </div>
        -->
      </div>
      <div class="conf">достоверность данных: ${Math.round((p.confidence || 0) * 100)}%</div>
    </div>`);
}

// ---------------- Status / events ----------------
function setStatus(msg) {
  $("status").textContent = msg;
}

// ---------------- Auth (v0.4.0.1: cookie session) ----------------
//
// The browser's `EventSource` cannot send custom HTTP headers, so
// the SSE path needs a credential the browser WILL send.  Cookies
// work — the browser attaches them automatically when the
// EventSource is opened with `withCredentials: true` (or even by
// default for same-origin requests).  We POST the user's api_key
// to /api/auth/login, the server mints an HMAC-signed session
// cookie, and every subsequent call to /api/stream/* rides on
// that cookie.  The X-API-Key header is still accepted as a
// fallback so curl / the dev-edge-injected dev-key keep working.

async function checkAuthStatus() {
  try {
    const r = await fetch(`${API}/api/auth/status`, { credentials: "same-origin" });
    if (!r.ok) return false;
    const d = await r.json();
    return !!d.authenticated;
  } catch (e) {
    return false;
  }
}

function showAuthModal(errorMsg) {
  const modal = $("authModal");
  const errEl = $("authError");
  if (errorMsg) {
    errEl.textContent = errorMsg;
    errEl.classList.remove("hidden");
  } else {
    errEl.classList.add("hidden");
  }
  modal.classList.remove("hidden");
  // Focus the input so the user can paste/type immediately.
  setTimeout(() => $("authApiKey").focus(), 0);
}

function hideAuthModal() {
  $("authModal").classList.add("hidden");
}

async function doLogin() {
  const apiKey = ($("authApiKey").value || "").trim();
  if (!apiKey) {
    showAuthModal("Введите API ключ");
    return;
  }
  const btn = $("authLoginBtn");
  btn.disabled = true;
  btn.textContent = "Входим…";
  try {
    const r = await fetch(`${API}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (r.status === 200) {
      hideAuthModal();
      setStatus("Вход выполнен");
      // Re-fetch the board with the new cookie; SSE will pick
      // up the cookie on its next reconnect (we restart it
      // explicitly to surface any auth errors right away).
      stopSSE();
      await refresh();
      startSSE();
    } else if (r.status === 401) {
      showAuthModal("Неверный API ключ");
    } else if (r.status === 429) {
      showAuthModal("Слишком много попыток. Подождите минуту.");
    } else {
      showAuthModal(`Ошибка ${r.status}`);
    }
  } catch (e) {
    showAuthModal(`Сеть: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Войти";
  }
}

async function doLogout() {
  try {
    await fetch(`${API}/api/auth/logout`, {
      method: "POST", credentials: "same-origin",
    });
  } catch (e) { /* ignore — we still want to show the modal */ }
  stopSSE();
  showAuthModal();
}
function setupAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  if (isAutoRefreshOn()) {
    const interval = _lastFast ? REFRESH_FAST : REFRESH_SLOW;
    refreshTimer = setInterval(refresh, interval);
  }
}

// ---------------- SSE — live updates push (0.1.1) ----------------
//
// Connects to the business service's `/api/stream/matches` endpoint.
// On every `board_update` event the browser re-fetches the board —
// the SSE payload is just a tiny summary (live/prematch/postmatch
// counts) so the actual card rendering still uses the regular
// `/api/board` JSON.
//
// `EventSource` handles the low-level reconnect for free, but only
// for transient network errors; we also re-attach the handler after
// a manual `close()` (used when switching tabs).
let sseSource = null;
let sseReconnectTimer = null;
const SSE_RECONNECT_MS = 5000;

function startSSE() {
  if (sseSource) return;  // already connected
  try {
    // v0.4.0.1: withCredentials=true tells the browser to send
    // the dota_analyst_session cookie on the long-lived stream.
    // EventSource doesn't let you set custom headers, so the
    // cookie is the only way to authenticate.  Same-origin
    // requests would attach the cookie by default, but
    // withCredentials is the explicit, future-proof form
    // (in case the static UI is served from a different
    // subdomain in the future).
    sseSource = new EventSource(`${API}/api/stream/matches`, { withCredentials: true });
  } catch (e) {
    scheduleSSEReconnect();
    return;
  }
  sseSource.addEventListener("board_update", (ev) => {
    // Re-fetch the full board; the SSE payload is only a summary.
    if (isAutoRefreshOn()) refresh();
  });
  sseSource.addEventListener("error", () => {
    // EventSource auto-reconnects on transient errors; we just log
    // the most recent state to make debugging easier.
    setStatus("SSE reconnecting…");
  });
  sseSource.addEventListener("open", () => {
    setStatus("SSE connected");
  });
}

function stopSSE() {
  if (sseReconnectTimer) { clearTimeout(sseReconnectTimer); sseReconnectTimer = null; }
  if (sseSource) { sseSource.close(); sseSource = null; }
}

function scheduleSSEReconnect() {
  if (sseReconnectTimer) return;
  sseReconnectTimer = setTimeout(() => {
    sseReconnectTimer = null;
    startSSE();
  }, SSE_RECONNECT_MS);
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
  $("leagueSearch").oninput = (e) => renderLeagueList(e.target.value);
  $("watchAddBtn").onclick = () => {
    const v = $("watchInput").value;
    if (v) { addWatch(v); $("watchInput").value = ""; }
  };
  $("watchInput").onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); addWatch($("watchInput").value); $("watchInput").value = ""; }
  };
  $("refreshBtn").onclick = refresh;
  // v0.4.0.1: cookie auth modal — login flow wiring.
  $("authLoginBtn").onclick = doLogin;
  $("authApiKey").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); doLogin(); }
  });
  // v0.3.25f: auto-refresh is now a radio group (name="autoRefresh"),
  // not the old checkbox.  Hook the change at the group level.
  document.querySelectorAll('input[name="autoRefresh"]').forEach((r) => {
    r.addEventListener("change", () => {
      setupAutoRefresh();
      if (isAutoRefreshOn()) startSSE(); else stopSSE();
    });
  });
  // v0.3.25f: theme switcher (Тёмная/Светлая).  Persisted in
  // localStorage so the user's choice survives a refresh.
  const applyTheme = (t) => {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem(LS_THEME, t); } catch (_) {}
  };
  const savedTheme = (() => { try { return localStorage.getItem(LS_THEME); } catch (_) { return null; } })();
  applyTheme(savedTheme === "light" ? "light" : "dark");
  const savedThemeRadio = document.querySelector(`input[name="theme"][value="${savedTheme === "light" ? "light" : "dark"}"]`);
  if (savedThemeRadio) savedThemeRadio.checked = true;
  document.querySelectorAll('input[name="theme"]').forEach((r) => {
    r.addEventListener("change", () => { if (r.checked) applyTheme(r.value); });
  });

  // v0.3.22 cont 4: when the user comes back to a background tab,
  // the page may have missed several auto-refresh ticks (browsers
  // throttle setInterval in inactive tabs).  A visibilitychange
  // → visible event forces an immediate refresh so the user
  // doesn't stare at a stale "Обновлено 11:24" status for 30+
  // minutes after switching back.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && isAutoRefreshOn()) {
      refresh();
    }
  });

  renderWatchList();
  // v0.4.0.1: cookie-auth gate.  Two modes, distinguished by
  // the `X-Edge-Mode` response header that nginx emits on
  // every response (default "dev", "prod" when PROD_MODE=1):
  //
  //   * dev: the nginx edge injects the dev X-API-Key into
  //     every /api/* call (see web/nginx.conf
  //     `effective_api_key` map).  The cookie is OPTIONAL.
  //     The static UI works out of the box — no login modal
  //     on page load.  Ctrl+L still shows it for users who
  //     want to log in explicitly (e.g. to test the auth
  //     flow).
  //
  //   * prod: the nginx edge does NOT inject the dev key.
  //     The only way to authenticate is a session cookie
  //     from POST /api/auth/login.  We probe
  //     /api/auth/status on init; if the answer is "no",
  //     we show the login modal.
  //
  // We probe X-Edge-Mode via a lightweight HEAD-style fetch
  // on the index.html.  The header is set on EVERY nginx
  // response (incl. static assets), so the first
  // index.html fetch is enough.  In dev we optimistically
  // start the board refresh; if the edge is misconfigured
  // (X-Edge-Mode not set at all), the very first /api/board
  // will 401 and the user sees an empty board.  They can
  // hit Ctrl+L to enter the dev key manually.
  fetch(`${API}/index.html?v=0.4.0.1`, { method: "HEAD", credentials: "same-origin" })
    .then((r) => {
      const mode = r.headers.get("X-Edge-Mode") || "dev";
      window.__EDGE_MODE__ = mode;
      if (mode === "prod") {
        return checkAuthStatus().then((authed) => {
          if (!authed) {
            showAuthModal();
            return;
          }
          loadLeagues().then(refresh);
          setupAutoRefresh();
          startSSE();
        });
      } else {
        // dev: skip the modal, rely on the nginx-injected key.
        loadLeagues().then(refresh);
        setupAutoRefresh();
        startSSE();
      }
    })
    .catch(() => {
      // HEAD probe failed (network blip, old nginx without
      // X-Edge-Mode).  Fall back to dev behavior — show the
      // board.  If auth is actually missing, /api/board
      // returns 401 and the user can hit Ctrl+L to log in.
      window.__EDGE_MODE__ = "dev";
      loadLeagues().then(refresh);
      setupAutoRefresh();
      startSSE();
    });

  // v0.4.0.1: Ctrl+L re-shows the auth modal.  Useful in dev
  // (where the modal is hidden by default) and in prod (where
  // the user might want to switch accounts without reloading).
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "l") {
      e.preventDefault();
      showAuthModal();
    }
  });
}

init();

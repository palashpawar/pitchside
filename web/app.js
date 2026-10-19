// Pitchside client: renders the live match, overlays and personalised story.
(() => {
  const L = 105, W = 68;
  const $ = (id) => document.getElementById(id);
  const S = {
    meta: null, players: [], byId: {}, byIndex: [], duration: 6000,
    prev: null, cur: null, prevAt: 0, curAt: 0,
    state: null, cues: [], lower: null, speed: 8, paused: false,
    profile: { mode: "casual", club: null, player_id: null, language: "en", metric: null },
    cam: { x: L / 2, y: W / 2, z: 1 }, voice: false, azureTts: null, cueLog: [],
  };
  const KIND_LABEL = {
    goal: "Goal", big_chance: "Big chance", big_save: "Big save", press_shift: "Tactical shift",
    tempo_shift: "Rhythm change", chaos_spell: "Chaos", control_regained: "Control", momentum_swing: "Momentum",
    line_breaking_pass: "Line-breaker", speed_record: "Top speed", milestone: "Milestone", long_spell: "Possession",
  };

  // ---------- socket ----------
  let ws;
  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${location.search}`);
    ws.onmessage = (e) => { const { type, data } = JSON.parse(e.data); (H[type] || (() => {}))(data); };
    ws.onclose = () => setTimeout(connect, 1500);
  }
  const send = (msg) => ws && ws.readyState === 1 && ws.send(JSON.stringify(msg));

  const H = {
    meta(m) {
      S.meta = m; S.duration = m.duration;
      S.players = [...m.teams.home.players.map((p) => ({ ...p, team: "home" })), ...m.teams.away.players.map((p) => ({ ...p, team: "away" }))];
      S.byId = Object.fromEntries(S.players.map((p) => [p.id, p]));
      S.byIndex = S.players.slice().sort((a, b) => a.index - b.index);
      for (const side of ["home", "away"]) {
        const t = m.teams[side];
        $(`${side}-name`).textContent = t.name;
        $(`${side}-swatch`).style.background = t.color;
        document.documentElement.style.setProperty(`--${side}`, t.color);
      }
      $("provider").textContent = m.provider;
      const club = $("club");
      club.length = 1;
      for (const side of ["home", "away"]) club.add(new Option(m.teams[side].name, side));
      const pl = $("player");
      pl.length = 1;
      for (const side of ["home", "away"]) {
        const g = document.createElement("optgroup");
        g.label = m.teams[side].name;
        for (const p of m.teams[side].players) g.appendChild(new Option(`${p.number} ${p.name} (${p.position})`, p.id));
        pl.appendChild(g);
      }
      setSpeedButtons(m.speed);
      $("scrub-marks").innerHTML = "";
      $("feed").innerHTML = '<div class="empty">Insights appear here as the match unfolds.</div>';
      updateLangNote();
    },
    frame(f) {
      S.prev = S.cur; S.prevAt = S.curAt;
      S.cur = f; S.curAt = performance.now();
      $("scrub-fill").style.width = `${Math.min(100, (100 * f.t) / S.duration)}%`;
    },
    state(s) { S.state = s; renderState(s); },
    cue(c) {
      const life = Math.max(2500, Math.min(7000, ((c.t_out - c.t_in) / S.speed) * 1000 + 1500));
      S.cues.push({ ...c, until: performance.now() + life });
      logCue(c);
    },
    insight(c) {
      addInsight(c);
      logCue(c);
      if (!c.replayed && (c.priority >= 3 || c.involves_focus_player || S.profile.mode === "analyst" || !S.lower || performance.now() > S.lower.until)) {
        showLower(c);
        speak(c.line, c.language);
      }
    },
    event(e) {
      if (e.type === "goal") mark(e.t, "#fff");
      else if (e.type === "card") mark(e.t, "#f5c84b");
      if (["goal", "card", "period_end", "full_time"].includes(e.type)) addEventRow(e);
    },
    recap(r) { addRecap(r); },
    profile(p) { S.profile = p; syncControls(); },
    seeked() { S.cues = []; },
    done() {},
    error(e) { console.warn("server", e); },
  };

  // ---------- controls ----------
  function setSpeedButtons(v) {
    S.speed = v;
    document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", +b.dataset.speed === v));
  }
  document.querySelectorAll("#speeds button").forEach((b) => b.addEventListener("click", () => {
    setSpeedButtons(+b.dataset.speed); send({ type: "speed", value: S.speed });
  }));
  $("play").addEventListener("click", () => {
    S.paused = !S.paused; $("play").textContent = S.paused ? "▶" : "❚❚";
    $("play").setAttribute("aria-label", S.paused ? "Play" : "Pause");
    send({ type: "pause", value: S.paused });
  });
  $("scrub").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    send({ type: "seek", t: ((e.clientX - r.left) / r.width) * S.duration });
  });
  document.querySelectorAll("#mode-seg button").forEach((b) => b.addEventListener("click", () => {
    S.profile.mode = b.dataset.mode; pushProfile();
  }));
  for (const id of ["club", "player", "language", "metric"]) {
    $(id).addEventListener("change", (e) => {
      const key = id === "player" ? "player_id" : id;
      S.profile[key] = e.target.value || null;
      pushProfile();
    });
  }
  $("voice").addEventListener("change", (e) => { S.voice = e.target.checked; if (!S.voice) speechSynthesis?.cancel(); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    for (const t of ["story", "stats", "feed"]) $(`tab-${t}`).hidden = t !== b.dataset.tab;
  }));

  function pushProfile() {
    syncControls();
    $("feed").innerHTML = "";
    send({ type: "profile", value: S.profile });
    const q = new URLSearchParams(Object.entries(S.profile).filter(([, v]) => v));
    $("track-link").href = `/api/overlay-track?${q}`;
  }
  function syncControls() {
    document.querySelectorAll("#mode-seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.profile.mode));
    $("club").value = S.profile.club || "";
    $("player").value = S.profile.player_id || "";
    $("language").value = S.profile.language || "en";
    $("metric").value = S.profile.metric || "";
    const p = S.profile.player_id && S.byId[S.profile.player_id];
    $("focus-chip").hidden = !p;
    if (p) $("focus-chip").textContent = `Player focus · #${p.number} ${p.name}`;
    $("focus-card").hidden = !p;
    updateLangNote();
  }
  function updateLangNote() {
    const offline = S.meta && S.meta.provider === "offline";
    $("lang-note").hidden = !(offline && !["en", "es"].includes(S.profile.language));
  }

  // ---------- story feed ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function teamColor(side) { return S.meta?.teams[side]?.color || "#4fe3c1"; }

  function addInsight(c) {
    const feed = $("feed");
    feed.querySelector(".empty")?.remove();
    const el = document.createElement("article");
    el.className = "insight" + (c.involves_focus_player ? " focus" : "");
    const side = c.facts.team === S.meta.teams.home.name ? "home" : c.facts.team === S.meta.teams.away.name ? "away" : null;
    if (side) el.style.borderLeftColor = teamColor(side);
    const facts = Object.entries(c.facts).filter(([k]) => !["home_team", "away_team"].includes(k))
      .map(([k, v]) => `<span>${esc(k.replaceAll("_", " "))}</span><span>${esc(v)}</span>`).join("");
    const checks = c.checks.map((x) => `${x.stage} ${x.attempt}: ${x.issues.length ? "✗ " + esc(x.issues.join("; ")) : "✓ grounded"} (${x.source || ""})`).join("<br>");
    el.innerHTML = `
      <div class="meta"><span class="kind">${esc(KIND_LABEL[c.kind] || c.kind)} · ${esc(c.clock)}</span>
        <span>${c.verified ? "✓ verified" : "⚠ unverified"} · ${c.latency_ms ?? "–"} ms</span></div>
      <div class="h">${esc(c.overlay)}</div>
      <div class="l">${esc(c.line)}</div>
      ${S.profile.mode === "analyst" && c.language === "en" && c.why && c.why !== c.line ? `<div class="why">${esc(c.why)}</div>` : ""}
      <details><summary>Why? Show the evidence</summary>
        <div class="facts">${facts}</div>
        <ul class="evidence" data-ids="${c.evidence_ids.join(",")}"><li>loading events…</li></ul>
        <div class="checks">${checks}</div>
      </details>`;
    el.querySelector("details").addEventListener("toggle", (e) => { if (e.target.open) loadEvidence(e.target.querySelector(".evidence")); }, { once: true });
    feed.prepend(el);
    while (feed.children.length > 60) feed.lastChild.remove();
  }

  async function loadEvidence(ul) {
    const ids = ul.dataset.ids;
    if (!ids) { ul.innerHTML = "<li>Derived from rolling window metrics (no single event).</li>"; return; }
    const evs = await (await fetch(`/api/events?ids=${ids}${location.search.replace("?", "&")}`)).json();
    ul.innerHTML = evs.map((e) => {
      const who = S.byId[e.player_id]?.name || "";
      const to = e.recipient_id ? ` → ${S.byId[e.recipient_id]?.name}` : "";
      const out = e.outcome ? ` · ${e.outcome}` : "";
      return `<li>#${e.id} ${e.clock} ${e.type} · ${esc(who)}${esc(to)}${esc(out)}</li>`;
    }).join("") || "<li>No events</li>";
  }

  function addRecap(r) {
    const feed = $("feed");
    feed.querySelector(".empty")?.remove();
    const x = r.recap;
    const el = document.createElement("article");
    el.className = "insight recap";
    el.innerHTML = `<div class="meta"><span class="kind">${esc(r.label)} recap</span><span>${r.verified ? "✓ verified" : "⚠ unverified"}</span></div>
      <div class="h">${esc(x.headline)}</div><div class="l">${esc(x.summary)}</div>
      <ul>${(x.key_moments || []).map((k) => `<li>${esc(k)}</li>`).join("")}</ul>
      <div class="why"><b>Player of the match:</b> ${esc(x.player_of_the_match)}</div>
      <div class="why"><b>Social:</b> ${esc(x.social_post)}</div>`;
    feed.prepend(el);
    speak(x.summary, S.profile.language);
  }

  function addEventRow(e) {
    const feed = $("feed");
    feed.querySelector(".empty")?.remove();
    const el = document.createElement("div");
    el.className = "event-row";
    const label = { goal: "⚽ Goal", card: "🟨 Card", period_end: "Whistle", full_time: "Full time" }[e.type];
    el.innerHTML = `${esc(e.clock)} <b>${label}</b> ${esc(e.player || "")} ${e.score ? esc(e.score.join("–")) : ""}`;
    feed.prepend(el);
  }

  function mark(t, color) {
    const i = document.createElement("i");
    i.style.left = `${(100 * t) / S.duration}%`;
    i.style.background = color;
    $("scrub-marks").appendChild(i);
  }

  function showLower(c) {
    $("lower-third").hidden = false;
    $("lower-third").style.animation = "none"; void $("lower-third").offsetWidth; $("lower-third").style.animation = "";
    $("lt-kind").textContent = `${KIND_LABEL[c.kind] || c.kind} · ${c.clock}`;
    $("lt-headline").textContent = c.overlay;
    $("lt-line").textContent = c.line;
    $("lt-verified").textContent = c.verified ? "✓ verified against match data" : "⚠ unverified";
    S.lower = { until: performance.now() + (c.priority >= 3 ? 9000 : 6500) };
  }

  function logCue(c) {
    const slim = { id: c.id, lane: c.lane, type: c.type, t_in: c.t_in, t_out: c.t_out, anchor: c.anchor, priority: c.priority,
      payload: c.payload || { overlay: c.overlay, line: c.line, language: c.language }, evidence_ids: c.evidence_ids };
    S.cueLog.unshift(JSON.stringify(slim));
    S.cueLog.length = Math.min(S.cueLog.length, 40);
    if (!$("tab-feed").hidden) $("cue-log").textContent = S.cueLog.join("\n\n");
  }

  // ---------- voice ----------
  const VOICE_LANG = { en: "en-GB", es: "es-ES", fr: "fr-FR", pt: "pt-BR", de: "de-DE", ar: "ar-SA", hi: "hi-IN", ja: "ja-JP" };
  async function speak(text, lang) {
    if (!S.voice || !text) return;
    if (S.azureTts !== false) {
      try {
        const r = await fetch(`/api/tts?lang=${lang}&text=${encodeURIComponent(text)}`);
        if (r.ok) { S.azureTts = true; new Audio(URL.createObjectURL(await r.blob())).play(); return; }
        S.azureTts = false;
      } catch { S.azureTts = false; }
    }
    if (!window.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = VOICE_LANG[lang] || "en-GB";
    u.rate = 1.05;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }

  // ---------- state panels ----------
  function renderState(s) {
    $("score-h").textContent = s.score[0];
    $("score-a").textContent = s.score[1];
    $("clock").textContent = s.clock;
    $("period").textContent = s.period === 1 ? "1H" : "2H";
    const a = s.agents;
    $("provider").textContent = a.provider;
    $("agent-stats").textContent = `${a.moments} insights · ${a.first_pass_grounded} grounded first time · ${a.fallbacks} fallbacks` +
      (a.median_latency_ms != null ? ` · ${a.median_latency_ms} ms` : "");
    const ix = s.indices;
    if (ix && ix.home) {
      $("chaos-marker").style.left = `${ix.chaos}%`;
      $("chaos-val").textContent = ix.chaos >= 70 ? `Chaotic (${ix.chaos})` : ix.chaos <= 30 ? `Controlled (${ix.chaos})` : `Balanced (${ix.chaos})`;
      bars("press-bars", ix.home.press, ix.away.press);
      bars("tempo-bars", ix.home.tempo, ix.away.tempo);
    }
    drawMomentum(s.history || []);
    renderStats(s.teams);
    if (s.focus) renderFocus(s.focus);
  }

  function bars(id, h, a) {
    const t = S.meta.teams;
    $(id).innerHTML = [["home", h], ["away", a]].map(([side, v]) =>
      `<div class="bar"><span>${t[side].short}</span><div class="track"><div class="fill" style="width:${v}%;background:${t[side].color}"></div></div><b>${v}</b></div>`).join("");
  }

  function renderStats(t) {
    const rows = [
      ["Possession", "possession_pct", "%"], ["Shots", "shots", ""], ["On target", "on_target", ""], ["xG", "xg", ""],
      ["Pass accuracy", "pass_accuracy_pct", "%"], ["Passes", "passes", ""], ["Line-breaking passes", "line_breaking_passes", ""],
      ["Progressive passes", "progressive_passes", ""], ["Pressures", "pressures", ""], ["Tackles won", "tackles_won", ""],
      ["Interceptions", "interceptions", ""], ["Distance (km)", "distance_km", ""], ["Corners", "corners", ""],
    ];
    const m = S.meta.teams;
    $("stats-table").innerHTML = `<tr class="head"><td style="color:${m.home.color}">${m.home.short}</td><td></td><td style="color:${m.away.color}">${m.away.short}</td></tr>` +
      rows.map(([label, k, u]) => `<tr><td>${t.home[k] ?? "–"}${u}</td><td>${label}</td><td>${t.away[k] ?? "–"}${u}</td></tr>`).join("");
  }

  function renderFocus(p) {
    const color = teamColor(p.team);
    $("focus-card").innerHTML = `
      <div class="fc-num" style="background:${color}">${p.number}</div>
      <div><div class="fc-name">${esc(p.name)}</div><div class="fc-sub">${p.position} · ${esc(S.meta.teams[p.team].name)} · rating <b>${p.rating}</b></div></div>
      <div class="fc-grid">
        <div><b>${p.speed_kmh}</b><span>km/h now</span></div>
        <div><b>${p.top_speed_kmh}</b><span>top km/h</span></div>
        <div><b>${p.distance_km}</b><span>km run</span></div>
        <div><b>${p.sprints}</b><span>sprints</span></div>
        <div><b>${p.completed}/${p.passes}</b><span>passes</span></div>
        <div><b>${p.passes_over_expected > 0 ? "+" : ""}${p.passes_over_expected}</b><span>vs xPass</span></div>
        <div><b>${p.goals}/${p.assists}</b><span>goals/assists</span></div>
        <div><b>${p.xg}</b><span>xG</span></div>
      </div>`;
  }

  function drawMomentum(hist) {
    const c = $("momentum");
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth, h = 70;
    if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = "#233044"; g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    if (hist.length < 2) return;
    const t0 = hist[0].t, t1 = Math.max(hist[hist.length - 1].t, t0 + 600);
    const X = (t) => ((t - t0) / (t1 - t0)) * w, Y = (m) => h / 2 - (m / 100) * (h / 2 - 4);
    for (const [side, sign] of [["home", 1], ["away", -1]]) {
      g.beginPath(); g.moveTo(X(hist[0].t), h / 2);
      for (const p of hist) g.lineTo(X(p.t), Y(sign > 0 ? Math.max(0, p.momentum) : Math.min(0, p.momentum)));
      g.lineTo(X(hist[hist.length - 1].t), h / 2); g.closePath();
      g.fillStyle = teamColor(side) + "cc"; g.fill();
    }
    g.fillStyle = "#8a9bb2"; g.font = "10px Inter";
    g.fillText(S.meta.teams.home.short, 4, 11); g.fillText(S.meta.teams.away.short, 4, h - 4);
  }

  // ---------- pitch rendering ----------
  const canvas = $("pitch"), ctx = canvas.getContext("2d");
  function resize() {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  }
  new ResizeObserver(resize).observe(canvas);

  function positions() {
    if (!S.cur) return null;
    if (!S.prev) return S.cur.p;
    const span = Math.max(1, S.curAt - S.prevAt);
    const a = Math.min(1, (performance.now() - S.curAt) / span);
    return S.cur.p.map((q, i) => {
      const p = S.prev.p[i];
      if (Math.hypot(q[0] - p[0], q[1] - p[1]) > 8) return q;
      return [p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a];
    });
  }

  function draw() {
    requestAnimationFrame(draw);
    const cw = canvas.width, ch = canvas.height;
    if (!cw) return;
    const pos = positions();
    const focus = S.profile.player_id ? S.byId[S.profile.player_id] : null;
    const target = focus && pos ? { x: pos[focus.index][0], y: pos[focus.index][1], z: 1.9 } : { x: L / 2, y: W / 2, z: 1 };
    S.cam.x += (target.x - S.cam.x) * 0.06; S.cam.y += (target.y - S.cam.y) * 0.06; S.cam.z += (target.z - S.cam.z) * 0.05;
    const base = Math.min(cw / (L + 6), ch / (W + 6));
    const k = base * S.cam.z;
    const vx = Math.max(Math.min(S.cam.x, L + 3 - cw / k / 2), -3 + cw / k / 2);
    const vy = Math.max(Math.min(S.cam.y, W + 3 - ch / k / 2), -3 + ch / k / 2);
    const cx = S.cam.z > 1.01 ? vx : L / 2, cy = S.cam.z > 1.01 ? vy : W / 2;
    const P = (x, y) => [cw / 2 + (x - cx) * k, ch / 2 - (y - cy) * k];

    ctx.fillStyle = "#1a6239"; ctx.fillRect(0, 0, cw, ch);
    for (let i = 0; i < 14; i++) {
      if (i % 2) continue;
      const [x0, y0] = P((i * L) / 14, W); const [x1] = P(((i + 1) * L) / 14, 0);
      ctx.fillStyle = "#1d6b3f"; ctx.fillRect(x0, Math.min(y0, P(0, 0)[1]) - 2000, x1 - x0, 6000);
    }
    ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = Math.max(1, k * 0.12);
    const rect = (x, y, w, h) => { const [a, b] = P(x, y + h); ctx.strokeRect(a, b, w * k, h * k); };
    rect(0, 0, L, W); rect(0, W / 2 - 20.16, 16.5, 40.32); rect(L - 16.5, W / 2 - 20.16, 16.5, 40.32);
    rect(0, W / 2 - 9.16, 5.5, 18.32); rect(L - 5.5, W / 2 - 9.16, 5.5, 18.32);
    ctx.beginPath(); ctx.moveTo(...P(L / 2, 0)); ctx.lineTo(...P(L / 2, W)); ctx.stroke();
    ctx.beginPath(); ctx.arc(...P(L / 2, W / 2), 9.15 * k, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.85)";
    for (const gx of [0, L]) { const [a, b] = P(gx - (gx ? 0 : 1.2), W / 2 + 3.66); ctx.fillRect(a, b, 1.2 * k, 7.32 * k); }
    if (!pos || !S.meta) return;

    const now = performance.now();
    S.cues = S.cues.filter((c) => c.until > now);
    const ball = pos[22];
    let carrier = -1, best = 1.6;
    for (let i = 0; i < 22; i++) { const d = Math.hypot(pos[i][0] - ball[0], pos[i][1] - ball[1]); if (d < best) { best = d; carrier = i; } }

    const r = Math.max(5, 1.15 * k);
    for (let i = 0; i < 22; i++) {
      const p = S.byIndex[i]; const [x, y] = P(...pos[i]);
      const t = S.meta.teams[p.team];
      const isFocus = focus && focus.index === i;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = p.position === "GK" ? "#e6c84a" : t.color; ctx.fill();
      ctx.lineWidth = isFocus ? 3 : 1.5; ctx.strokeStyle = isFocus ? "#ffd166" : "rgba(0,0,0,.45)"; ctx.stroke();
      if (isFocus) { ctx.beginPath(); ctx.arc(x, y, r + 6 + 2 * Math.sin(now / 200), 0, Math.PI * 2); ctx.strokeStyle = "rgba(255,209,102,.6)"; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.fillStyle = p.position === "GK" ? "#111" : "#fff";
      ctx.font = `700 ${Math.round(r * 0.95)}px Inter`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(p.number, x, y + 0.5);
      if (i === carrier || isFocus) tag(x, y - r - 12, p.name.split(" ").slice(-1)[0], isFocus ? "#ffd166" : "#fff");
    }
    const [bx, by] = P(...ball);
    ctx.beginPath(); ctx.arc(bx, by, Math.max(3.5, 0.55 * k), 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.strokeStyle = "#111"; ctx.lineWidth = 1; ctx.stroke();

    for (const c of S.cues) {
      const pid = c.anchor?.player_id; if (!pid || !S.byId[pid]) continue;
      if (c.type === "distance_milestone" && pid !== S.profile.player_id) continue;
      const idx = S.byId[pid].index; const [x, y] = P(...pos[idx]);
      const pl = c.payload;
      const txt = {
        speed_alert: `⚡ ${pl.speed_kmh} km/h${pl.match_record ? " · match record" : ""}`,
        shot_badge: `Shot · ${pl.xg} xG${pl.shot_speed_kmh >= 40 ? ` · ${pl.shot_speed_kmh} km/h` : ""} · ${String(pl.outcome).replace("_", " ")}`,
        pass_badge: `Pass ${pl.distance_m} m · difficulty ${pl.difficulty}${pl.bypassed >= 3 ? ` · ${pl.bypassed} bypassed` : ""}`,
        card: "🟨 Yellow card",
        distance_milestone: `${pl.km} km covered`,
      }[c.type];
      if (txt) badge(x, y + r + 16, txt, c.type === "shot_badge" ? "#ffb547" : c.type === "speed_alert" ? "#4fe3c1" : "#e8eef6");
    }
    ctx.textAlign = "right"; ctx.textBaseline = "top"; ctx.font = "600 12px JetBrains Mono"; ctx.fillStyle = "rgba(255,255,255,.75)";
    ctx.fillText(`ball ${S.cur.bs} km/h`, cw - 12, 10);

    if (S.lower && now > S.lower.until) { $("lower-third").hidden = true; S.lower = null; }
  }

  function tag(x, y, text, color) {
    ctx.font = "600 12px Inter"; const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = "rgba(8,12,18,.8)"; ctx.fillRect(x - w / 2, y - 9, w, 18);
    ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y);
  }
  function badge(x, y, text, color) {
    ctx.font = "600 12px Inter"; const w = ctx.measureText(text).width + 14;
    ctx.fillStyle = "rgba(8,12,18,.9)"; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(x - w / 2, y - 11, w, 22, 11); ctx.fill(); ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y);
  }

  connect();
  requestAnimationFrame(draw);
})();

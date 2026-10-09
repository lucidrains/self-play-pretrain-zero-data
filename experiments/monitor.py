"""live dashboard for a running replicate_table1.py - serves the run's live.json

  python experiments/monitor.py --live experiments/runs/default/live.json --port 8000

panels: discovery timeline vs uniform prior, learner loss, archive elites with
appendix C family tags, and the latest generator proposals
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fire import Fire

from self_play_pretrain_zero_data.executors.base import default

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>self-play monitor</title>
<style>
:root {
  --bg:#0b0d10; --ink:#e8eaed; --dim:#868e96; --line:#20242b;
  --live:#4ade80; --done:#a3a3a3;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
* { box-sizing:border-box }
body { margin:0; height:100vh; display:grid; grid-template-columns:360px 1fr; grid-template-rows:52px 1fr;
  background:var(--bg); color:var(--ink); font:12px/1.5 var(--sans); }
header { grid-column:1/-1; display:flex; align-items:center; gap:10px; padding:0 20px; border-bottom:1px solid var(--line); }
h1 { margin:0; font-size:13px; font-weight:600; letter-spacing:.01em; }
h2 { margin:0; font-size:10.5px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:var(--dim); }
#run, #stats, #scope, #shown { font:11px var(--mono); color:var(--dim); }
#ablate { font:10.5px var(--mono); color:#e3b341; }
#stats { margin-left:auto; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.dot { width:6px; height:6px; border-radius:50%; background:var(--done); flex:none; }
.dot.run { background:var(--live); }
#left { overflow-y:auto; padding:18px 20px; border-right:1px solid var(--line); }
#left h2 { margin:0 0 2px; }
#discoveries, #recent { margin-bottom:22px; }
.row { display:flex; align-items:center; gap:8px; min-height:26px; }
.fam { font-weight:500; }
.meta { margin-left:auto; font:11px var(--mono); color:var(--dim); white-space:nowrap; }
.prog, .bytes { font:11.5px var(--mono); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prog { color:var(--ink); }
.bytes { color:var(--dim); margin-bottom:6px; }
#right { display:grid; grid-template-rows:212px 1fr; overflow:hidden; }
#losswrap { padding:16px 20px 0; }
#losswrap h2 { margin-bottom:8px; }
canvas { width:100%; height:160px; display:block; }
#archivewrap { overflow-y:auto; padding:0 20px 24px; }
.toolbar { display:flex; align-items:baseline; gap:12px; padding:12px 0 6px; }
#filter { margin-left:auto; background:none; border:0; border-bottom:1px solid var(--line); color:var(--ink);
  font:11px var(--mono); padding:2px 0; width:200px; outline:none; }
#filter:focus { border-color:var(--dim); }
table { width:100%; border-collapse:collapse; }
th { text-align:left; font:10.5px var(--sans); font-weight:500; letter-spacing:.08em; text-transform:uppercase; color:var(--dim);
  padding:4px 10px 6px 0; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--bg); }
td { padding:7px 10px 7px 0; border-bottom:1px solid var(--line); vertical-align:middle; }
td.num { text-align:right; white-space:nowrap; font:11px var(--mono); }
.cellfam { display:flex; align-items:center; gap:7px; white-space:nowrap; font-size:11px; }
.f-arithmetic { background:#6ea8fe; } .f-quadratic { background:#56d364; } .f-cubic { background:#f0883e; }
.f-fibonacci { background:#bc8cff; } .f-geometric { background:#e3b341; } .f-none { background:#3d444d; }
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>self-play monitor</h1>
  <span id="run"></span>
  <span id="ablate"></span>
  <span id="stats">loading …</span>
</header>

<div id="left">
  <h2>discoveries</h2>
  <div id="discoveries"></div>
  <h2>latest proposals</h2>
  <div id="recent"></div>
</div>

<div id="right">
  <div id="losswrap">
    <h2>learner loss <span id="scope"></span></h2>
    <canvas id="loss"></canvas>
  </div>
  <div id="archivewrap">
    <div class="toolbar"><h2>archive elites</h2><span id="shown"></span><input id="filter" placeholder="filter programs"></div>
    <table>
      <thead><tr><th>family</th><th>program</th><th>output bytes</th><th style="text-align:right">reward</th><th style="text-align:right">len / loops</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</div>

<script>
const FAMILIES = ['arithmetic', 'quadratic', 'cubic', 'fibonacci', 'geometric'];
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text != null) node.textContent = text; return node; };
const fmt = n => (n == null) ? '—' : Number(n).toLocaleString('en-US', { maximumFractionDigits:0 });
const bytes = arr => (arr && arr.length) ? arr.join(' ') : '·';
const dot = family => el('span', 'dot f-' + (family || 'none'));
const clip = (node, text) => { node.textContent = text; node.title = text; return node; };
let state = null;

function renderDiscoveries(s) {
  const box = $('discoveries'); box.replaceChildren();
  for (const family of FAMILIES) {
    const record = s.discoveries[family];
    const expected = s.baseline.expected_round[family];
    const row = el('div', 'row');
    row.append(dot(family), el('span', 'fam', family));
    const hits = (s.sample_hits || {})[family] || 0;
    const meta = record ? `round ${record.round} · ${hits} hit${hits === 1 ? '' : 's'}` : 'not yet';
    row.append(el('span', 'meta', expected != null ? `${meta} · uniform ${s.baseline.hits[family] > 0 ? '≈' : '>'} ${fmt(expected)}` : meta));
    box.append(row);
    if (record) box.append(clip(el('div', 'prog'), record.program), el('div', 'bytes', 'output  ' + bytes(record.output_head)));
  }
}

function renderRecent(s) {
  const box = $('recent'); box.replaceChildren();
  for (const sample of (s.recent || []).slice().reverse()) {
    const row = el('div', 'row');
    row.append(dot(sample.family), clip(el('span', 'prog', ''), sample.program));
    box.append(row, el('div', 'bytes', 'output  ' + bytes(sample.output_head)));
  }
}

function renderArchive(s) {
  const filter = $('filter').value;
  const rows = (s.archive_rows || []).filter(r => !filter || r.program.includes(filter));
  const body = $('rows'); body.replaceChildren();
  for (const r of rows.slice(0, 80)) {
    const tr = el('tr');
    const fam = el('td'); const cell = el('span', 'cellfam'); cell.append(dot(r.family), el('span', null, r.family || '—')); fam.append(cell);
    const program = el('td'); program.append(clip(el('div', 'prog'), r.program));
    tr.append(fam, program, el('td', 'bytes', bytes(r.output_head)));
    tr.append(el('td', 'num', r.reward.toFixed(2)), el('td', 'num', `${r.length} / ${fmt(r.loops)}`));
    body.append(tr);
  }
  $('shown').textContent = `${rows.length} shown · archive ${fmt(s.archive_size)} · sorted by learning reward`;
}

function drawLoss(s) {
  const canvas = $('loss');
  const dpr = devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * dpr; canvas.height = height * dpr;
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const losses = s.losses || []; if (losses.length < 2) return;
  const pad = { l:30, r:4, t:8, b:16 };
  const lo = Math.min(...losses), hi = Math.max(...losses), span = (hi - lo) || 1;
  const x = i => pad.l + (width - pad.l - pad.r) * i / (losses.length - 1);
  const y = v => pad.t + (height - pad.t - pad.b) * (1 - (v - lo) / span);
  ctx.font = '10px ui-monospace'; ctx.fillStyle = '#868e96';
  ctx.fillText(hi.toFixed(1), 0, y(hi) + 3); ctx.fillText(lo.toFixed(1), 0, y(lo) + 3);
  ctx.strokeStyle = '#20242b'; ctx.beginPath(); ctx.moveTo(pad.l, height - pad.b); ctx.lineTo(width - pad.r, height - pad.b); ctx.stroke();
  ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 1; ctx.beginPath();
  losses.forEach((loss, i) => i ? ctx.lineTo(x(i), y(loss)) : ctx.moveTo(x(0), y(loss)));
  ctx.stroke();
  const colors = { arithmetic:'#6ea8fe', quadratic:'#56d364', cubic:'#f0883e', fibonacci:'#bc8cff', geometric:'#e3b341' };
  for (const family of FAMILIES) {
    const record = s.discoveries[family];
    if (!record || record.round >= losses.length) continue;
    ctx.strokeStyle = colors[family]; ctx.globalAlpha = 0.5; ctx.setLineDash([1, 4]);
    ctx.beginPath(); ctx.moveTo(x(record.round), pad.t); ctx.lineTo(x(record.round), height - pad.b); ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }
}

function render(s) {
  if (!s || !s.config) { $('stats').textContent = 'waiting for live.json …'; return; }
  state = s;
  const c = s.config, last = s.losses && s.losses.length ? s.losses[s.losses.length - 1] : null;
  $('run').textContent = `${c.dim}d × ${c.depth}L · ${((s.num_parameters || 0) / 1e6).toFixed(2)}M · tape ${c.max_output} · ${c.reference} · seed ${c.seed}`;
  const flags = [];
  if (c.ablate_generator !== 'none') flags.push(`generator=${c.ablate_generator}`);
  if (c.ablate_reward !== 'none') flags.push(`reward=${c.ablate_reward}`);
  if (c.ablate_proposals !== 'none') flags.push(`proposals=${c.ablate_proposals}`);
  $('ablate').textContent = flags.length ? `ablate · ${flags.join(' · ')}` : '';
  $('dot').className = 'dot' + (s.status === 'finished' ? '' : ' run');
  $('stats').textContent = `round ${fmt(s.round)} · ${(+s.elapsed_min).toFixed(1)}/${c.minutes} min · loss ${last == null ? '—' : last.toFixed(3)} · archive ${fmt(s.archive_size)} · ${c.detect_samples}/round`;
  $('scope').textContent = '· mod 256 recurrences, period ≥ 30';
  renderDiscoveries(s); renderRecent(s); renderArchive(s); drawLoss(s);
}

async function poll() {
  try { render(await (await fetch('/state', { cache:'no-store' })).json()); }
  catch (error) { $('stats').textContent = 'waiting for the monitor server …'; }
  setTimeout(poll, 2000);
}

$('filter').addEventListener('input', () => state && renderArchive(state));
window.addEventListener('resize', () => state && drawLoss(state));
poll();
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
        elif self.path == '/state':
            live = self.server.live_path
            body = live.read_bytes() if live.exists() else b'{}'
            content_type = 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

def serve(
    live = 'experiments/runs/default/live.json',
    host = '127.0.0.1',
    port = 8000,
    label = None
):
    server = ThreadingHTTPServer((host, port), Handler)
    server.live_path = Path(live)
    server.label = default(label, server.live_path.parent.name)

    print(f'monitor on http://{host}:{port} - watching {server.live_path} (label {server.label})')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    Fire(serve)

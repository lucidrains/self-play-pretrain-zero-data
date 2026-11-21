"""live dashboard for replicate_table1.py and discover_nca_rules.py

  python experiments/monitor.py --live experiments/runs/default/live.json --port 8000
  python experiments/monitor.py --live runs/bf/live.json --nca runs/nca/live.json --port 8000

auto-detects Brainfuck/Forth Table 1 runs and NeuralCellularAutomata (`nca.py`)
rule-discovery runs from live.json
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fire import Fire

from self_play_pretrain_zero_data.executors.base import default

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-Play Pretraining · Monitor</title>
<style>
:root {
  --bg: #f7f7f5;
  --surface: #ffffff;
  --surface-2: #f2f1ed;
  --hover: #f6f5f1;
  --border: #e6e4de;
  --border-subtle: #efede8;
  --ink: #141619;
  --ink-2: #4a515b;
  --dim: #7b828e;
  --live: #16a34a;
  --live-ring: rgba(22, 163, 74, 0.22);
  --done: #8b93a0;
  --loss-ema: #18202c;
  --loss-raw: rgba(100, 116, 139, 0.34);
  --loss-fill-top: rgba(24, 32, 44, 0.08);
  --loss-fill-bot: rgba(24, 32, 44, 0.0);
  --grid: #eceae4;
  --prior: #64748b;
  --bar-bg: #ebe9e3;
  --warn-bg: #fef7e0;
  --warn-ink: #92400e;
  --warn-border: #f3d27a;
  --backdrop: rgba(15, 18, 24, 0.28);
  --shadow-sm: 0 1px 2px rgba(15, 18, 24, 0.04);
  --shadow-lg: 0 12px 36px rgba(15, 18, 24, 0.14);
  --cell-off: #efeee9;

  --f-arithmetic: #2563eb;
  --f-quadratic: #0d9488;
  --f-cubic: #d95f26;
  --f-fibonacci: #7c3aed;
  --f-geometric: #b45309;
  --f-none: #9ca3af;

  --c-complex: #2563eb;
  --c-periodic: #0d9488;
  --c-chaotic: #d95f26;
  --c-frozen: #94a3b8;

  --tk-macro: #6d28d9;
  --tk-loop: #0369a1;
  --tk-io: #b45309;
  --tk-num: #0d9488;
  --tk-dead: #a3a9b3;

  --sans: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  --mono: "SF Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

html[data-theme="dark"] {
  --bg: #0b0d11;
  --surface: #11141b;
  --surface-2: #171b24;
  --hover: #191e29;
  --border: #222834;
  --border-subtle: #1a1f29;
  --ink: #edf0f5;
  --ink-2: #a8b1c0;
  --dim: #6f7889;
  --live: #4ade80;
  --live-ring: rgba(74, 222, 128, 0.25);
  --done: #64748b;
  --loss-ema: #e2e8f0;
  --loss-raw: rgba(148, 163, 184, 0.28);
  --loss-fill-top: rgba(226, 232, 240, 0.10);
  --loss-fill-bot: rgba(226, 232, 240, 0.0);
  --grid: #1b202b;
  --prior: #94a3b8;
  --bar-bg: #1e2430;
  --warn-bg: rgba(234, 179, 8, 0.12);
  --warn-ink: #facc15;
  --warn-border: rgba(234, 179, 8, 0.32);
  --backdrop: rgba(0, 0, 0, 0.55);
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.25);
  --shadow-lg: 0 18px 48px rgba(0, 0, 0, 0.55);
  --cell-off: #191f2a;

  --f-arithmetic: #60a5fa;
  --f-quadratic: #2dd4bf;
  --f-cubic: #fb923c;
  --f-fibonacci: #c084fc;
  --f-geometric: #facc15;
  --f-none: #525c6e;

  --c-complex: #60a5fa;
  --c-periodic: #2dd4bf;
  --c-chaotic: #fb923c;
  --c-frozen: #64748b;

  --tk-macro: #c084fc;
  --tk-loop: #38bdf8;
  --tk-io: #fbbf24;
  --tk-num: #2dd4bf;
  --tk-dead: #4b5563;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--ink);
  font: 13px/1.5 var(--sans);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

button, input { font: inherit; color: inherit; }

/* Header */
.topbar {
  position: sticky;
  top: 0;
  z-index: 30;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}

.topbar-inner {
  max-width: 1400px;
  margin: 0 auto;
  height: 52px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--done);
  flex: none;
}

.status-dot.run {
  background: var(--live);
  box-shadow: 0 0 0 3px var(--live-ring);
  animation: pulse 2.2s ease-in-out infinite;
}

.status-dot.wait { background: var(--f-geometric); }

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 2px var(--live-ring); }
  50% { box-shadow: 0 0 0 5px rgba(22, 163, 74, 0.05); }
}

.title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: -0.015em;
  white-space: nowrap;
}

.sep { width: 1px; height: 14px; background: var(--border); flex: none; }

.run-spec {
  font: 11.5px var(--mono);
  color: var(--dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ablate-pill {
  display: none;
  font: 11px var(--mono);
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--warn-bg);
  color: var(--warn-ink);
  border: 1px solid var(--warn-border);
  white-space: nowrap;
}

.ablate-pill.show { display: inline-flex; align-items: center; }

.top-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  flex: none;
}

.progress-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  font: 11.5px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--ink-2);
}

.progress-track {
  width: 68px;
  height: 4px;
  border-radius: 999px;
  background: var(--bar-bg);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  width: 0%;
  background: var(--ink);
  border-radius: 999px;
  transition: width 0.35s ease;
}

.seg {
  display: inline-flex;
  padding: 2px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 7px;
  gap: 1px;
}

.seg button {
  border: 0;
  background: transparent;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 500;
  color: var(--dim);
  border-radius: 5px;
  cursor: pointer;
  transition: color 0.12s, background 0.12s;
  line-height: 1.2;
  white-space: nowrap;
}

.seg button:hover { color: var(--ink); }

.seg button.active {
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow-sm);
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--ink-2);
  font-size: 11.5px;
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s, background 0.12s;
  white-space: nowrap;
}

.btn.sm { height: 24px; padding: 0 8px; font-size: 11px; }
.btn:hover { border-color: var(--dim); color: var(--ink); background: var(--hover); }
.btn.icon-only { width: 28px; padding: 0; justify-content: center; }

/* Main container */
.main {
  max-width: 1400px;
  margin: 0 auto;
  padding: 18px 24px 36px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.banner {
  display: none;
  padding: 10px 14px;
  border-radius: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  font: 12px var(--mono);
  color: var(--dim);
  align-items: center;
  gap: 10px;
}
.banner.show { display: flex; }

/* KPI Strip */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

.kpi {
  padding: 13px 18px;
  border-right: 1px solid var(--border-subtle);
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}
.kpi:last-child { border-right: 0; }

.kpi-label {
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
}

.kpi-value {
  font-size: 21px;
  font-weight: 600;
  letter-spacing: -0.025em;
  font-variant-numeric: tabular-nums;
  line-height: 1.25;
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.kpi-unit { font-size: 12px; font-weight: 400; color: var(--dim); letter-spacing: 0; }

.kpi-sub {
  font: 11.5px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: flex;
  align-items: center;
  gap: 6px;
}

.kpi-delta.good { color: var(--live); font-weight: 600; }

.mini-dots { display: inline-flex; align-items: center; gap: 4px; }
.mini-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--border); }

/* Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.card-head {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.card-title { margin: 0; font-size: 12.5px; font-weight: 600; letter-spacing: -0.01em; }
.card-sub { font-size: 11.5px; color: var(--dim); }
.card-tools { margin-left: auto; display: flex; align-items: center; gap: 10px; }

.legend { display: inline-flex; align-items: center; gap: 11px; font-size: 11px; color: var(--dim); }
.legend-item { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }

.swatch-line { width: 14px; height: 2px; border-radius: 1px; background: var(--loss-ema); }
.swatch-line.raw { height: 1.5px; background: var(--loss-raw); }
.swatch-line.dash { border-top: 1.5px dashed var(--dim); background: none; height: 0; }
.swatch-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--f-arithmetic); }
.swatch-sq { width: 8px; height: 8px; border-radius: 2px; }
.swatch-diamond { width: 7px; height: 7px; border: 1.5px solid var(--prior); transform: rotate(45deg); display: inline-block; }

/* Charts Row */
.charts-grid {
  display: grid;
  grid-template-columns: 1.22fr 1fr;
  gap: 16px;
}

.chart-body { position: relative; padding: 10px 14px 12px; height: 226px; }

#primary-canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
#secondary-svg { width: 100%; height: 100%; display: block; }

.chart-tooltip {
  position: absolute;
  pointer-events: none;
  z-index: 10;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 10px;
  box-shadow: var(--shadow-lg);
  font: 11px/1.45 var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--ink);
  opacity: 0;
  transition: opacity 0.1s ease;
  white-space: nowrap;
}
.chart-tooltip.show { opacity: 1; }
.tt-head { font-weight: 600; margin-bottom: 2px; }
.tt-row { display: flex; justify-content: space-between; gap: 14px; color: var(--ink-2); }
.tt-disc { margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 6px; font-weight: 500; }

/* Section Eyebrow */
.section-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 2px 2px 0;
}

.eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
}

.section-right { display: flex; align-items: center; gap: 10px; }
.section-note { font: 11px var(--mono); color: var(--dim); }

/* 5-card Discovery / NCA Frontier Grid */
.families-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}

.fam-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 2px solid var(--border);
  border-radius: 10px;
  padding: 12px 13px 11px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.15s, transform 0.15s;
  min-width: 0;
}

.fam-card.found { cursor: pointer; }
.fam-card.found:hover { border-color: var(--dim); transform: translateY(-1px); }
.fam-card.pending { opacity: 0.78; }

.fam-top { display: flex; align-items: center; gap: 7px; }
.dot { width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--f-none); }
.fam-name { font-size: 12.5px; font-weight: 600; letter-spacing: -0.01em; }
.fam-name.cap { text-transform: capitalize; }
.fam-name.mono { font-family: var(--mono); }

.speed-pill {
  margin-left: auto;
  font: 11px var(--mono);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  padding: 1px 6px;
  border-radius: 5px;
  background: var(--surface-2);
  color: var(--ink);
  white-space: nowrap;
}
.speed-pill.muted { font-weight: 400; color: var(--dim); }

.fam-meta {
  font: 11px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.prog-chip {
  font: 11.5px var(--mono);
  background: var(--surface-2);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 5px 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--ink);
}
.prog-chip.empty { color: var(--dim); border-style: dashed; background: transparent; }

.fam-wave { height: 28px; display: flex; align-items: center; }
.fam-wave svg { width: 100%; height: 28px; display: block; }

.fam-bytes {
  font: 10.5px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* NCA Canvas Preview inside Frontier Cards */
.nca-stage {
  display: flex;
  justify-content: center;
  align-items: center;
  background: var(--surface-2);
  border: 1px solid var(--border-subtle);
  border-radius: 7px;
  padding: 6px;
  height: 112px;
}

.nca-stage canvas {
  image-rendering: pixelated;
  border-radius: 3px;
  display: block;
}

.bit-strip {
  display: grid;
  grid-template-columns: repeat(16, 1fr);
  gap: 2px;
  height: 6px;
}
.bit-strip.b8 { grid-template-columns: repeat(8, 1fr); }
.bit-cell { border-radius: 1.5px; background: var(--bar-bg); }
.bit-cell.on { background: var(--ink-2); }

/* Bottom Workspace */
.workspace-grid {
  display: grid;
  grid-template-columns: 1.85fr 1fr;
  gap: 16px;
  align-items: start;
}

.filter-bar {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: transparent;
  font-size: 11px;
  font-weight: 500;
  color: var(--ink-2);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.pill:hover { background: var(--hover); color: var(--ink); }
.pill.active { background: var(--ink); border-color: var(--ink); color: var(--surface); }
.pill.zero { opacity: 0.45; }
.pill .cnt { font: 10.5px var(--mono); opacity: 0.72; }

.search-wrap { position: relative; display: flex; align-items: center; }

#filter {
  width: 196px;
  height: 26px;
  padding: 0 24px 0 9px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  font: 11.5px var(--mono);
  color: var(--ink);
  outline: none;
  transition: border-color 0.12s, background 0.12s;
}
#filter:focus { border-color: var(--dim); background: var(--surface); }
#filter::placeholder { color: var(--dim); }

.search-clear {
  position: absolute;
  right: 5px;
  border: 0;
  background: none;
  color: var(--dim);
  cursor: pointer;
  font-size: 13px;
  line-height: 1;
  display: none;
  padding: 2px 4px;
}
.search-clear.show { display: block; }

/* Table */
.table-scroll { overflow-x: auto; overflow-y: auto; max-height: 452px; }

table { width: 100%; border-collapse: collapse; }

thead th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--surface);
  text-align: left;
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  user-select: none;
}
thead th.sortable { cursor: pointer; }
thead th.sortable:hover { color: var(--ink); }
thead th.num { text-align: right; }

.sort-arrow { display: inline-block; margin-left: 3px; font-size: 10px; opacity: 0.85; }

tbody tr { border-bottom: 1px solid var(--border-subtle); cursor: pointer; transition: background 0.08s; }
tbody tr:last-child { border-bottom: 0; }
tbody tr:hover { background: var(--hover); }
tbody td { padding: 7.5px 12px; vertical-align: middle; }

td.num {
  text-align: right;
  font: 11.5px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--ink-2);
  white-space: nowrap;
}

.fam-cell {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 11.5px;
  font-weight: 500;
  white-space: nowrap;
  color: var(--ink-2);
}

.prog-inline {
  font: 11.5px var(--mono);
  max-width: 220px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.out-cell { display: flex; align-items: center; gap: 9px; min-width: 150px; }
.spark-mini { flex: none; display: block; }
.thumb-canvas { image-rendering: pixelated; border-radius: 3px; border: 1px solid var(--border-subtle); flex: none; display: block; }

.bytes-inline {
  font: 11px var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--dim);
  max-width: 165px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.reward-cell { display: inline-flex; align-items: center; justify-content: flex-end; gap: 8px; }
.reward-bar { width: 34px; height: 4px; border-radius: 999px; background: var(--bar-bg); overflow: hidden; flex: none; }
.reward-fill { height: 100%; background: var(--ink); border-radius: 999px; }

.empty-state { padding: 34px 20px; text-align: center; font: 12px var(--mono); color: var(--dim); }

/* Feed */
.feed { display: flex; flex-direction: column; }

.feed-item {
  padding: 9.5px 16px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  flex-direction: column;
  gap: 5px;
  cursor: pointer;
  transition: background 0.08s;
}
.feed-item:last-child { border-bottom: 0; }
.feed-item:hover { background: var(--hover); }

.feed-top { display: flex; align-items: center; gap: 8px; min-width: 0; }
.feed-prog { font: 11.5px var(--mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
.feed-tag { font: 10px var(--mono); padding: 1px 6px; border-radius: 4px; background: var(--surface-2); color: var(--ink-2); flex: none; }
.feed-bot { display: flex; align-items: center; gap: 8px; padding-left: 15px; }
.feed-bot .bytes-inline { max-width: none; flex: 1; }

/* Syntax Tokens */
.tk-macro { color: var(--tk-macro); font-weight: 600; }
.tk-loop  { color: var(--tk-loop); font-weight: 500; }
.tk-io    { color: var(--tk-io); font-weight: 600; }
.tk-num   { color: var(--tk-num); }
.tk-dead  { color: var(--tk-dead); text-decoration: line-through; text-decoration-color: rgba(128,128,128,0.35); }

/* Semantic Colors */
.c-arithmetic { --fam-c: var(--f-arithmetic); }
.c-quadratic  { --fam-c: var(--f-quadratic); }
.c-cubic      { --fam-c: var(--f-cubic); }
.c-fibonacci  { --fam-c: var(--f-fibonacci); }
.c-geometric  { --fam-c: var(--f-geometric); }
.c-none       { --fam-c: var(--f-none); }
.c-complex    { --fam-c: var(--c-complex); }
.c-periodic   { --fam-c: var(--c-periodic); }
.c-chaotic    { --fam-c: var(--c-chaotic); }
.c-frozen     { --fam-c: var(--c-frozen); }

.dot[class*="c-"] { background: var(--fam-c); }

/* Drawer & Modal */
.backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  background: var(--backdrop);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s ease;
}
.backdrop.show { opacity: 1; pointer-events: auto; }

.drawer {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  z-index: 60;
  width: min(470px, 94vw);
  background: var(--surface);
  border-left: 1px solid var(--border);
  box-shadow: var(--shadow-lg);
  transform: translateX(100%);
  transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1);
  display: flex;
  flex-direction: column;
}
.drawer.open { transform: translateX(0); }

.drawer-head {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}

.drawer-title { font-size: 13.5px; font-weight: 600; }
.drawer-sub { font: 11px var(--mono); color: var(--dim); }

.drawer-close {
  margin-left: auto;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  color: var(--dim);
}
.drawer-close:hover { color: var(--ink); background: var(--hover); }

.drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 18px 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.insp-section { display: flex; flex-direction: column; gap: 7px; }
.insp-label-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.insp-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--dim); }

.code-box {
  font: 12px/1.6 var(--mono);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  word-break: break-all;
}

.macro-list { display: flex; flex-wrap: wrap; gap: 6px; }
.macro-chip { font: 11px var(--mono); padding: 3px 7px; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--border-subtle); color: var(--ink-2); }
.macro-chip b { color: var(--tk-macro); }

.kv-mini { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.kv-cell { padding: 8px 10px; border-right: 1px solid var(--border-subtle); }
.kv-cell:last-child { border-right: 0; }
.kv-cell .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); }
.kv-cell .v { font: 13px var(--mono); font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }

.recurrence-note { font: 11.5px var(--mono); padding: 7px 10px; border-radius: 7px; background: var(--surface-2); border: 1px solid var(--border-subtle); color: var(--ink-2); }

.diff-table { width: 100%; border-collapse: collapse; font: 11px var(--mono); font-variant-numeric: tabular-nums; }
.diff-table th { text-align: right; font-size: 10px; font-weight: 600; color: var(--dim); padding: 5px 8px; border-bottom: 1px solid var(--border); position: static; }
.diff-table th:first-child, .diff-table td:first-child { text-align: left; }
.diff-table td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--border-subtle); color: var(--ink-2); }
.diff-table tr { cursor: default; }
.diff-table tr:hover { background: transparent; }

/* Interactive NCA Simulator in Inspector */
.sim-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 12px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 9px;
}

.sim-box canvas {
  image-rendering: pixelated;
  border-radius: 5px;
  border: 1px solid var(--border);
  display: block;
}

.sim-controls {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  flex-wrap: wrap;
}

.sim-btn-group { display: inline-flex; align-items: center; gap: 5px; }

/* Truth table visual grid for von Neumann / 1D neighborhoods */
.truth-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
}

.truth-cell {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  padding: 6px 8px;
  border-radius: 7px;
  border: 1px solid var(--border-subtle);
  background: var(--surface-2);
  font: 10.5px var(--mono);
  color: var(--dim);
}

.truth-cell.on {
  border-color: var(--border);
  background: var(--surface);
  color: var(--ink);
  font-weight: 600;
}

.vn-cross {
  display: grid;
  grid-template-columns: repeat(3, 6px);
  grid-template-rows: repeat(3, 6px);
  gap: 1px;
  flex: none;
}

.vn-px { width: 6px; height: 6px; border-radius: 1px; }
.vn-px.off { background: var(--bar-bg); }
.vn-px.on { background: var(--ink); }
.vn-px.ctr { background: transparent; border: 1px solid var(--dim); }

/* Modal */
.modal {
  position: fixed;
  top: 50%;
  left: 50%;
  z-index: 70;
  transform: translate(-50%, -48%) scale(0.98);
  width: min(620px, 94vw);
  max-height: 84vh;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow-lg);
  display: flex;
  flex-direction: column;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.16s ease, transform 0.16s ease;
}
.modal.open { opacity: 1; pointer-events: auto; transform: translate(-50%, -50%) scale(1); }
.modal-body { overflow-y: auto; padding: 16px 18px 22px; display: flex; flex-direction: column; gap: 18px; }
.export-row { display: flex; flex-wrap: wrap; gap: 8px; }
.cfg-groups { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.cfg-group { border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 12px; background: var(--surface-2); }
.cfg-group h4 { margin: 0 0 7px; font-size: 10.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--dim); }
.cfg-row { display: flex; justify-content: space-between; gap: 10px; font: 11.5px/1.6 var(--mono); }
.cfg-row .ck { color: var(--dim); }
.cfg-row .cv { color: var(--ink); font-weight: 500; text-align: right; word-break: break-all; }
.macro-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }

/* Responsive */
@media (max-width: 1140px) {
  .families-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .workspace-grid { grid-template-columns: 1fr; }
}
@media (max-width: 880px) {
  .topbar-inner { padding: 0 14px; }
  .run-spec, .sep { display: none; }
  .main { padding: 14px 14px 28px; }
  .kpi-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi:nth-child(2) { border-right: 0; }
  .kpi:nth-child(1), .kpi:nth-child(2) { border-bottom: 1px solid var(--border-subtle); }
  .charts-grid { grid-template-columns: 1fr; }
  .families-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cfg-groups, .macro-grid, .truth-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 540px) {
  .families-grid { grid-template-columns: 1fr; }
  .progress-wrap { display: none; }
}
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span id="status-dot" class="status-dot wait"></span>
      <h1 class="title">Self-Play Pretraining</h1>
      <span class="sep"></span>
      <span id="run-spec" class="run-spec">connecting…</span>
      <span id="ablate-pill" class="ablate-pill"></span>
    </div>

    <div class="top-right">
      <div id="view-switch" class="seg" role="group" aria-label="Experiment view" style="display:none">
        <button id="view-programs" class="active" type="button">Programs</button>
        <button id="view-nca" type="button">NCA Rules</button>
      </div>

      <div id="progress-wrap" class="progress-wrap" style="display:none">
        <span id="progress-text">Round 0</span>
        <div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
      </div>

      <div class="seg" role="group" aria-label="Loss unit">
        <button id="unit-bits" class="active" type="button">bits/B</button>
        <button id="unit-nats" type="button">nats</button>
      </div>

      <button id="btn-config" class="btn" type="button">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
          <path d="M2.5 4h11M2.5 8h11M2.5 12h11"/><circle cx="5.5" cy="4" r="1.5" fill="var(--surface)"/><circle cx="10.5" cy="8" r="1.5" fill="var(--surface)"/><circle cx="6.5" cy="12" r="1.5" fill="var(--surface)"/>
        </svg>
        Config &amp; Export
      </button>

      <button id="btn-theme" class="btn icon-only" type="button" title="Toggle light / dark theme" aria-label="Toggle theme"></button>
    </div>
  </div>
</header>

<main class="main">
  <div id="banner" class="banner show">
    <span class="status-dot wait"></span>
    <span id="banner-text">Waiting for live.json…</span>
  </div>

  <section class="kpi-strip" aria-label="Run summary">
    <div class="kpi">
      <div class="kpi-label" id="kpi-1-label">Families Discovered</div>
      <div class="kpi-value"><span id="kpi-1-val">0</span><span id="kpi-1-unit" class="kpi-unit">/ 5</span></div>
      <div class="kpi-sub"><span id="kpi-1-dots" class="mini-dots"></span><span id="kpi-1-sub">awaiting discovery</span></div>
    </div>
    <div class="kpi">
      <div class="kpi-label" id="kpi-2-label">Learner Loss</div>
      <div class="kpi-value"><span id="kpi-2-val">—</span><span id="kpi-2-unit" class="kpi-unit">bits/B</span></div>
      <div class="kpi-sub" id="kpi-2-sub">next-token cross-entropy</div>
    </div>
    <div class="kpi">
      <div class="kpi-label" id="kpi-3-label">Programs Sampled</div>
      <div class="kpi-value"><span id="kpi-3-val">0</span><span id="kpi-3-unit" class="kpi-unit"></span></div>
      <div class="kpi-sub" id="kpi-3-sub">—</div>
    </div>
    <div class="kpi">
      <div class="kpi-label" id="kpi-4-label">QD Archive</div>
      <div class="kpi-value"><span id="kpi-4-val">0</span><span id="kpi-4-unit" class="kpi-unit">elites</span></div>
      <div class="kpi-sub" id="kpi-4-sub">—</div>
    </div>
  </section>

  <section class="charts-grid">
    <div class="card">
      <div class="card-head">
        <h2 class="card-title" id="primary-title">Learner Loss</h2>
        <span class="card-sub" id="primary-sub">bits / byte</span>
        <div class="card-tools">
          <div id="primary-legend" class="legend"></div>
          <div id="xscale-toggle" class="seg" role="group" aria-label="X-axis scale">
            <button id="xscale-lin" class="active" type="button">Lin</button>
            <button id="xscale-log" type="button">Log</button>
          </div>
        </div>
      </div>
      <div class="chart-body">
        <canvas id="primary-canvas"></canvas>
        <div id="primary-tooltip" class="chart-tooltip"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 class="card-title" id="secondary-title">Discovery vs. Uniform Prior</h2>
        <span class="card-sub" id="secondary-sub">Table 1 · log rounds</span>
        <div class="card-tools">
          <div id="secondary-legend" class="legend"></div>
        </div>
      </div>
      <div class="chart-body">
        <svg id="secondary-svg"></svg>
        <div id="secondary-tooltip" class="chart-tooltip"></div>
      </div>
    </div>
  </section>

  <div class="section-bar">
    <span class="eyebrow" id="strip-eyebrow">Appendix C · Mathematical Sequence Families</span>
    <div class="section-right">
      <span class="section-note" id="strip-note">mod 256 recurrences · ≤30 leading bytes · period ≥ 30 · click card to inspect</span>
      <button id="btn-play-cards" class="btn sm" type="button" style="display:none">Pause</button>
    </div>
  </div>
  <section id="families-grid" class="families-grid"></section>

  <section class="workspace-grid">
    <div class="card">
      <div class="card-head">
        <h2 class="card-title" id="archive-title">Archive Elites</h2>
        <span id="archive-count" class="card-sub">sorted by learning reward</span>
        <div class="card-tools">
          <div class="search-wrap">
            <input id="filter" type="text" placeholder="Filter…">
            <button id="filter-clear" class="search-clear" type="button" aria-label="Clear filter">&times;</button>
          </div>
        </div>
      </div>
      <div id="family-pills" class="filter-bar"></div>
      <div class="table-scroll">
        <table>
          <thead id="archive-head"></thead>
          <tbody id="archive-rows"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Latest Proposals</h2>
        <span class="card-sub" id="recent-sub">fresh generator stream</span>
      </div>
      <div id="recent-feed" class="feed"></div>
    </div>
  </section>
</main>

<div id="backdrop" class="backdrop"></div>

<aside id="inspector" class="drawer" aria-hidden="true">
  <div class="drawer-head">
    <span id="insp-dot" class="dot"></span>
    <div>
      <div id="insp-title" class="drawer-title">Inspector</div>
      <div id="insp-sub" class="drawer-sub"></div>
    </div>
    <button id="insp-close" class="drawer-close" type="button" aria-label="Close inspector">&times;</button>
  </div>
  <div id="insp-body" class="drawer-body"></div>
</aside>

<div id="config-modal" class="modal" role="dialog" aria-modal="true" aria-hidden="true">
  <div class="drawer-head">
    <div>
      <div class="drawer-title">Experiment Config &amp; Export</div>
      <div id="modal-sub" class="drawer-sub"></div>
    </div>
    <button id="modal-close" class="drawer-close" type="button" aria-label="Close config">&times;</button>
  </div>
  <div id="modal-body" class="modal-body"></div>
</div>

<script>
const FAMILIES = ['arithmetic', 'quadratic', 'cubic', 'fibonacci', 'geometric'];
const SHORT_FAM = { arithmetic: 'arith', quadratic: 'quad', cubic: 'cubic', fibonacci: 'fib', geometric: 'geom' };
const FORMULAS = {
  arithmetic: 'Δ¹v ≡ c (mod 256)',
  quadratic:  'Δ²v ≡ c (mod 256)',
  cubic:      'Δ³v ≡ c (mod 256)',
  fibonacci:  'vₙ ≡ vₙ₋₁ + vₙ₋₂',
  geometric:  'vₙ ≡ r · vₙ₋₁',
};

const CA_CLASSES = ['complex', 'periodic', 'chaotic', 'frozen'];
const CA_CLASS_LABELS = {
  complex:  'IV · Complex',
  periodic: 'II · Periodic',
  chaotic:  'III · Chaotic',
  frozen:   'I · Frozen',
};
const CA_CLASS_SHORT = { complex: 'IV', periodic: 'II', chaotic: 'III', frozen: 'I' };

const BF_MACROS = {
  Z: ['[-]', 'zero cell'],
  R: ['[->+<]', 'move right'],
  L: ['[->+++<]', 'triple & move right'],
  N: ['[-<->]', 'subtract left'],
  C: ['[->+>+<<]', 'copy-add right'],
  G: ['[>]', 'scan right to zero'],
  H: ['[<]', 'scan left to zero'],
  W: ['[[-]>+<]', 'booleanize right'],
  V: ['[.>]', 'emit right to zero'],
  X: ['[-]++++++++++++++++', 'set cell to 16'],
};

const FORTH_CTRL = new Set(['IF', 'ELSE', 'THEN', 'BEGIN', 'UNTIL']);
const FORTH_IO = new Set(['EMIT', 'KEY']);
const FORTH_OPS = new Set(['DUP', 'DROP', 'SWAP', 'OVER', 'ROT', 'NEGATE', '0=', '+', '-', '*']);

const $ = id => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

const fmtInt = n => (n == null || Number.isNaN(n)) ? '—' : Math.round(Number(n)).toLocaleString('en-US');
const fmtBytes = arr => (arr && arr.length) ? arr.join(' ') : '·';
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const famColor = fam => cssVar('--f-' + (fam || 'none')) || '#9ca3af';
const clsColor = cls => cssVar('--c-' + (cls || 'frozen')) || '#94a3b8';

let state = null;
let lastKey = null;
let activeView = new URLSearchParams(location.search).get('view') || 'live';
let lossUnit = localStorage.getItem('sp_unit') || 'bits';
let lossXScale = localStorage.getItem('sp_xscale') || 'lin';
let activeFilterPill = 'all';
let sortCol = 'reward';
let sortDir = -1;
let hoverIdx = null;
let primaryGeom = null;

// Live NCA animation timers
let cardAnimations = [];
let cardsPlaying = true;
let cardTimer = null;
let inspSimTimer = null;

function isNCA(s = state) {
  return !!(s && (s.experiment === 'nca' || s.nca));
}

function convertLoss(nats) {
  return nats == null ? null : (lossUnit === 'bits' ? nats / Math.LN2 : nats);
}

function fmtSpeedup(expected, obsRound, zeroBaselineHits) {
  if (expected == null || obsRound == null) return null;
  const ratio = expected / Math.max(1, obsRound + 1);
  const prefix = zeroBaselineHits ? '>' : '';
  if (ratio >= 100) return `${prefix}${fmtInt(ratio)}×`;
  if (ratio >= 10) return `${prefix}${ratio.toFixed(0)}×`;
  if (ratio >= 1) return `${prefix}${ratio.toFixed(1)}×`;
  return `${prefix}${ratio.toFixed(2)}×`;
}

function fmtDuration(mins) {
  if (mins == null || !Number.isFinite(mins)) return '—';
  if (mins < 1) return `${Math.round(mins * 60)}s`;
  if (mins < 60) return `${mins.toFixed(1)}m`;
  return `${Math.floor(mins / 60)}h ${Math.round(mins % 60)}m`;
}

function executorKind() {
  return (state && state.config && state.config.executor) || 'brainfuck';
}

/* ================= Exact `nca.py` Reversible Checkerboard & ECA Simulator in JS ================= */
function seededRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function make2DGrid(size, seedKind = 'droplet') {
  const g = Array.from({ length: size }, () => new Uint8Array(size));
  if (seedKind === 'droplet') {
    const rng = seededRng(7);
    const lo = Math.floor(size / 2) - 2;
    const hi = lo + 4;
    for (let r = lo; r < hi; r++) {
      for (let c = lo; c < hi; c++) {
        g[r][c] = rng() < 0.5 ? 1 : 0;
      }
    }
    // ensure non-empty
    g[lo][lo] = 1; g[lo + 1][lo + 2] = 1; g[lo + 2][lo + 1] = 1;
  } else if (seedKind === 'cross') {
    const m = Math.floor(size / 2);
    for (let d = -2; d <= 2; d++) { g[m][(m + d + size) % size] = 1; g[(m + d + size) % size][m] = 1; }
  } else {
    const rng = seededRng(13);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) g[r][c] = rng() < 0.25 ? 1 : 0;
  }
  return g;
}

// Exact bit-for-bit `self_play_pretrain_zero_data.executors.nca.half_step` (modulus = 2)
function nca2DHalfStep(grid, rule, parity) {
  const H = grid.length, W = grid[0].length;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      if ((r + c) % 2 !== parity) continue;
      const n = grid[(r - 1 + H) % H][c];
      const s = grid[(r + 1) % H][c];
      const w = grid[r][(c - 1 + W) % W];
      const e = grid[r][(c + 1) % W];
      const nb = (n << 3) | (s << 2) | (w << 1) | e;
      grid[r][c] ^= (rule >> nb) & 1;
    }
  }
}

function nca2DFullStep(grid, rule, reverse = false) {
  if (reverse) {
    nca2DHalfStep(grid, rule, 1);
    nca2DHalfStep(grid, rule, 0);
  } else {
    nca2DHalfStep(grid, rule, 0);
    nca2DHalfStep(grid, rule, 1);
  }
}

function nca1DRollout(rule, mode, width, steps, seedKind = 'droplet') {
  const rows = [];
  if (mode === 'eca') {
    let row = new Uint8Array(width);
    if (seedKind === 'droplet') {
      row[Math.floor(width / 2)] = 1;
      row[Math.floor(width / 2) - 1] = 1;
      row[Math.floor(width / 2) + 2] = 1;
    } else {
      const rng = seededRng(3);
      for (let i = 0; i < width; i++) row[i] = rng() < 0.3 ? 1 : 0;
    }
    rows.push(row.slice());
    for (let t = 0; t < steps; t++) {
      const nxt = new Uint8Array(width);
      for (let i = 0; i < width; i++) {
        const nb = (row[(i - 1 + width) % width] << 2) | (row[i] << 1) | row[(i + 1) % width];
        nxt[i] = (rule >> nb) & 1;
      }
      row = nxt;
      rows.push(row.slice());
    }
  } else {
    // exact nca.py height=2 reversible strip
    const g = [new Uint8Array(width), new Uint8Array(width)];
    const rng = seededRng(seedKind === 'droplet' ? 7 : 3);
    if (seedKind === 'droplet') {
      const m = Math.floor(width / 2);
      for (let d = -3; d <= 3; d++) {
        g[0][(m + d + width) % width] = rng() < 0.5 ? 1 : 0;
        g[1][(m + d + width) % width] = rng() < 0.5 ? 1 : 0;
      }
      g[0][m] = 1;
    } else {
      for (let i = 0; i < width; i++) { g[0][i] = rng() < 0.25 ? 1 : 0; g[1][i] = rng() < 0.25 ? 1 : 0; }
    }
    rows.push(g[0].slice());
    for (let t = 0; t < steps; t++) {
      for (const parity of [0, 1]) {
        for (let r = 0; r < 2; r++) {
          for (let c = 0; c < width; c++) {
            if ((r + c) % 2 !== parity) continue;
            const v = g[1 - r][c];
            const w = g[r][(c - 1 + width) % width];
            const e = g[r][(c + 1) % width];
            const nb = (v << 2) | (w << 1) | e;
            g[r][c] ^= (rule >> nb) & 1;
          }
        }
      }
      rows.push(g[0].slice());
    }
  }
  return rows;
}

function draw2DGrid(canvas, grid, onColor) {
  const ctx = canvas.getContext('2d');
  const H = grid.length, W = grid[0].length;
  const cw = canvas.width / W, ch = canvas.height / H;
  const off = cssVar('--cell-off') || '#efeee9';
  const gap = cw >= 6 ? 1 : 0;
  ctx.fillStyle = cssVar('--surface-2');
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      ctx.fillStyle = grid[r][c] ? onColor : off;
      ctx.fillRect(c * cw, r * ch, cw - gap, ch - gap);
    }
  }
}

function draw1DSpacetime(canvas, rows, onColor) {
  const ctx = canvas.getContext('2d');
  const T = rows.length, W = rows[0].length;
  const cw = canvas.width / W, ch = canvas.height / T;
  const off = cssVar('--cell-off') || '#efeee9';
  ctx.fillStyle = off;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = onColor;
  for (let t = 0; t < T; t++) {
    for (let x = 0; x < W; x++) {
      if (rows[t][x]) ctx.fillRect(x * cw, t * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}

function makeNCAThumb(rule, mode, color, wPx = 28, hPx = 28) {
  const canvas = el('canvas', 'thumb-canvas');
  canvas.width = wPx;
  canvas.height = hPx;
  canvas.style.width = `${wPx}px`;
  canvas.style.height = `${hPx}px`;
  if (mode === 'nca2d') {
    const g = make2DGrid(12, 'droplet');
    for (let i = 0; i < 6; i++) nca2DFullStep(g, rule, false);
    draw2DGrid(canvas, g, color);
  } else {
    const rows = nca1DRollout(rule, mode, 24, 14, 'random');
    draw1DSpacetime(canvas, rows, color);
  }
  return canvas;
}

/* Synthesize an authentic executable `nca.py` demo-frame program demonstrating a rule! */
function synthesizeNcaPyProgram(rule, mode) {
  if (mode === 'eca') return `ECA Rule ${rule} (${rule.toString(2).padStart(8, '0')})`;
  // Build minimal 4x4 checkerboard demo pair demonstrating the rule under nca.py infer_rule
  const size = mode === 'nca2d' ? 6 : 8;
  const H = mode === 'nca2d' ? 6 : 2;
  const g0 = Array.from({ length: H }, (_, r) => Array.from({ length: size }, (_, c) => ((r * 3 + c * 5 + r * c) % 3 === 0 ? 1 : 0)));
  const g1 = g0.map(row => row.slice());
  if (mode === 'nca2d') {
    nca2DHalfStep(g1, rule, 0);
  } else {
    for (let r = 0; r < 2; r++) for (let c = 0; c < size; c++) if ((r + c) % 2 === 0) {
      const nb = (g1[1 - r][c] << 2) | (g1[r][(c - 1 + size) % size] << 1) | g1[r][(c + 1) % size];
      g1[r][c] ^= (rule >> nb) & 1;
    }
  }
  const ser = g => g.map(row => row.join('')).join('/');
  return `+8;${ser(g0)};${ser(g1)};${ser(g0)}!`;
}

/* ================= Safe Syntax Highlighting & Sparklines (Brainfuck / Forth) ================= */
function renderCode(program, exec = executorKind()) {
  const frag = document.createDocumentFragment();
  if (!program) { frag.append(document.createTextNode('—')); return frag; }
  if (exec === 'forth') {
    const parts = program.split(/(\s+)/);
    let halted = false;
    for (const part of parts) {
      if (!part) continue;
      if (/^\s+$/.test(part)) { frag.append(document.createTextNode(part)); continue; }
      const up = part.toUpperCase();
      if (halted || up === 'BYE') {
        halted = true;
        const sp = el('span', 'tk-dead', part);
        sp.title = 'Halted at BYE (unexecuted)';
        frag.append(sp);
      } else if (FORTH_CTRL.has(up)) frag.append(el('span', 'tk-loop', part));
      else if (FORTH_IO.has(up)) frag.append(el('span', 'tk-io', part));
      else if (FORTH_OPS.has(up)) frag.append(el('span', 'tk-macro', part));
      else if (/^[+-]?\d+$/.test(part)) frag.append(el('span', 'tk-num', part));
      else frag.append(document.createTextNode(part));
    }
    return frag;
  }

  const haltIdx = program.indexOf('F');
  const active = haltIdx === -1 ? program : program.slice(0, haltIdx);
  const dead = haltIdx === -1 ? '' : program.slice(haltIdx);
  let buf = '', bufCls = '';
  const flush = () => {
    if (!buf) return;
    frag.append(bufCls ? el('span', bufCls, buf) : document.createTextNode(buf));
    buf = ''; bufCls = '';
  };
  for (const ch of active) {
    if (BF_MACROS[ch]) {
      flush();
      const sp = el('span', 'tk-macro', ch);
      sp.title = `${ch} → ${BF_MACROS[ch][0]} (${BF_MACROS[ch][1]})`;
      frag.append(sp);
    } else {
      const cls = (ch === '[' || ch === ']') ? 'tk-loop' : (ch === '.' || ch === ',') ? 'tk-io' : '';
      if (cls !== bufCls) flush();
      bufCls = cls;
      buf += ch;
    }
  }
  flush();
  if (dead) {
    const sp = el('span', 'tk-dead', dead);
    sp.title = 'Halted at F (unexecuted)';
    frag.append(sp);
  }
  return frag;
}

function makeSparkline(bytes, color, width, height, detailed = false) {
  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: detailed ? '100%' : width,
    height,
    class: detailed ? '' : 'spark-mini',
    preserveAspectRatio: detailed ? 'none' : 'xMidYMid meet',
  });
  if (!bytes || !bytes.length) {
    svg.append(svgEl('line', {
      x1: 2, y1: height / 2, x2: width - 2, y2: height / 2,
      stroke: cssVar('--border'), 'stroke-width': 1, 'stroke-dasharray': '2 3'
    }));
    return svg;
  }
  const padX = detailed ? 3 : 2, padY = detailed ? 4 : 2.5;
  const w = width - padX * 2, h = height - padY * 2, n = bytes.length;
  const pts = bytes.map((b, i) => [n === 1 ? padX + w / 2 : padX + (w * i) / (n - 1), padY + h * (1 - b / 255)]);
  if (detailed && pts.length > 1) {
    const d = `M ${pts[0][0].toFixed(1)} ${(height - padY).toFixed(1)} ` +
      pts.map(p => `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ') +
      ` L ${pts[pts.length - 1][0].toFixed(1)} ${(height - padY).toFixed(1)} Z`;
    svg.append(svgEl('path', { d, fill: color, 'fill-opacity': '0.10' }));
  }
  if (pts.length > 1) {
    svg.append(svgEl('polyline', {
      points: pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': detailed ? 1.5 : 1.35,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke',
    }));
  }
  if (detailed) {
    for (const [x, y] of pts) svg.append(svgEl('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: 1.7, fill: color }));
  } else if (pts.length === 1) {
    svg.append(svgEl('circle', { cx: pts[0][0].toFixed(1), cy: pts[0][1].toFixed(1), r: 2, fill: color }));
  }
  return svg;
}

/* ================= Header & KPI Rendering (Dual Mode: Programs | NCA) ================= */
function renderSummary(s) {
  const c = s.config;
  const finished = s.status === 'finished';
  $('status-dot').className = 'status-dot ' + (finished ? '' : 'run');

  const paramsM = ((s.num_parameters || 0) / 1e6).toFixed(2);
  const flags = [];
  if (c.ablate_generator && c.ablate_generator !== 'none') flags.push(`generator=${c.ablate_generator}`);
  if (c.ablate_reward && c.ablate_reward !== 'none') flags.push(`reward=${c.ablate_reward}`);
  if (c.ablate_proposals && c.ablate_proposals !== 'none') flags.push(`proposals=${c.ablate_proposals}`);

  if (isNCA(s)) {
    const mode = s.nca.mode || c.mode || 'nca2d';
    const disTag = c.use_disagreement ? `disagreement ×${c.disagreement_weight}` : 'jvp-only';
    const jvpTag = c.jvp_reward_mode && c.jvp_reward_mode !== 'abs' ? ` · jvp ${c.jvp_reward_mode}` : '';
    $('run-spec').textContent = `${mode} · ${fmtInt(s.nca.rule_space)} rules · ${c.dim}d × ${c.depth}L · ${paramsM}M · ${disTag}${jvpTag} · seed ${c.seed}`;
    if (c.use_disagreement === false) flags.push('disagreement=off');
  } else {
    const exec = c.executor || 'brainfuck';
    $('run-spec').textContent = `${exec} · ${c.dim}d × ${c.depth}L · ${paramsM}M · ${c.reference} · seed ${c.seed}`;
  }

  const abPill = $('ablate-pill');
  if (flags.length) {
    abPill.textContent = `ablate · ${flags.join(' · ')}`;
    abPill.classList.add('show');
  } else {
    abPill.classList.remove('show');
  }

  const totalRounds = c.rounds || 0;
  $('progress-wrap').style.display = 'flex';
  if (totalRounds > 0) {
    const pct = Math.min(100, (s.round / totalRounds) * 100);
    $('progress-text').textContent = `${finished ? 'Done · ' : ''}${fmtInt(s.round)} / ${fmtInt(totalRounds)}`;
    $('progress-fill').style.width = `${pct.toFixed(1)}%`;
  } else {
    $('progress-text').textContent = `Round ${fmtInt(s.round)}`;
    $('progress-fill').style.width = finished ? '100%' : '0%';
  }

  const banner = $('banner');
  if (s.round === 0 && !finished) {
    banner.classList.add('show');
    $('banner-text').textContent = 'Round 0 · computing uniform prior baseline or awaiting first self-play round…';
  } else {
    banner.classList.remove('show');
  }

  const losses = s.losses || [];
  const lastNats = losses.length ? losses[losses.length - 1] : null;
  const firstNats = losses.length ? losses[0] : null;
  const lastVal = convertLoss(lastNats);

  if (isNCA(s)) {
    const hist = s.nca.history || [];
    const base = s.nca.uniform_baseline || { mean_entropy: 0, mean_dmg: 0, class_rates: {} };
    const latest = hist.length ? hist[hist.length - 1] : null;

    // KPI 1: Class IV Complex Share
    $('kpi-1-label').textContent = 'Class IV · Complex';
    const c4Now = latest ? latest.class_rates.complex * 100 : 0;
    const c4Unif = (base.class_rates.complex || 0) * 100;
    const c3Diff = latest ? ((latest.class_rates.chaotic - (base.class_rates.chaotic || 0)) * 100) : 0;
    $('kpi-1-val').textContent = latest ? `${c4Now.toFixed(1)}%` : '—';
    $('kpi-1-unit').textContent = '';
    $('kpi-1-dots').replaceChildren();
    const k1Sub = $('kpi-1-sub');
    k1Sub.replaceChildren();
    if (latest) {
      const sp = el('span', 'kpi-delta' + (c4Now > c4Unif * 1.15 ? ' good' : ''), `${(c4Now / Math.max(c4Unif, 0.1)).toFixed(1)}× unif`);
      const chaosDelta = c3Diff < -1 ? ` · chaos ${c3Diff.toFixed(0)}pp` : ` · chaos ${(latest.class_rates.chaotic * 100).toFixed(0)}%`;
      k1Sub.append(sp, document.createTextNode(` (${c4Unif.toFixed(1)}%)${chaosDelta}`));
    } else {
      k1Sub.textContent = `uniform ${c4Unif.toFixed(1)}% complex`;
    }

    // KPI 2: Learner Loss
    $('kpi-2-label').textContent = 'Learner Loss';
    $('kpi-2-val').textContent = lastVal == null ? '—' : lastVal.toFixed(3);
    $('kpi-2-unit').textContent = lossUnit === 'bits' ? 'bits/tok' : 'nats';
    const k2Sub = $('kpi-2-sub');
    k2Sub.replaceChildren();
    if (lastNats != null) {
      if (losses.length > 1) {
        const delta = convertLoss(lastNats) - convertLoss(firstNats);
        const sp = el('span', 'kpi-delta' + (delta < 0 ? ' good' : ''), `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)} vs r.1`);
        k2Sub.append(sp, document.createTextNode(' · '));
      }
      const alt = lossUnit === 'bits' ? `${lastNats.toFixed(3)} nats` : `${(lastNats / Math.LN2).toFixed(3)} bits`;
      k2Sub.append(document.createTextNode(alt));
    } else {
      k2Sub.textContent = 'spacetime next-token CE';
    }

    // KPI 3: Rules Evaluated
    $('kpi-3-label').textContent = 'Rules Sampled';
    const checked = s.samples_checked != null ? s.samples_checked : (s.round * (c.detect_samples || 0));
    $('kpi-3-val').textContent = fmtInt(checked);
    $('kpi-3-unit').textContent = '';
    const entStr = latest && latest.mean_entropy != null ? `H ${latest.mean_entropy.toFixed(2)}` : `${fmtInt(c.detect_samples)}/rd`;
    $('kpi-3-sub').textContent = `${entStr} · ${fmtDuration(s.elapsed_min)} ${finished ? 'total' : 'elapsed'}`;

    // KPI 4: Frontier Archive
    $('kpi-4-label').textContent = 'QD Archive';
    $('kpi-4-val').textContent = fmtInt(s.archive_size);
    $('kpi-4-unit').textContent = 'elites';
    const rows = s.archive_rows || [];
    const peakRew = rows.length ? Math.max(...rows.map(r => r.reward || 0)) : 0;
    const lamStr = latest ? `λ ${latest.lam.toFixed(2)} · vac0 ${Math.round(100 * latest.vac0)}%` : 'reversible checkerboard';
    $('kpi-4-sub').textContent = `peak r ${peakRew.toFixed(4)} · ${lamStr}`;
    return;
  }

  // Brainfuck / Forth Table 1 KPIs
  const discovered = FAMILIES.filter(f => s.discoveries && s.discoveries[f]);
  $('kpi-1-label').textContent = 'Families Discovered';
  $('kpi-1-val').textContent = String(discovered.length);
  $('kpi-1-unit').textContent = '/ 5';
  const dotsWrap = $('kpi-1-dots');
  dotsWrap.replaceChildren();
  for (const f of FAMILIES) {
    const d = el('span', 'mini-dot');
    if (s.discoveries && s.discoveries[f]) d.style.background = famColor(f);
    dotsWrap.append(d);
  }
  $('kpi-1-sub').textContent = discovered.length ? discovered.map(f => SHORT_FAM[f]).join(' · ') : 'awaiting first recurrence';

  $('kpi-2-label').textContent = 'Learner Loss';
  $('kpi-2-val').textContent = lastVal == null ? '—' : lastVal.toFixed(3);
  $('kpi-2-unit').textContent = lossUnit === 'bits' ? 'bits/B' : 'nats/B';
  const lossSub = $('kpi-2-sub');
  lossSub.replaceChildren();
  if (lastNats != null) {
    if (losses.length > 1) {
      const delta = convertLoss(lastNats) - convertLoss(firstNats);
      const sp = el('span', 'kpi-delta' + (delta < 0 ? ' good' : ''), `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)} vs r.1`);
      lossSub.append(sp, document.createTextNode(' · '));
    }
    lossSub.append(document.createTextNode(lossUnit === 'bits' ? `${lastNats.toFixed(3)} nats` : `${(lastNats / Math.LN2).toFixed(3)} bits/B`));
  } else {
    lossSub.textContent = 'next-byte cross-entropy';
  }

  $('kpi-3-label').textContent = 'Programs Sampled';
  const checked = s.samples_checked != null ? s.samples_checked : (s.round * (c.detect_samples || 0));
  $('kpi-3-val').textContent = fmtInt(checked);
  $('kpi-3-unit').textContent = '';
  let timePart = `${fmtDuration(s.elapsed_min)} elapsed`;
  if (!finished && s.round > 0 && totalRounds > s.round && s.elapsed_min > 0) {
    timePart += ` · ~${fmtDuration((s.elapsed_min / s.round) * (totalRounds - s.round))} left`;
  } else if (finished) {
    timePart = `finished in ${fmtDuration(s.elapsed_min)}`;
  }
  $('kpi-3-sub').textContent = `${fmtInt(c.detect_samples)}/rd · ${timePart}`;

  $('kpi-4-label').textContent = 'QD Archive';
  $('kpi-4-val').textContent = fmtInt(s.archive_size);
  $('kpi-4-unit').textContent = 'elites';
  const rows = s.archive_rows || [];
  if (rows.length) {
    const peak = Math.max(...rows.map(r => r.reward));
    const structured = rows.filter(r => r.family).length;
    $('kpi-4-sub').textContent = `peak r ${peak.toFixed(2)} · ${structured}/${rows.length} structured`;
  } else {
    $('kpi-4-sub').textContent = 'quality-diversity niches';
  }
}

/* ================= Charts Helpers ================= */
function niceTicks(lo, hi, maxTicks = 4) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const rawStep = span / maxTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm <= 1.5 ? 1 : norm <= 3 ? 2 : norm <= 7 ? 5 : 10) * mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) ticks.push(Number(v.toFixed(6)));
  return ticks;
}

function computeEMA(arr) {
  if (!arr.length) return [];
  const alpha = Math.min(0.28, Math.max(0.05, 10 / arr.length));
  const out = [arr[0]];
  for (let i = 1; i < arr.length; i++) out.push(alpha * arr[i] + (1 - alpha) * out[i - 1]);
  return out;
}

/* Primary Canvas Chart: Learner Loss (EMA + Raw) */
function drawPrimaryChart(s) {
  const ncaMode = isNCA(s);
  $('xscale-toggle').style.display = 'inline-flex';
  const leg = $('primary-legend');
  leg.replaceChildren();
  $('primary-title').textContent = 'Learner Loss';
  $('primary-sub').textContent = lossUnit === 'bits' ? (ncaMode ? 'bits / token' : 'bits / byte') : (ncaMode ? 'nats / token' : 'nats / byte');
  const l1 = el('span', 'legend-item'); l1.append(el('span', 'swatch-line'), document.createTextNode('EMA'));
  const l2 = el('span', 'legend-item'); l2.append(el('span', 'swatch-line raw'), document.createTextNode('Raw'));
  leg.append(l1, l2);
  drawLossCanvas(s);
}

function drawLossCanvas(s) {
  const canvas = $('primary-canvas');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const rawNats = (s && s.losses) || [];
  if (!rawNats.length) {
    ctx.fillStyle = cssVar('--dim');
    ctx.font = '12px ' + cssVar('--mono');
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting learner loss telemetry…', width / 2, height / 2);
    primaryGeom = null;
    return;
  }

  const values = rawNats.map(convertLoss);
  const ema = computeEMA(values);
  const n = values.length;
  const pad = { l: 42, r: 14, t: 28, b: 24 };
  const plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;

  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1e-4) { lo -= 0.25; hi += 0.25; }
  const margin = (hi - lo) * 0.1;
  lo = Math.max(0, lo - margin);
  hi = hi + margin;

  const xOf = i => n === 1 ? pad.l + plotW / 2 : (lossXScale === 'log' ? pad.l + plotW * (Math.log10(i + 1) / Math.log10(n)) : pad.l + (plotW * i) / (n - 1));
  const yOf = v => pad.t + plotH * (1 - (v - lo) / (hi - lo));

  primaryGeom = { kind: 'loss', pad, plotW, plotH, n, values, ema, rawNats, xOf, yOf };

  const yTicks = niceTicks(lo, hi, 4);
  ctx.font = '10.5px ' + cssVar('--mono');
  ctx.textBaseline = 'middle';
  for (const t of yTicks) {
    const y = Math.round(yOf(t)) + 0.5;
    if (y < pad.t - 2 || y > height - pad.b + 2) continue;
    ctx.strokeStyle = cssVar('--grid'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke();
    ctx.fillStyle = cssVar('--dim'); ctx.textAlign = 'right';
    ctx.fillText(t.toFixed(t >= 10 ? 1 : 2), pad.l - 7, y);
  }

  const axisY = height - pad.b + 0.5;
  ctx.strokeStyle = cssVar('--border'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, axisY); ctx.lineTo(width - pad.r, axisY); ctx.stroke();
  ctx.fillStyle = cssVar('--dim'); ctx.font = '10px ' + cssVar('--mono');
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  if (n > 1) {
    const xTicks = lossXScale === 'log'
      ? (() => { const out = [1]; for (let p = 10; p <= n; p *= 10) out.push(p); if (out[out.length - 1] < n * 0.72) out.push(n); return out; })()
      : (() => { const st = niceTicks(1, n, 5).filter(v => v >= 1 && v <= n); if (!st.length || st[0] > 1) st.unshift(1); if (st[st.length - 1] < n - n / 12) st.push(n); return st; })();
    for (const r of xTicks) ctx.fillText(r >= 1000 ? `${(r / 1000).toFixed(r % 1000 === 0 ? 0 : 1)}k` : String(Math.round(r)), xOf(r - 1), axisY + 6);
  } else {
    ctx.fillText('1', xOf(0), axisY + 6);
  }

  // Discovery vertical markers (Brainfuck / Forth mode)
  if (!isNCA(s) && s.discoveries) {
    const markers = [];
    for (const fam of FAMILIES) {
      const rec = s.discoveries[fam];
      if (!rec || rec.round >= n) continue;
      markers.push({ fam, round: rec.round, cx: xOf(rec.round), label: SHORT_FAM[fam], color: famColor(fam) });
    }
    markers.sort((a, b) => a.cx - b.cx);
    ctx.font = '600 9.5px ' + cssVar('--mono');
    const laneRight = [-Infinity, -Infinity];
    for (const m of markers) {
      const tw = ctx.measureText(m.label).width + 8;
      let lx = Math.max(pad.l, Math.min(width - pad.r - tw, m.cx - tw / 2));
      let lane = 0;
      if (lx < laneRight[0] + 4) { lane = 1; if (lx < laneRight[1] + 4) lx = Math.min(width - pad.r - tw, laneRight[1] + 4); }
      laneRight[lane] = lx + tw;
      const ly = lane === 0 ? 2 : 14;
      ctx.save();
      ctx.strokeStyle = m.color; ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(m.cx, ly + 11); ctx.lineTo(m.cx, height - pad.b); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = m.color; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(m.label, lx + 4, ly + 1);
    }
  }

  if (n > 1) {
    ctx.strokeStyle = cssVar('--loss-raw'); ctx.lineWidth = 1.1; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < n; i++) i ? ctx.lineTo(xOf(i), yOf(values[i])) : ctx.moveTo(xOf(0), yOf(values[0]));
    ctx.stroke();

    const grad = ctx.createLinearGradient(0, pad.t, 0, height - pad.b);
    grad.addColorStop(0, cssVar('--loss-fill-top')); grad.addColorStop(1, cssVar('--loss-fill-bot'));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(xOf(0), height - pad.b);
    for (let i = 0; i < n; i++) ctx.lineTo(xOf(i), yOf(ema[i]));
    ctx.lineTo(xOf(n - 1), height - pad.b);
    ctx.closePath(); ctx.fill();

    ctx.strokeStyle = cssVar('--loss-ema'); ctx.lineWidth = 1.85; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < n; i++) i ? ctx.lineTo(xOf(i), yOf(ema[i])) : ctx.moveTo(xOf(0), yOf(ema[0]));
    ctx.stroke();
  } else {
    ctx.fillStyle = cssVar('--loss-ema');
    ctx.beginPath(); ctx.arc(xOf(0), yOf(values[0]), 3.5, 0, Math.PI * 2); ctx.fill();
  }

  if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < n) {
    const hx = xOf(hoverIdx), hyEma = yOf(ema[hoverIdx]), hyRaw = yOf(values[hoverIdx]);
    ctx.save();
    ctx.strokeStyle = cssVar('--dim'); ctx.globalAlpha = 0.5; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(hx, pad.t); ctx.lineTo(hx, height - pad.b); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = cssVar('--loss-raw'); ctx.beginPath(); ctx.arc(hx, hyRaw, 2.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cssVar('--surface'); ctx.beginPath(); ctx.arc(hx, hyEma, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cssVar('--loss-ema'); ctx.beginPath(); ctx.arc(hx, hyEma, 3.3, 0, Math.PI * 2); ctx.fill();
  }
}

function handlePrimaryHover(ev) {
  const tt = $('primary-tooltip');
  if (!primaryGeom || !state) { tt.classList.remove('show'); return; }
  const rect = $('primary-canvas').getBoundingClientRect();
  const mx = ev.clientX - rect.left;

  const { pad, plotW, n, values, ema, rawNats, xOf } = primaryGeom;
  if (mx < pad.l - 6 || mx > rect.width - pad.r + 6) { hoverIdx = null; tt.classList.remove('show'); drawPrimaryChart(state); return; }
  let idx = 0;
  if (n > 1) {
    const t = Math.max(0, Math.min(1, (mx - pad.l) / plotW));
    idx = lossXScale === 'log' ? Math.round(Math.pow(10, t * Math.log10(n)) - 1) : Math.round(t * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));
  }
  if (hoverIdx !== idx) { hoverIdx = idx; drawPrimaryChart(state); }

  const u = lossUnit === 'bits' ? 'bits' : 'nats';
  const alt = lossUnit === 'bits' ? `${rawNats[idx].toFixed(3)} nats` : `${(rawNats[idx] / Math.LN2).toFixed(3)} bits`;
  tt.replaceChildren();
  tt.append(el('div', 'tt-head', `Round ${idx + 1}`));
  const r1 = el('div', 'tt-row'); r1.append(el('span', null, 'EMA'), el('span', null, `${ema[idx].toFixed(3)} ${u}`));
  const r2 = el('div', 'tt-row'); r2.append(el('span', null, 'Raw'), el('span', null, `${values[idx].toFixed(3)} ${u} (${alt})`));
  tt.append(r1, r2);
  if (!isNCA(state) && state.discoveries) {
    for (const fam of FAMILIES) {
      const rec = state.discoveries[fam];
      if (rec && Math.abs(rec.round - idx) <= Math.max(1, Math.floor(n / 120))) {
        const d = el('div', 'tt-disc');
        d.append(el('span', 'dot c-' + fam), el('span', null, `${fam} discovered (r.${rec.round})`));
        tt.append(d);
      }
    }
  }
  tt.classList.add('show');
  const px = xOf(idx), ttW = tt.offsetWidth || 180;
  tt.style.left = `${(px + 14 + ttW > rect.width) ? Math.max(8, px - ttW - 14) : px + 14}px`;
  tt.style.top = '12px';
}

/* Secondary SVG Chart: Table 1 Speedup (Programs) OR Dynamical Phase Shift Stacked Stream (NCA) */
function renderSecondaryChart(s) {
  const leg = $('secondary-legend');
  leg.replaceChildren();
  if (isNCA(s)) {
    $('secondary-title').textContent = 'Dynamical Class Shift';
    $('secondary-sub').textContent = 'Frozen (I) · Periodic (II) · Chaotic (III) · Complex (IV)';
    for (const c of ['complex', 'periodic', 'frozen', 'chaotic']) {
      const item = el('span', 'legend-item');
      const sq = el('span', 'swatch-sq');
      sq.style.background = clsColor(c);
      if (c === 'chaotic') sq.style.opacity = '0.45';
      item.append(sq, document.createTextNode(CA_CLASS_SHORT[c]));
      leg.append(item);
    }
    renderNCAPhaseSVG(s);
  } else {
    $('secondary-title').textContent = 'Discovery vs. Uniform Prior';
    const l1 = el('span', 'legend-item'); l1.append(el('span', 'swatch-dot'), document.createTextNode('Self-play'));
    const l2 = el('span', 'legend-item'); l2.append(el('span', 'swatch-diamond'), document.createTextNode('Uniform E[rd]'));
    leg.append(l1, l2);
    renderSpeedupSVG(s);
  }
}

function renderNCAPhaseSVG(s) {
  const svg = $('secondary-svg');
  const tt = $('secondary-tooltip');
  svg.replaceChildren();
  const rect = svg.getBoundingClientRect();
  const W = Math.max(320, Math.round(rect.width) || 460);
  const H = Math.max(180, Math.round(rect.height) || 204);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const hist = (s.nca && s.nca.history) || [];
  const base = (s.nca && s.nca.uniform_baseline && s.nca.uniform_baseline.class_rates) || { complex: 0, periodic: 0, frozen: 0, chaotic: 1 };

  const padL = 36, padR = 14, padT = 16, padB = 26;
  const barW = 24, gap = 16;
  const streamX = padL + barW + gap;
  const streamW = W - streamX - padR;
  const plotH = H - padT - padB;
  const yOf = frac => padT + plotH * (1 - frac);

  // Order from bottom to top: complex (anchored at 0!) -> periodic -> frozen -> chaotic (at top!)
  const order = [
    ['complex', 0.88],
    ['periodic', 0.72],
    ['frozen', 0.45],
    ['chaotic', 0.28],
  ];

  // Y ticks (0%, 25%, 50%, 75%, 100%)
  for (const pct of [0, 0.25, 0.5, 0.75, 1.0]) {
    const y = yOf(pct);
    svg.append(svgEl('line', { x1: padL, y1: y, x2: W - padR, y2: y, stroke: cssVar('--grid'), 'stroke-width': 1 }));
    const t = svgEl('text', { x: padL - 6, y: y + 3, 'text-anchor': 'end', fill: cssVar('--dim'), 'font-size': '9.5', 'font-family': cssVar('--mono') });
    t.textContent = `${Math.round(pct * 100)}%`;
    svg.append(t);
  }

  // Left reference stacked bar: Uniform Prior
  let acc = 0;
  for (const [cls, alpha] of order) {
    const v = base[cls] || 0;
    const yTop = yOf(acc + v), yBot = yOf(acc);
    svg.append(svgEl('rect', {
      x: padL, y: yTop, width: barW, height: Math.max(0, yBot - yTop),
      fill: clsColor(cls), 'fill-opacity': alpha
    }));
    acc += v;
  }
  svg.append(svgEl('rect', { x: padL, y: padT, width: barW, height: plotH, fill: 'none', stroke: cssVar('--border'), 'stroke-width': 1 }));
  const uLbl = svgEl('text', { x: padL + barW / 2, y: H - padB + 14, 'text-anchor': 'middle', fill: cssVar('--dim'), 'font-size': '9.5', 'font-family': cssVar('--mono') });
  uLbl.textContent = 'Unif';
  svg.append(uLbl);

  if (!hist.length) return;

  const minR = hist[0].round, maxR = Math.max(hist[hist.length - 1].round, minR + 1);
  const xOf = r => hist.length === 1 ? streamX + streamW / 2 : streamX + streamW * ((r - minR) / (maxR - minR));

  // Stacked areas across rounds
  const cumLow = hist.map(() => 0);
  for (const [cls, alpha] of order) {
    const cumHigh = hist.map((h, i) => Math.min(1, cumLow[i] + (h.class_rates[cls] || 0)));
    if (hist.length > 1) {
      let d = `M ${xOf(hist[0].round).toFixed(1)} ${yOf(cumLow[0]).toFixed(1)}`;
      for (let i = 0; i < hist.length; i++) d += ` L ${xOf(hist[i].round).toFixed(1)} ${yOf(cumHigh[i]).toFixed(1)}`;
      for (let i = hist.length - 1; i >= 0; i--) d += ` L ${xOf(hist[i].round).toFixed(1)} ${yOf(cumLow[i]).toFixed(1)}`;
      d += ' Z';
      svg.append(svgEl('path', { d, fill: clsColor(cls), 'fill-opacity': alpha }));
      if (cls === 'complex') {
        const linePts = hist.map((h, i) => `${xOf(h.round).toFixed(1)},${yOf(cumHigh[i]).toFixed(1)}`).join(' ');
        svg.append(svgEl('polyline', { points: linePts, fill: 'none', stroke: clsColor('complex'), 'stroke-width': 1.7 }));
      }
    }
    for (let i = 0; i < hist.length; i++) cumLow[i] = cumHigh[i];
  }

  // Uniform Class IV dashed horizontal benchmark across stream
  const yUnifC4 = yOf(base.complex || 0);
  svg.append(svgEl('line', {
    x1: streamX, y1: yUnifC4, x2: W - padR, y2: yUnifC4,
    stroke: cssVar('--surface'), 'stroke-width': 1.3, 'stroke-dasharray': '3 3', opacity: '0.85'
  }));

  // Frame border & X ticks
  svg.append(svgEl('rect', { x: streamX, y: padT, width: streamW, height: plotH, fill: 'none', stroke: cssVar('--border'), 'stroke-width': 1 }));
  for (const r of niceTicks(minR, maxR, 4).filter(v => v >= minR && v <= maxR)) {
    const t = svgEl('text', { x: xOf(r), y: H - padB + 14, 'text-anchor': 'middle', fill: cssVar('--dim'), 'font-size': '9.5', 'font-family': cssVar('--mono') });
    t.textContent = `r.${Math.round(r)}`;
    svg.append(t);
  }

  // Interactive hover slices
  const hoverLine = svgEl('line', { x1: 0, y1: padT, x2: 0, y2: H - padB, stroke: cssVar('--ink'), 'stroke-width': 1, 'stroke-dasharray': '2 2', opacity: '0' });
  svg.append(hoverLine);

  const hit = svgEl('rect', { x: streamX, y: padT, width: streamW, height: plotH, fill: 'transparent', style: 'cursor:crosshair' });
  hit.addEventListener('mousemove', ev => {
    const br = svg.getBoundingClientRect();
    const mx = ((ev.clientX - br.left) / br.width) * W;
    let best = 0, bestD = Infinity;
    hist.forEach((h, i) => { const d = Math.abs(xOf(h.round) - mx); if (d < bestD) { bestD = d; best = i; } });
    const h = hist[best];
    const hx = xOf(h.round);
    hoverLine.setAttribute('x1', hx); hoverLine.setAttribute('x2', hx); hoverLine.setAttribute('opacity', '0.55');
    tt.replaceChildren();
    tt.append(el('div', 'tt-head', `Round ${h.round} · Class Mix`));
    for (const c of CA_CLASSES) {
      const r = el('div', 'tt-row');
      const now = (100 * (h.class_rates[c] || 0)).toFixed(1);
      const u = (100 * (base[c] || 0)).toFixed(1);
      r.append(el('span', null, CA_CLASS_LABELS[c]), el('span', null, `${now}% (unif ${u}%)`));
      tt.append(r);
    }
    tt.classList.add('show');
    const px = (hx / W) * br.width, ttW = tt.offsetWidth || 190;
    tt.style.left = `${(px + 12 + ttW > br.width) ? Math.max(8, px - ttW - 12) : px + 12}px`;
    tt.style.top = '10px';
  });
  hit.addEventListener('mouseleave', () => {
    hoverLine.setAttribute('opacity', '0');
    tt.classList.remove('show');
  });
  svg.append(hit);
}

function renderSpeedupSVG(s) {
  const svg = $('secondary-svg');
  const tt = $('secondary-tooltip');
  svg.replaceChildren();
  const rect = svg.getBoundingClientRect();
  const W = Math.max(320, Math.round(rect.width) || 460);
  const H = Math.max(180, Math.round(rect.height) || 204);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const baseline = (s && s.baseline) || { hits: {}, expected_round: {}, num_samples: 0 };
  const c = (s && s.config) || {};
  $('secondary-sub').textContent = baseline.num_samples
    ? `at ${fmtInt(c.detect_samples || 256)}/rd · ${fmtInt(baseline.num_samples)} uniform baseline`
    : 'log-scale round of first discovery';

  const padL = 82, padR = 72, padT = 18, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const vals = [100, Math.max(1, s ? s.round : 1)];
  for (const f of FAMILIES) {
    const exp = baseline.expected_round && baseline.expected_round[f];
    const rec = s && s.discoveries && s.discoveries[f];
    if (exp != null && exp > 0) vals.push(exp * (baseline.hits[f] === 0 ? 1.8 : 1.25));
    if (rec) vals.push(Math.max(1, rec.round + 1));
  }
  const maxPow = Math.max(2, Math.ceil(Math.log10(Math.max(...vals))));
  const logX = v => padL + plotW * (Math.log10(Math.max(1, v)) / maxPow);

  for (let p = 0; p <= maxPow; p++) {
    const x = padL + (plotW * p) / maxPow;
    svg.append(svgEl('line', { x1: x, y1: padT - 4, x2: x, y2: H - padB, stroke: cssVar('--grid'), 'stroke-width': 1 }));
    const lbl = svgEl('text', { x, y: H - padB + 14, 'text-anchor': 'middle', fill: cssVar('--dim'), 'font-size': '10', 'font-family': cssVar('--mono') });
    if (p === 0) lbl.textContent = '1';
    else if (p === 1) lbl.textContent = '10';
    else { lbl.textContent = '10'; const sup = svgEl('tspan', { dy: '-3.5', 'font-size': '7.5' }); sup.textContent = String(p); lbl.append(sup); }
    svg.append(lbl);
  }
  svg.append(svgEl('line', { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: cssVar('--border'), 'stroke-width': 1 }));

  if (s && s.round >= 1) {
    const xNow = logX(s.round);
    svg.append(svgEl('line', { x1: xNow, y1: padT - 4, x2: xNow, y2: H - padB, stroke: cssVar('--dim'), 'stroke-width': 1, 'stroke-dasharray': '2 3', opacity: '0.65' }));
    const nowTxt = svgEl('text', { x: xNow, y: padT - 8, 'text-anchor': 'middle', fill: cssVar('--dim'), 'font-size': '9.5', 'font-family': cssVar('--mono') });
    nowTxt.textContent = `r.${s.round}`;
    svg.append(nowTxt);
  }

  const rowH = plotH / FAMILIES.length;
  FAMILIES.forEach((fam, idx) => {
    const cy = padT + rowH * (idx + 0.5);
    const rec = s && s.discoveries && s.discoveries[fam];
    const exp = baseline.expected_round && baseline.expected_round[fam];
    const zeroHits = baseline.hits && baseline.hits[fam] === 0;
    const color = famColor(fam);

    const g = svgEl('g', { style: rec ? 'cursor:pointer' : 'cursor:default' });
    const bg = svgEl('rect', { x: 0, y: cy - rowH / 2 + 2, width: W, height: rowH - 4, rx: 5, fill: 'transparent' });
    g.append(bg);
    g.append(svgEl('line', { x1: padL, y1: cy, x2: W - padR, y2: cy, stroke: cssVar('--border-subtle'), 'stroke-width': 1 }));

    const nameTxt = svgEl('text', { x: padL - 10, y: cy + 3.5, 'text-anchor': 'end', fill: rec ? cssVar('--ink') : cssVar('--dim'), 'font-size': '11.5', 'font-weight': rec ? '600' : '400', 'font-family': cssVar('--sans') });
    nameTxt.textContent = fam[0].toUpperCase() + fam.slice(1);
    g.append(nameTxt);

    const xObs = rec ? logX(Math.max(1, rec.round + 1)) : null;
    const xExp = exp != null ? logX(exp) : null;
    if (xObs != null && xExp != null && xExp > xObs) {
      g.append(svgEl('line', { x1: xObs, y1: cy, x2: xExp, y2: cy, stroke: color, 'stroke-width': 3.5, 'stroke-linecap': 'round', opacity: '0.28' }));
    }
    if (xExp != null) {
      if (zeroHits) {
        const ax2 = Math.min(W - padR - 2, xExp + 15);
        g.append(svgEl('line', { x1: xExp + 4, y1: cy, x2: ax2, y2: cy, stroke: cssVar('--prior'), 'stroke-width': 1.25 }));
        g.append(svgEl('polyline', { points: `${ax2 - 3.5},${cy - 3} ${ax2},${cy} ${ax2 - 3.5},${cy + 3}`, fill: 'none', stroke: cssVar('--prior'), 'stroke-width': 1.25, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
      }
      g.append(svgEl('polygon', { points: `${xExp},${cy - 4.2} ${xExp + 4.2},${cy} ${xExp},${cy + 4.2} ${xExp - 4.2},${cy}`, fill: cssVar('--surface'), stroke: cssVar('--prior'), 'stroke-width': 1.5 }));
    }
    if (xObs != null) {
      g.append(svgEl('circle', { cx: xObs, cy, r: 6.2, fill: cssVar('--surface') }));
      g.append(svgEl('circle', { cx: xObs, cy, r: 4.5, fill: color }));
    }

    const speedStr = rec ? (fmtSpeedup(exp, rec.round, zeroHits) || `r.${rec.round}`) : (exp != null ? `${zeroHits ? '>' : '≈'}${fmtInt(exp)}` : '—');
    const rightTxt = svgEl('text', { x: W - 4, y: cy + 3.5, 'text-anchor': 'end', fill: rec ? color : cssVar('--dim'), 'font-size': '11', 'font-weight': rec ? '600' : '400', 'font-family': cssVar('--mono') });
    rightTxt.textContent = speedStr;
    g.append(rightTxt);

    g.addEventListener('mouseenter', () => {
      bg.setAttribute('fill', cssVar('--hover'));
      const hits = (s && s.sample_hits && s.sample_hits[fam]) || 0;
      const bHits = baseline.hits ? baseline.hits[fam] : null;
      tt.replaceChildren();
      tt.append(el('div', 'tt-head', fam[0].toUpperCase() + fam.slice(1) + ' · ' + FORMULAS[fam]));
      const r1 = el('div', 'tt-row'); r1.append(el('span', null, 'Self-play'), el('span', null, rec ? `round ${rec.round} (${hits} hits)` : 'not yet discovered')); tt.append(r1);
      if (exp != null) { const r2 = el('div', 'tt-row'); r2.append(el('span', null, 'Uniform prior'), el('span', null, `${zeroHits ? '>' : '≈'}${fmtInt(exp)} rds (${fmtInt(bHits)}/${fmtInt(baseline.num_samples)})`)); tt.append(r2); }
      if (rec && exp != null) { const r3 = el('div', 'tt-row'); r3.append(el('span', null, 'Speedup'), el('span', null, fmtSpeedup(exp, rec.round, zeroHits))); tt.append(r3); }
      tt.classList.add('show');
      tt.style.left = `${padL + 6}px`;
      tt.style.top = `${Math.max(4, cy > H / 2 ? cy - 78 : cy + 16)}px`;
    });
    g.addEventListener('mouseleave', () => { bg.setAttribute('fill', 'transparent'); tt.classList.remove('show'); });
    if (rec) {
      g.addEventListener('click', () => openProgramInspector({
        source: `First discovery · round ${rec.round}`, family: fam, program: rec.program,
        output_head: rec.output_head, length: rec.program.length, loops: rec.loops, steps: rec.steps, hits: (s.sample_hits || {})[fam] || 0,
      }));
    }
    svg.append(g);
  });
}

/* ================= 5-Card Strip: Appendix C Families OR Discovered NCA Frontier ================= */
function stopCardTimer() {
  if (cardTimer) { clearInterval(cardTimer); cardTimer = null; }
  cardAnimations = [];
}

function startCardTimer() {
  stopCardTimer();
  if (!cardsPlaying || !state || !isNCA(state)) return;
  const mode = state.nca.mode || 'nca2d';
  if (mode !== 'nca2d') return;
  cardTimer = setInterval(() => {
    if (!cardsPlaying) return;
    for (const anim of cardAnimations) {
      anim.step++;
      if (anim.step > 20) {
        anim.grid = make2DGrid(12, 'droplet');
        anim.step = 0;
      } else {
        nca2DFullStep(anim.grid, anim.rule, false);
      }
      draw2DGrid(anim.canvas, anim.grid, anim.color);
    }
  }, 420);
}

function renderDiscoveryStrip(s) {
  stopCardTimer();
  const grid = $('families-grid');
  grid.replaceChildren();

  if (isNCA(s)) {
    const mode = s.nca.mode || 'nca2d';
    $('strip-eyebrow').textContent = mode === 'nca2d'
      ? 'Discovered NCA Frontier · 2D Reversible Checkerboard Solitons & Waves'
      : (mode === 'nca1d' ? 'Discovered NCA Frontier · 1D-Strip Reversible Checkerboard' : 'Discovered ECA Frontier · Wolfram Cellular Automata');
    $('strip-note').textContent = mode === 'nca2d'
      ? 'nca.py half_step · live droplet evolution · click card to inspect & reverse time'
      : 'spacetime rollouts · click card to inspect';
    $('btn-play-cards').style.display = mode === 'nca2d' ? 'inline-flex' : 'none';
    $('btn-play-cards').textContent = cardsPlaying ? 'Pause' : 'Play';

    const top = (s.nca.top_discovered || []).slice(0, 5);
    const totalSlots = 5;
    for (let i = 0; i < totalSlots; i++) {
      const item = top[i];
      if (!item) {
        const card = el('div', 'fam-card pending');
        const topRow = el('div', 'fam-top');
        topRow.append(el('span', 'dot c-frozen'), el('span', 'fam-name', `Slot ${i + 1}`), el('span', 'speed-pill muted', 'searching'));
        card.append(topRow, el('div', 'fam-meta', 'awaiting structured rule'), el('div', 'nca-stage', '—'));
        grid.append(card);
        continue;
      }

      const color = clsColor(item.cls);
      const card = el('div', 'fam-card found');
      card.style.borderTopColor = color;

      const topRow = el('div', 'fam-top');
      topRow.append(el('span', 'dot c-' + item.cls), el('span', 'fam-name mono', item.hex));
      const pill = el('span', 'speed-pill', `r ${(item.reward ?? 0).toFixed(4)}`);
      pill.style.color = color;
      topRow.append(pill);
      card.append(topRow);

      const nBits = s.nca.rule_bits || 16;
      card.append(el('div', 'fam-meta', `${CA_CLASS_SHORT[item.cls] || item.cls} · hw ${item.hw}/${nBits} · vac ${item.bit0} · H ${(item.entropy ?? 0).toFixed(2)}`));

      // Live CA canvas
      const stage = el('div', 'nca-stage');
      const canvas = el('canvas');
      if (mode === 'nca2d') {
        canvas.width = 96; canvas.height = 96;
        canvas.style.width = '96px'; canvas.style.height = '96px';
        const g = make2DGrid(12, 'droplet');
        // Advance 4 steps for an immediately striking initial frame
        for (let k = 0; k < 4; k++) nca2DFullStep(g, item.rule, false);
        draw2DGrid(canvas, g, color);
        cardAnimations.push({ canvas, grid: g, rule: item.rule, color, step: 4 });
      } else {
        canvas.width = 180; canvas.height = 96;
        canvas.style.width = '100%'; canvas.style.height = '96px';
        const rows = nca1DRollout(item.rule, mode, 36, 19, 'random');
        draw1DSpacetime(canvas, rows, color);
      }
      stage.append(canvas);
      card.append(stage);

      // Bit-pattern visual strip
      const strip = el('div', 'bit-strip' + (nBits === 8 ? ' b8' : ''));
      for (let b = 0; b < nBits; b++) {
        const bitOn = (item.rule >> b) & 1;
        const bc = el('span', 'bit-cell' + (bitOn ? ' on' : ''));
        if (bitOn) bc.style.background = color;
        strip.append(bc);
      }
      card.append(strip);
      card.append(el('div', 'fam-bytes', `r.${item.round} · dmg ${item.dmg.toFixed(2)} · sp ${item.spread.toFixed(2)}`));

      card.addEventListener('click', () => openNCAInspector(item, `Frontier elite · discovered r.${item.round}`));
      grid.append(card);
    }

    startCardTimer();
    return;
  }

  // Brainfuck / Forth Appendix C Families
  $('strip-eyebrow').textContent = 'Appendix C · Mathematical Sequence Families';
  $('strip-note').textContent = 'mod 256 recurrences · ≤30 leading bytes · minimal period ≥ 30 · click card to inspect';
  $('btn-play-cards').style.display = 'none';

  const baseline = (s && s.baseline) || { hits: {}, expected_round: {} };
  for (const fam of FAMILIES) {
    const rec = s && s.discoveries && s.discoveries[fam];
    const exp = baseline.expected_round && baseline.expected_round[fam];
    const zeroHits = baseline.hits && baseline.hits[fam] === 0;
    const hits = (s && s.sample_hits && s.sample_hits[fam]) || 0;
    const color = famColor(fam);

    const card = el('div', 'fam-card ' + (rec ? 'found' : 'pending'));
    if (rec) card.style.borderTopColor = color;

    const top = el('div', 'fam-top');
    top.append(el('span', 'dot c-' + (rec ? fam : 'none')), el('span', 'fam-name cap', fam));
    if (rec) {
      const sp = fmtSpeedup(exp, rec.round, zeroHits);
      const badge = el('span', 'speed-pill', sp || `r.${rec.round}`);
      badge.style.color = color;
      top.append(badge);
    } else {
      top.append(el('span', 'speed-pill muted', 'pending'));
    }
    card.append(top);

    if (rec) {
      const lastSeen = rec.last_seen_round != null ? ` · last r.${rec.last_seen_round}` : '';
      card.append(el('div', 'fam-meta', `r.${rec.round} · ${fmtInt(hits)} hit${hits === 1 ? '' : 's'}${lastSeen}`));
      const chip = el('div', 'prog-chip'); chip.title = rec.program; chip.append(renderCode(rec.program));
      card.append(chip);
      const wave = el('div', 'fam-wave'); wave.append(makeSparkline(rec.output_head, color, 200, 28, true));
      card.append(wave);
      const bRow = el('div', 'fam-bytes', fmtBytes(rec.output_head)); bRow.title = fmtBytes(rec.output_head);
      card.append(bRow);
      card.addEventListener('click', () => openProgramInspector({
        source: `First discovery · round ${rec.round}`, family: fam, program: rec.program,
        output_head: rec.output_head, length: rec.program.length, loops: rec.loops, steps: rec.steps, hits,
      }));
    } else {
      card.append(el('div', 'fam-meta', exp != null ? `uniform ${zeroHits ? '>' : '≈'} ${fmtInt(exp)} rds` : 'awaiting baseline'));
      card.append(el('div', 'prog-chip empty', FORMULAS[fam]));
      const wave = el('div', 'fam-wave'); wave.append(makeSparkline([], color, 200, 28, false));
      card.append(wave);
      card.append(el('div', 'fam-bytes', 'no matching program yet'));
    }
    grid.append(card);
  }
}

/* ================= Archive Table & Recent Stream (Dual Mode) ================= */
function renderArchive(s) {
  const ncaMode = isNCA(s);
  const allRows = (s && s.archive_rows) || [];
  const head = $('archive-head');
  const bar = $('family-pills');
  const tbody = $('archive-rows');
  head.replaceChildren(); bar.replaceChildren(); tbody.replaceChildren();

  const q = $('filter').value.trim().toLowerCase();
  $('filter-clear').classList.toggle('show', q.length > 0);

  if (ncaMode) {
    $('archive-title').textContent = 'Frontier Archive Elites';
    $('filter').placeholder = 'Filter hex, bits, class…';
    const mode = s.nca.mode || 'nca2d';

    // Pills: All + 4 Wolfram classes
    const counts = { all: allRows.length, complex: 0, periodic: 0, chaotic: 0, frozen: 0 };
    for (const r of allRows) if (counts[r.cls] != null) counts[r.cls]++;
    for (const [key, lbl, dotCls] of [
      ['all', 'All', null],
      ['complex', 'IV · Complex', 'complex'],
      ['periodic', 'II · Periodic', 'periodic'],
      ['chaotic', 'III · Chaotic', 'chaotic'],
      ['frozen', 'I · Frozen', 'frozen'],
    ]) {
      const p = el('button', 'pill' + (activeFilterPill === key ? ' active' : '') + (counts[key] === 0 && key !== 'all' ? ' zero' : ''));
      p.type = 'button';
      if (dotCls && activeFilterPill !== key) p.append(el('span', 'dot c-' + dotCls));
      p.append(document.createTextNode(lbl), el('span', 'cnt', String(counts[key])));
      p.addEventListener('click', () => { activeFilterPill = key; renderArchive(state); });
      bar.append(p);
    }

    // Headers
    const hr = el('tr');
    const cols = [
      ['cls', 'Class', false],
      ['rule', 'Rule', false],
      [null, 'Dynamics', false],
      ['reward', 'Reward', true],
      ['entropy', 'Entropy H', true],
      ['dmg', 'Dmg', true],
      ['spread', 'Spread', true],
      ['age', 'Age', true],
    ];
    for (const [key, label, isNum] of cols) {
      const th = el('th', (isNum ? 'num ' : '') + (key ? 'sortable' : ''));
      th.append(document.createTextNode(label));
      if (key) {
        const arr = el('span', 'sort-arrow', sortCol === key ? (sortDir === -1 ? '↓' : '↑') : '');
        th.append(arr);
        th.addEventListener('click', () => {
          if (sortCol === key) sortDir *= -1;
          else { sortCol = key; sortDir = key === 'cls' ? 1 : -1; }
          renderArchive(state);
        });
      }
      hr.append(th);
    }
    head.append(hr);

    let rows = allRows.filter(r => {
      if (activeFilterPill !== 'all' && r.cls !== activeFilterPill) return false;
      if (!q) return true;
      return r.hex.toLowerCase().includes(q) || r.bits.includes(q) || r.cls.includes(q);
    });
    rows = rows.slice().sort((a, b) => {
      if (sortCol === 'cls') return a.cls.localeCompare(b.cls) * sortDir;
      return ((a[sortCol] ?? 0) - (b[sortCol] ?? 0)) * sortDir;
    });

    const maxRew = Math.max(1e-6, ...allRows.map(r => Math.abs(r.reward || 0)));
    if (!rows.length) {
      const tr = el('tr');
      const td = el('td', 'empty-state', 'No NCA archive elites match the current filter.');
      td.colSpan = 8; tr.append(td); tr.style.cursor = 'default'; tbody.append(tr);
    } else {
      for (const r of rows) {
        const tr = el('tr');
        const color = clsColor(r.cls);

        const tdCls = el('td');
        const cWrap = el('span', 'fam-cell');
        cWrap.append(el('span', 'dot c-' + r.cls), el('span', null, CA_CLASS_SHORT[r.cls] || r.cls));
        tdCls.append(cWrap);

        const tdRule = el('td');
        const rWrap = el('div', 'prog-inline');
        rWrap.append(el('strong', null, r.hex), el('span', 'bytes-inline', `  hw ${r.hw} · vac ${r.bit0}`));
        tdRule.append(rWrap);

        const tdPrev = el('td');
        const pWrap = el('div', 'out-cell');
        pWrap.append(makeNCAThumb(r.rule, mode, color, mode === 'nca2d' ? 24 : 44, 24));
        const bSpan = el('span', 'bytes-inline', r.bits);
        bSpan.title = r.bits;
        pWrap.append(bSpan);
        tdPrev.append(pWrap);

        const tdRew = el('td', 'num');
        const rCell = el('span', 'reward-cell');
        const track = el('span', 'reward-bar');
        const fill = el('span', 'reward-fill');
        fill.style.width = `${Math.min(100, (Math.max(0, r.reward || 0) / maxRew) * 100).toFixed(0)}%`;
        fill.style.background = color;
        track.append(fill);
        rCell.append(track, document.createTextNode((r.reward ?? 0).toFixed(4)));
        tdRew.append(rCell);

        tr.append(
          tdCls, tdRule, tdPrev, tdRew,
          el('td', 'num', (r.entropy ?? 0).toFixed(2)),
          el('td', 'num', (r.dmg ?? 0).toFixed(2)),
          el('td', 'num', (r.spread ?? 0).toFixed(2)),
          el('td', 'num', fmtInt(r.age))
        );
        tr.addEventListener('click', () => openNCAInspector(r, `Archive elite · age ${r.age}`));
        tbody.append(tr);
      }
    }
    $('archive-count').textContent = `${rows.length} shown · ${fmtInt(s.archive_size)} total`;
    return;
  }

  // Brainfuck / Forth Archive Table
  $('archive-title').textContent = 'Archive Elites';
  $('filter').placeholder = 'Filter programs…';

  const counts = { all: allRows.length, other: 0 };
  for (const f of FAMILIES) counts[f] = 0;
  for (const r of allRows) { if (r.family && counts[r.family] != null) counts[r.family]++; else counts.other++; }
  for (const [key, label, dotFam] of [
    ['all', 'All', null],
    ...FAMILIES.map(f => [f, SHORT_FAM[f][0].toUpperCase() + SHORT_FAM[f].slice(1), f]),
    ['other', 'Other', 'none'],
  ]) {
    const p = el('button', 'pill' + (activeFilterPill === key ? ' active' : '') + (counts[key] === 0 && key !== 'all' ? ' zero' : ''));
    p.type = 'button';
    if (dotFam && activeFilterPill !== key) p.append(el('span', 'dot c-' + dotFam));
    p.append(document.createTextNode(label), el('span', 'cnt', String(counts[key])));
    p.addEventListener('click', () => { activeFilterPill = key; renderArchive(state); });
    bar.append(p);
  }

  const hr = el('tr');
  for (const [key, label, isNum] of [
    ['family', 'Family', false], [null, 'Program', false], [null, 'Output Trace', false],
    ['reward', 'Reward', true], ['length', 'Len', true], ['loops', 'Loops', true], ['age', 'Age', true],
  ]) {
    const th = el('th', (isNum ? 'num ' : '') + (key ? 'sortable' : ''));
    th.append(document.createTextNode(label));
    if (key) {
      th.append(el('span', 'sort-arrow', sortCol === key ? (sortDir === -1 ? '↓' : '↑') : ''));
      th.addEventListener('click', () => {
        if (sortCol === key) sortDir *= -1;
        else { sortCol = key; sortDir = key === 'family' ? 1 : -1; }
        renderArchive(state);
      });
    }
    hr.append(th);
  }
  head.append(hr);

  let rows = allRows.filter(r => {
    if (activeFilterPill === 'other' && r.family) return false;
    if (activeFilterPill !== 'all' && activeFilterPill !== 'other' && r.family !== activeFilterPill) return false;
    if (!q) return true;
    return r.program.toLowerCase().includes(q) || (r.family && r.family.toLowerCase().includes(q)) || fmtBytes(r.output_head).includes(q);
  });
  rows = rows.slice().sort((a, b) => {
    if (sortCol === 'family') return (a.family || 'zzz').localeCompare(b.family || 'zzz') * sortDir;
    return ((a[sortCol] ?? 0) - (b[sortCol] ?? 0)) * sortDir;
  });

  const maxReward = Math.max(1e-6, ...allRows.map(r => Math.abs(r.reward || 0)));
  if (!rows.length) {
    const tr = el('tr');
    const td = el('td', 'empty-state', allRows.length ? 'No archive elites match the current filter.' : 'Archive is empty — waiting for positively rewarded programs.');
    td.colSpan = 7; tr.append(td); tr.style.cursor = 'default'; tbody.append(tr);
  } else {
    for (const r of rows) {
      const tr = el('tr');
      const tdFam = el('td');
      const famWrap = el('span', 'fam-cell');
      famWrap.append(el('span', 'dot c-' + (r.family || 'none')), el('span', null, r.family ? SHORT_FAM[r.family] : '—'));
      tdFam.append(famWrap);

      const tdProg = el('td');
      const pDiv = el('div', 'prog-inline'); pDiv.title = r.program; pDiv.append(renderCode(r.program));
      tdProg.append(pDiv);

      const tdOut = el('td');
      const outWrap = el('div', 'out-cell');
      outWrap.append(makeSparkline(r.output_head, famColor(r.family), 54, 16, false));
      const bSpan = el('span', 'bytes-inline', fmtBytes(r.output_head)); bSpan.title = fmtBytes(r.output_head);
      outWrap.append(bSpan); tdOut.append(outWrap);

      const tdRew = el('td', 'num');
      const rewWrap = el('span', 'reward-cell');
      const barTrack = el('span', 'reward-bar');
      const barFill = el('span', 'reward-fill');
      barFill.style.width = `${Math.min(100, (Math.max(0, r.reward) / maxReward) * 100).toFixed(0)}%`;
      if (r.family) barFill.style.background = famColor(r.family);
      barTrack.append(barFill);
      rewWrap.append(barTrack, document.createTextNode((r.reward ?? 0).toFixed(3)));
      tdRew.append(rewWrap);

      tr.append(tdFam, tdProg, tdOut, tdRew, el('td', 'num', fmtInt(r.length)), el('td', 'num', fmtInt(r.loops)), el('td', 'num', fmtInt(r.age)));
      tr.addEventListener('click', () => openProgramInspector({
        source: `Archive elite · age ${r.age}`, family: r.family, program: r.program,
        output_head: r.output_head, length: r.length, loops: r.loops, age: r.age, reward: r.reward,
      }));
      tbody.append(tr);
    }
  }
  $('archive-count').textContent = `${rows.length} shown · ${fmtInt(s.archive_size)} total`;
}

function renderRecent(s) {
  const feed = $('recent-feed');
  feed.replaceChildren();
  const items = ((s && s.recent) || []).slice().reverse();
  $('recent-sub').textContent = items.length ? `${items.length} newest · fresh` : 'fresh generator stream';

  if (!items.length) {
    feed.append(el('div', 'empty-state', 'No fresh proposals recorded yet.'));
    return;
  }

  if (isNCA(s)) {
    const mode = s.nca.mode || 'nca2d';
    for (const sample of items) {
      const color = clsColor(sample.cls);
      const item = el('div', 'feed-item');
      const top = el('div', 'feed-top');
      top.append(el('span', 'dot c-' + sample.cls), el('div', 'feed-prog', `${sample.hex}  ·  hw ${sample.hw} · vac ${sample.bit0}`));
      const tag = el('span', 'feed-tag', `${CA_CLASS_SHORT[sample.cls]} · H ${(sample.entropy ?? 0).toFixed(2)}`);
      tag.style.color = color;
      top.append(tag);

      const bot = el('div', 'feed-bot');
      bot.append(makeNCAThumb(sample.rule, mode, color, mode === 'nca2d' ? 22 : 40, 22));
      bot.append(el('span', 'bytes-inline', `dmg ${(sample.dmg ?? 0).toFixed(2)} · sp ${(sample.spread ?? 0).toFixed(2)} · flips ${(sample.flips ?? 0).toFixed(2)}`));
      item.append(top, bot);
      item.addEventListener('click', () => openNCAInspector(sample, 'Latest fresh proposal'));
      feed.append(item);
    }
    return;
  }

  for (const sample of items) {
    const item = el('div', 'feed-item');
    const top = el('div', 'feed-top');
    const prog = el('div', 'feed-prog'); prog.title = sample.program; prog.append(renderCode(sample.program));
    top.append(el('span', 'dot c-' + (sample.family || 'none')), prog);
    if (sample.family) {
      const tag = el('span', 'feed-tag', SHORT_FAM[sample.family]);
      tag.style.color = famColor(sample.family);
      top.append(tag);
    }
    const bot = el('div', 'feed-bot');
    bot.append(makeSparkline(sample.output_head, famColor(sample.family), 46, 14, false));
    const b = el('span', 'bytes-inline', fmtBytes(sample.output_head)); b.title = fmtBytes(sample.output_head);
    bot.append(b);
    item.append(top, bot);
    item.addEventListener('click', () => openProgramInspector({
      source: 'Latest fresh proposal', family: sample.family, program: sample.program, output_head: sample.output_head, length: sample.program.length,
    }));
    feed.append(item);
  }
}

/* ================= Copy Helper & Inspectors ================= */
async function copyText(text, btn, doneLabel = 'Copied') {
  const orig = btn.textContent;
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.append(ta); ta.select(); document.execCommand('copy'); ta.remove();
  }
  btn.textContent = doneLabel;
  setTimeout(() => { btn.textContent = orig; }, 1400);
}

function stopInspectorSim() {
  if (inspSimTimer) { clearInterval(inspSimTimer); inspSimTimer = null; }
}

/* Interactive NCA Rule Inspector (with LIVE Forward + Reverse Time Simulation!) */
function openNCAInspector(item, sourceLabel) {
  closeOverlays();
  const mode = (state && state.nca && state.nca.mode) || 'nca2d';
  const nBits = (state && state.nca && state.nca.rule_bits) || 16;
  const color = clsColor(item.cls);

  $('insp-dot').className = 'dot c-' + item.cls;
  $('insp-title').textContent = `${item.hex} · ${CA_CLASS_LABELS[item.cls] || item.cls}`;
  $('insp-sub').textContent = sourceLabel || '';

  const body = $('insp-body');
  body.replaceChildren();

  // Live Interactive Simulator Section
  const simSec = el('div', 'insp-section');
  const simHead = el('div', 'insp-label-row');
  simHead.append(el('span', 'insp-label', mode === 'nca2d' ? 'Live Reversible nca.py Simulator (20×20 Torus)' : 'Spacetime Rollout'));
  const stepBadge = el('span', 'drawer-sub', 't = 0');
  simHead.append(stepBadge);
  simSec.append(simHead);

  const simBox = el('div', 'sim-box');
  const canvas = el('canvas');
  let seedKind = 'droplet';
  let simPlaying = true;
  let step = 0;
  let grid = null;

  const resetSim = () => {
    step = 0;
    if (mode === 'nca2d') {
      grid = make2DGrid(20, seedKind);
      draw2DGrid(canvas, grid, color);
      stepBadge.textContent = 't = 0';
    } else {
      const rows = nca1DRollout(item.rule, mode, 48, 32, seedKind);
      draw1DSpacetime(canvas, rows, color);
      stepBadge.textContent = '32 steps';
    }
  };

  if (mode === 'nca2d') {
    canvas.width = 240; canvas.height = 240;
    canvas.style.width = '240px'; canvas.style.height = '240px';
  } else {
    canvas.width = 384; canvas.height = 192;
    canvas.style.width = '100%'; canvas.style.height = '192px';
  }
  simBox.append(canvas);

  const controls = el('div', 'sim-controls');
  if (mode === 'nca2d') {
    const leftBtns = el('div', 'sim-btn-group');
    const btnPlay = el('button', 'btn sm', 'Pause');
    btnPlay.type = 'button';
    btnPlay.addEventListener('click', () => {
      simPlaying = !simPlaying;
      btnPlay.textContent = simPlaying ? 'Pause' : 'Play';
    });
    const btnBack = el('button', 'btn sm', '−1 Rev');
    btnBack.type = 'button';
    btnBack.title = 'Exact time-reversal via inverse checkerboard half-steps';
    btnBack.addEventListener('click', () => {
      simPlaying = false; btnPlay.textContent = 'Play';
      nca2DFullStep(grid, item.rule, true);
      step--;
      draw2DGrid(canvas, grid, color);
      stepBadge.textContent = `t = ${step}`;
    });
    const btnFwd = el('button', 'btn sm', '+1');
    btnFwd.type = 'button';
    btnFwd.addEventListener('click', () => {
      simPlaying = false; btnPlay.textContent = 'Play';
      nca2DFullStep(grid, item.rule, false);
      step++;
      draw2DGrid(canvas, grid, color);
      stepBadge.textContent = `t = ${step}`;
    });
    const btnReset = el('button', 'btn sm', 'Reset');
    btnReset.type = 'button';
    btnReset.addEventListener('click', resetSim);
    leftBtns.append(btnPlay, btnBack, btnFwd, btnReset);
    controls.append(leftBtns);
  }

  const seedSeg = el('div', 'seg');
  const seedOpts = mode === 'nca2d' ? [['droplet', 'Droplet'], ['random', '25%'], ['cross', 'Cross']] : [['droplet', 'Seed'], ['random', '25%']];
  const seedBtns = [];
  for (const [k, lbl] of seedOpts) {
    const b = el('button', k === seedKind ? 'active' : '', lbl);
    b.type = 'button';
    b.addEventListener('click', () => {
      seedKind = k;
      seedBtns.forEach(([bk, bn]) => bn.classList.toggle('active', bk === k));
      resetSim();
    });
    seedBtns.push([k, b]);
    seedSeg.append(b);
  }
  controls.append(seedSeg);
  simBox.append(controls);
  simSec.append(simBox);
  body.append(simSec);

  resetSim();
  if (mode === 'nca2d') {
    inspSimTimer = setInterval(() => {
      if (!simPlaying) return;
      nca2DFullStep(grid, item.rule, false);
      step++;
      draw2DGrid(canvas, grid, color);
      stepBadge.textContent = `t = ${step}`;
    }, 280);
  }

  // Metrics strip
  const kv = el('div', 'kv-mini');
  for (const [k, v] of [
    ['Block H', (item.entropy ?? 0).toFixed(3)],
    ['Damage', (item.dmg ?? 0).toFixed(3)],
    ['Spread / λ', `${(item.spread ?? 0).toFixed(2)} / ${(item.hw / nBits).toFixed(2)}`],
    ['Reward', item.reward != null ? item.reward.toFixed(4) : '—'],
  ]) {
    const cell = el('div', 'kv-cell');
    cell.append(el('div', 'k', k), el('div', 'v', v));
    kv.append(cell);
  }
  body.append(kv);

  // Copyable formats
  const copySec = el('div', 'insp-section');
  const copyRow = el('div', 'insp-label-row');
  copyRow.append(el('span', 'insp-label', 'Rule Encoding'));
  const btnGroup = el('div', 'sim-btn-group');
  const btnHex = el('button', 'btn sm', 'Copy Bits');
  btnHex.type = 'button';
  btnHex.addEventListener('click', () => copyText(`${item.hex} (${item.bits})`, btnHex));
  const btnNca = el('button', 'btn sm', 'Copy nca.py Program');
  btnNca.type = 'button';
  btnNca.addEventListener('click', () => copyText(synthesizeNcaPyProgram(item.rule, mode), btnNca));
  btnGroup.append(btnHex, btnNca);
  copyRow.append(btnGroup);
  copySec.append(copyRow, el('div', 'code-box', `${item.hex}  ·  bits[0..${nBits - 1}] = ${item.bits}  ·  vacuum(0) → ${item.bit0}`));
  body.append(copySec);

  // Visual Neighborhood Truth Table
  const ttSec = el('div', 'insp-section');
  ttSec.append(el('span', 'insp-label', mode === 'nca2d' ? 'Von Neumann Neighborhood Table (N,S,W,E → Δ)' : 'Local Neighborhood Table (→ Δ)'));
  const tGrid = el('div', 'truth-grid');
  for (let nb = 0; nb < nBits; nb++) {
    const on = (item.rule >> nb) & 1;
    const cell = el('div', 'truth-cell' + (on ? ' on' : ''));
    if (mode === 'nca2d') {
      const n = (nb >> 3) & 1, s = (nb >> 2) & 1, w = (nb >> 1) & 1, e = nb & 1;
      const cross = el('div', 'vn-cross');
      const slots = [0, n ? 1 : -1, 0, w ? 1 : -1, 2, e ? 1 : -1, 0, s ? 1 : -1, 0];
      for (const st of slots) {
        const px = el('span', 'vn-px ' + (st === 1 ? 'on' : st === -1 ? 'off' : st === 2 ? 'ctr' : ''));
        if (st === 1 && on) px.style.background = color;
        cross.append(px);
      }
      cell.append(cross, el('span', null, `${nb.toString(2).padStart(4, '0')}→${on}`));
    } else {
      cell.append(el('span', null, nb.toString(2).padStart(3, '0')), el('span', null, `→ ${on}`));
    }
    tGrid.append(cell);
  }
  ttSec.append(tGrid);
  body.append(ttSec);

  $('backdrop').classList.add('show');
  $('inspector').classList.add('open');
  $('inspector').setAttribute('aria-hidden', 'false');
}

/* Brainfuck / Forth Program Inspector */
function analyzeSequence(bytes) {
  if (!bytes || bytes.length < 3) return { note: 'Too few output bytes to compute finite differences.', d1: [], d2: [] };
  const d1 = [null];
  for (let i = 1; i < bytes.length; i++) d1.push((bytes[i] - bytes[i - 1] + 256) % 256);
  const d2 = [null, null];
  for (let i = 2; i < bytes.length; i++) d2.push((d1[i] - d1[i - 1] + 256) % 256);
  const d1Vals = d1.slice(1), d2Vals = d2.slice(2);
  let note = `Showing first ${bytes.length} emitted bytes (Appendix C evaluates full tape after ≤30 leading bytes).`;
  if (d1Vals.every(v => v === d1Vals[0])) {
    note = d1Vals[0] === 0 ? `Constant prefix: vₙ ≡ ${bytes[0]} (mod 256).` : `Exact arithmetic progression on head: Δ¹v ≡ +${d1Vals[0]} (mod 256).`;
  } else if (d2Vals.length >= 2 && d2Vals.every(v => v === d2Vals[0])) {
    note = `Exact quadratic recurrence on head: constant Δ²v ≡ +${d2Vals[0]} (mod 256).`;
  } else if (bytes.length >= 4 && bytes.slice(2).every((v, i) => (bytes[i] + bytes[i + 1]) % 256 === v)) {
    note = 'Exact Fibonacci recurrence on head: vₙ ≡ vₙ₋₁ + vₙ₋₂ (mod 256).';
  } else {
    for (let r = 2; r < 256; r++) {
      if (bytes.slice(1).every((v, i) => (r * bytes[i]) % 256 === v)) {
        note = `Exact geometric progression on head: vₙ ≡ ${r} · vₙ₋₁ (mod 256).`; break;
      }
    }
  }
  return { note, d1, d2 };
}

function openProgramInspector(item) {
  closeOverlays();
  const fam = item.family, color = famColor(fam);
  $('insp-dot').className = 'dot c-' + (fam || 'none');
  $('insp-title').textContent = fam ? `${fam[0].toUpperCase() + fam.slice(1)} Program` : 'Program Inspector';
  $('insp-sub').textContent = item.source || '';

  const body = $('insp-body');
  body.replaceChildren();

  const secCode = el('div', 'insp-section');
  const headRow = el('div', 'insp-label-row');
  headRow.append(el('span', 'insp-label', `${executorKind()} source`));
  const copyBtn = el('button', 'btn sm', 'Copy program');
  copyBtn.type = 'button';
  copyBtn.addEventListener('click', () => copyText(item.program, copyBtn));
  headRow.append(copyBtn);
  const codeBox = el('div', 'code-box'); codeBox.append(renderCode(item.program));
  secCode.append(headRow, codeBox);
  body.append(secCode);

  if (executorKind() === 'brainfuck') {
    const haltIdx = item.program.indexOf('F');
    const active = haltIdx === -1 ? item.program : item.program.slice(0, haltIdx);
    const used = [...new Set([...active].filter(ch => BF_MACROS[ch]))];
    if (used.length || haltIdx !== -1) {
      const secMacro = el('div', 'insp-section');
      secMacro.append(el('span', 'insp-label', 'Appendix E macro expansion'));
      if (used.length) {
        const list = el('div', 'macro-list');
        for (const ch of used) {
          const chip = el('span', 'macro-chip');
          chip.append(el('b', null, ch), document.createTextNode(` = ${BF_MACROS[ch][0]} (${BF_MACROS[ch][1]})`));
          list.append(chip);
        }
        secMacro.append(list, el('div', 'code-box', [...active].map(ch => BF_MACROS[ch] ? BF_MACROS[ch][0] : ch).join('')));
      }
      if (haltIdx !== -1) secMacro.append(el('div', 'recurrence-note', `Halts at token ${haltIdx + 1} ('F'); ${item.program.length - haltIdx - 1} trailing token(s) ignored.`));
      body.append(secMacro);
    }
  }

  const kv = el('div', 'kv-mini');
  for (const [k, v] of [
    ['Length', fmtInt(item.length ?? item.program.length)],
    ['Loops', item.loops != null ? fmtInt(item.loops) : '—'],
    item.reward != null ? ['Reward', item.reward.toFixed(3)] : ['Steps', item.steps != null ? fmtInt(item.steps) : '—'],
    item.hits != null ? ['Sample Hits', fmtInt(item.hits)] : (item.age != null ? ['Age', `${fmtInt(item.age)} rd`] : ['Family', fam || 'none']),
  ]) {
    const cell = el('div', 'kv-cell'); cell.append(el('div', 'k', k), el('div', 'v', v)); kv.append(cell);
  }
  body.append(kv);

  const bytes = item.output_head || [];
  const secPlot = el('div', 'insp-section');
  secPlot.append(el('span', 'insp-label', `Emitted byte head (${bytes.length} bytes · 0–255)`));
  const plotSvg = svgEl('svg', { viewBox: '0 0 400 106', style: 'width:100%;height:106px;display:block;border:1px solid var(--border-subtle);border-radius:8px;background:var(--surface-2)' });
  const pL = 38, pR = 14, pT = 10, pB = 18;
  const pW = 400 - pL - pR, pH = 106 - pT - pB;
  for (const val of [0, 128, 255]) {
    const y = pT + pH * (1 - val / 255);
    plotSvg.append(svgEl('line', { x1: pL, y1: y, x2: 400 - pR, y2: y, stroke: cssVar('--border'), 'stroke-width': 1 }));
    const t = svgEl('text', { x: pL - 7, y: y + 3, 'text-anchor': 'end', fill: cssVar('--dim'), 'font-size': '9', 'font-family': cssVar('--mono') });
    t.textContent = String(val); plotSvg.append(t);
  }
  if (bytes.length) {
    const pts = bytes.map((b, i) => [bytes.length === 1 ? pL + pW / 2 : pL + (pW * i) / (bytes.length - 1), pT + pH * (1 - b / 255)]);
    if (pts.length > 1) {
      const areaD = `M ${pts[0][0]} ${pT + pH} ` + pts.map(p => `L ${p[0]} ${p[1]}`).join(' ') + ` L ${pts[pts.length - 1][0]} ${pT + pH} Z`;
      plotSvg.append(svgEl('path', { d: areaD, fill: color, 'fill-opacity': '0.12' }));
      plotSvg.append(svgEl('polyline', { points: pts.map(p => `${p[0]},${p[1]}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 1.75, 'stroke-linejoin': 'round' }));
    }
    pts.forEach(([x, y], i) => {
      plotSvg.append(svgEl('circle', { cx: x, cy: y, r: 2.7, fill: color }));
      if (bytes.length <= 16) {
        const lbl = svgEl('text', { x, y: 102, 'text-anchor': 'middle', fill: cssVar('--dim'), 'font-size': '8.5', 'font-family': cssVar('--mono') });
        lbl.textContent = String(i); plotSvg.append(lbl);
      }
    });
  }
  secPlot.append(plotSvg); body.append(secPlot);

  const { note, d1, d2 } = analyzeSequence(bytes);
  const secDiff = el('div', 'insp-section');
  secDiff.append(el('span', 'insp-label', 'Mod-256 finite differences'), el('div', 'recurrence-note', note));
  if (bytes.length) {
    const tbl = el('table', 'diff-table');
    const thead = el('thead'), hr = el('tr');
    for (const h of ['i', 'vᵢ', 'hex', 'Δ¹ (mod 256)', 'Δ² (mod 256)']) hr.append(el('th', null, h));
    thead.append(hr);
    const tb = el('tbody');
    bytes.forEach((v, i) => {
      const tr = el('tr');
      tr.append(el('td', null, String(i)), el('td', null, String(v)), el('td', null, '0x' + v.toString(16).toUpperCase().padStart(2, '0')), el('td', null, d1[i] == null ? '—' : `+${d1[i]}`), el('td', null, d2[i] == null ? '—' : `+${d2[i]}`));
      tb.append(tr);
    });
    tbl.append(thead, tb); secDiff.append(tbl);
  }
  body.append(secDiff);

  $('backdrop').classList.add('show');
  $('inspector').classList.add('open');
  $('inspector').setAttribute('aria-hidden', 'false');
}

/* ================= Exporters & Config Modal ================= */
function buildMarkdownTable(s) {
  if (isNCA(s)) {
    const base = s.nca.uniform_baseline || { mean_entropy: 0, class_rates: {} };
    const hist = s.nca.history || [];
    const last = hist[hist.length - 1] || base;
    return [
      `| Dynamical Class (${s.nca.mode}, ${fmtInt(s.nca.rule_space)} rules) | Uniform Prior | Self-Play (r.${s.round}) | Shift |`,
      '|---|---:|---:|---:|',
      ...CA_CLASSES.map(c => {
        const u = 100 * (base.class_rates[c] || 0);
        const now = 100 * (last.class_rates[c] || 0);
        return `| ${CA_CLASS_LABELS[c]} | ${u.toFixed(1)}% | ${now.toFixed(1)}% | ${(now - u >= 0 ? '+' : '') + (now - u).toFixed(1)} pp |`;
      }),
    ].join('\n');
  }

  const b = s.baseline || { hits: {}, expected_round: {} };
  const lines = ['| Family | Self-Play Round | Sample Hits | Uniform E[round] | Speedup | Program |', '|---|---:|---:|---:|---:|---|'];
  for (const f of FAMILIES) {
    const rec = s.discoveries && s.discoveries[f];
    const exp = b.expected_round && b.expected_round[f];
    const zero = b.hits && b.hits[f] === 0;
    const expStr = exp == null ? '—' : `${zero ? '>' : '≈'}${fmtInt(exp)}`;
    const spd = rec ? (fmtSpeedup(exp, rec.round, zero) || '—') : '—';
    const prog = rec ? '`' + rec.program.replace(/`/g, '\\`') + '`' : '—';
    lines.push(`| ${f} | ${rec ? rec.round : '—'} | ${fmtInt((s.sample_hits || {})[f] || 0)} | ${expStr} | ${spd} | ${prog} |`);
  }
  return lines.join('\n');
}

function buildLatexTable(s) {
  if (isNCA(s)) {
    const base = s.nca.uniform_baseline || { class_rates: {} };
    const hist = s.nca.history || [];
    const last = hist[hist.length - 1] || base;
    return [
      '\\begin{tabular}{lrrr}',
      '  \\toprule',
      `  Class (${s.nca.mode}) & Uniform & Self-Play (r.${s.round}) & Shift \\\\`,
      '  \\midrule',
      ...CA_CLASSES.map(c => {
        const u = 100 * (base.class_rates[c] || 0);
        const now = 100 * (last.class_rates[c] || 0);
        return `  ${CA_CLASS_LABELS[c]} & ${u.toFixed(1)}\\% & ${now.toFixed(1)}\\% & ${(now - u >= 0 ? '+' : '') + (now - u).toFixed(1)}\\,pp \\\\`;
      }),
      '  \\bottomrule',
      '\\end{tabular}',
    ].join('\n');
  }
  const b = s.baseline || { hits: {}, expected_round: {} };
  const esc = str => str.replace(/([\\&#%_{}$~^])/g, '\\$1');
  const rows = FAMILIES.map(f => {
    const rec = s.discoveries && s.discoveries[f];
    const exp = b.expected_round && b.expected_round[f];
    const zero = b.hits && b.hits[f] === 0;
    const expStr = exp == null ? '--' : `${zero ? '$>$' : '$\\approx$'}${fmtInt(exp)}`;
    const spd = rec ? (fmtSpeedup(exp, rec.round, zero) || '--').replace('×', '$\\times$').replace('>', '$>$') : '--';
    const prog = rec ? `\\texttt{${esc(rec.program)}}` : '--';
    return `  ${f[0].toUpperCase() + f.slice(1)} & ${rec ? rec.round : '--'} & ${fmtInt((s.sample_hits || {})[f] || 0)} & ${expStr} & ${spd} & ${prog} \\\\`;
  });
  return ['\\begin{tabular}{lrrrrl}', '  \\toprule', '  Family & Round & Hits & Uniform $\\mathbb{E}[\\text{round}]$ & Speedup & Program \\\\', '  \\midrule', ...rows, '  \\bottomrule', '\\end{tabular}'].join('\n');
}

function openConfigModal() {
  if (!state || !state.config) return;
  closeOverlays();
  const c = state.config;
  $('modal-sub').textContent = `${c.out || 'run'} · round ${fmtInt(state.round)} · ${state.status}`;

  const body = $('modal-body');
  body.replaceChildren();

  const expSec = el('div', 'insp-section');
  expSec.append(el('span', 'insp-label', isNCA(state) ? 'Export NCA Summary & Snapshot' : 'Export Table 1 & Snapshot'));
  const row = el('div', 'export-row');
  const btnMd = el('button', 'btn', 'Copy Markdown');
  btnMd.type = 'button';
  btnMd.addEventListener('click', () => copyText(buildMarkdownTable(state), btnMd, 'Copied Markdown'));
  const btnTex = el('button', 'btn', 'Copy LaTeX');
  btnTex.type = 'button';
  btnTex.addEventListener('click', () => copyText(buildLatexTable(state), btnTex, 'Copied LaTeX'));
  const btnJson = el('button', 'btn', 'Download live.json');
  btnJson.type = 'button';
  btnJson.addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `self_play_${isNCA(state) ? state.nca.mode : 'table1'}_round_${state.round}.json`;
    a.click(); URL.revokeObjectURL(a.href);
  });
  row.append(btnMd, btnTex, btnJson);
  expSec.append(row); body.append(expSec);

  const groups = isNCA(state) ? [
    ['Architecture', [
      ['learner', `${c.dim}d × ${c.depth}L (${c.heads}h)`],
      ['generator', `${c.gen_dim}d × ${c.gen_depth}L`],
      ['parameters', `${((state.num_parameters || 0) / 1e6).toFixed(2)}M`],
      ['device', String(c.device)],
    ]],
    ['Self-Play & Disagreement', [
      ['rounds / batch', `${fmtInt(c.rounds)} / ${c.batch_size}`],
      ['jvp reward', String(c.jvp_reward_mode || 'abs')],
      ['disagreement', c.use_disagreement ? `on (w=${c.disagreement_weight}, k=${c.num_disagreement_models})` : 'off (JVP only)'],
      ['learner / gen lr', `${c.learner_lr} / ${c.generator_lr}`],
      ['ema / kl', `${c.ema_decay} / ${c.kl_weight}`],
    ]],
    ['NCA Physics (nca.py)', [
      ['mode / space', `${c.mode} (${fmtInt(state.nca.rule_space)} rules)`],
      ['grid / frames', `${c.mode === 'nca2d' ? `${c.height}×${c.width}` : c.mode === 'nca1d' ? `2×${c.width}` : `${c.width} cells`} · ${c.num_frames}f`],
      ['stride / burn-in', `${c.stride} / ${c.burn_in}`],
      ['init density', String(c.density)],
    ]],
    ['Ablations & Archive', [
      ['seed', String(c.seed)],
      ['archive_max_age', String(c.archive_max_age)],
      ['ablate_generator', c.ablate_generator || 'none'],
      ['ablate_rew / prop', `${c.ablate_reward || 'none'} / ${c.ablate_proposals || 'none'}`],
    ]],
  ] : [
    ['Architecture', [
      ['dim × depth', `${c.dim} × ${c.depth}`],
      ['heads × dim_head', `${c.heads} × ${c.dim_head}`],
      ['parameters', `${((state.num_parameters || 0) / 1e6).toFixed(2)}M`],
      ['device / threads', `${c.device} · ${c.threads}t`],
    ]],
    ['Self-Play & RL', [
      ['rounds / batch', `${fmtInt(c.rounds)} / ${c.batch_size}`],
      ['reference', c.reference === 'ema' ? `ema (${c.ema_decay})` : c.reference],
      ['learner / gen lr', `${c.learner_lr} / ${c.generator_lr}`],
      ['kl / rep align', `${c.kl_weight} / ${c.representation_alignment_loss_weight}`],
    ]],
    ['Executor & Detection', [
      ['executor', c.executor || 'brainfuck'],
      ['max_len / tape', `${c.max_length} / ${c.max_output}`],
      ['max_steps', fmtInt(c.max_steps)],
      ['detect / baseline', `${fmtInt(c.detect_samples)} / ${fmtInt(c.baseline_samples)}`],
    ]],
    ['Ablations & Archive', [
      ['seed', String(c.seed)],
      ['bucketed_archive', String(!!c.bucketed_archive)],
      ['ablate_generator', c.ablate_generator || 'none'],
      ['ablate_rew / prop', `${c.ablate_reward || 'none'} / ${c.ablate_proposals || 'none'}`],
    ]],
  ];

  const gGrid = el('div', 'cfg-groups');
  for (const [title, pairs] of groups) {
    const box = el('div', 'cfg-group'); box.append(el('h4', null, title));
    for (const [k, v] of pairs) {
      const r = el('div', 'cfg-row'); r.append(el('span', 'ck', k), el('span', 'cv', v)); box.append(r);
    }
    gGrid.append(box);
  }
  body.append(gGrid);

  if (!isNCA(state) && executorKind() === 'brainfuck') {
    const refSec = el('div', 'insp-section');
    refSec.append(el('span', 'insp-label', 'Brainfuck Appendix E Macros (F = halt)'));
    const mg = el('div', 'macro-grid');
    for (const [k, [exp, desc]] of Object.entries(BF_MACROS)) {
      const chip = el('div', 'macro-chip'); chip.append(el('b', null, k), document.createTextNode(`  ${exp}  · ${desc}`)); mg.append(chip);
    }
    refSec.append(mg); body.append(refSec);
  }

  $('backdrop').classList.add('show');
  $('config-modal').classList.add('open');
  $('config-modal').setAttribute('aria-hidden', 'false');
}

function closeOverlays() {
  stopInspectorSim();
  $('backdrop').classList.remove('show');
  $('inspector').classList.remove('open');
  $('inspector').setAttribute('aria-hidden', 'true');
  $('config-modal').classList.remove('open');
  $('config-modal').setAttribute('aria-hidden', 'true');
}

/* ================= Theme, Polling & Wiring ================= */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('sp_theme', theme);
  $('btn-theme').innerHTML = theme === 'dark'
    ? '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1"/></svg>'
    : '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13.5 9.8A5.5 5.5 0 0 1 6.2 2.5a5.5 5.5 0 1 0 7.3 7.3Z"/></svg>';
  if (state) renderAll(state, true);
}

function syncToggles() {
  $('unit-bits').classList.toggle('active', lossUnit === 'bits');
  $('unit-nats').classList.toggle('active', lossUnit === 'nats');
  $('xscale-lin').classList.toggle('active', lossXScale === 'lin');
  $('xscale-log').classList.toggle('active', lossXScale === 'log');
  $('view-programs').classList.toggle('active', activeView === 'live');
  $('view-nca').classList.toggle('active', activeView === 'nca');
}

function renderAll(s, force = false) {
  if (!s || !s.config) {
    $('status-dot').className = 'status-dot wait';
    $('run-spec').textContent = 'waiting for live.json…';
    $('banner').classList.add('show');
    $('banner-text').textContent = 'Waiting for live.json…';
    return;
  }

  const prevWasNca = state ? isNCA(state) : null;
  const nowIsNca = isNCA(s);
  if (prevWasNca !== null && prevWasNca !== nowIsNca) {
    activeFilterPill = 'all';
    sortCol = 'reward';
    sortDir = -1;
    $('filter').value = '';
    hoverIdx = null;
    force = true;
  }

  const key = JSON.stringify([
    activeView, s.experiment || 'table1', s.status, s.round, s.updated,
    (s.losses || []).length, s.archive_size,
    nowIsNca ? (s.nca.history || []).length : Object.keys(s.discoveries || {}).length,
  ]);
  state = s;
  if (!force && key === lastKey) return;
  lastKey = key;

  renderSummary(s);
  drawPrimaryChart(s);
  renderSecondaryChart(s);
  renderDiscoveryStrip(s);
  renderArchive(s);
  renderRecent(s);
}

async function fetchMeta() {
  try {
    const meta = await (await fetch('/meta', { cache: 'no-store' })).json();
    if (meta.has_live && meta.has_nca) {
      $('view-switch').style.display = 'inline-flex';
    }
  } catch {}
}

async function pollOnce() {
  try {
    const url = activeView === 'nca' ? '/state?view=nca' : '/state';
    const data = await (await fetch(url, { cache: 'no-store' })).json();
    renderAll(data);
  } catch {
    $('status-dot').className = 'status-dot wait';
    $('banner').classList.add('show');
    $('banner-text').textContent = 'Connection lost — retrying monitor server…';
  }
}

async function pollLoop() {
  await pollOnce();
  setTimeout(pollLoop, 2000);
}

/* Events */
$('unit-bits').addEventListener('click', () => { lossUnit = 'bits'; localStorage.setItem('sp_unit', lossUnit); syncToggles(); if (state) { renderSummary(state); drawPrimaryChart(state); } });
$('unit-nats').addEventListener('click', () => { lossUnit = 'nats'; localStorage.setItem('sp_unit', lossUnit); syncToggles(); if (state) { renderSummary(state); drawPrimaryChart(state); } });
$('xscale-lin').addEventListener('click', () => { lossXScale = 'lin'; localStorage.setItem('sp_xscale', lossXScale); syncToggles(); if (state) drawPrimaryChart(state); });
$('xscale-log').addEventListener('click', () => { lossXScale = 'log'; localStorage.setItem('sp_xscale', lossXScale); syncToggles(); if (state) drawPrimaryChart(state); });

$('view-programs').addEventListener('click', () => { activeView = 'live'; lastKey = null; syncToggles(); closeOverlays(); pollOnce(); });
$('view-nca').addEventListener('click', () => { activeView = 'nca'; lastKey = null; syncToggles(); closeOverlays(); pollOnce(); });

$('btn-play-cards').addEventListener('click', () => {
  cardsPlaying = !cardsPlaying;
  $('btn-play-cards').textContent = cardsPlaying ? 'Pause' : 'Play';
  if (cardsPlaying) startCardTimer(); else if (cardTimer) { clearInterval(cardTimer); cardTimer = null; }
});

$('btn-theme').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
});
$('btn-config').addEventListener('click', openConfigModal);
$('insp-close').addEventListener('click', closeOverlays);
$('modal-close').addEventListener('click', closeOverlays);
$('backdrop').addEventListener('click', closeOverlays);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeOverlays(); });

$('filter').addEventListener('input', () => state && renderArchive(state));
$('filter-clear').addEventListener('click', () => { $('filter').value = ''; state && renderArchive(state); $('filter').focus(); });

$('primary-canvas').addEventListener('mousemove', handlePrimaryHover);
$('primary-canvas').addEventListener('mouseleave', () => {
  hoverIdx = null;
  $('primary-tooltip').classList.remove('show');
  if (state) drawPrimaryChart(state);
});

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state) { drawPrimaryChart(state); renderSecondaryChart(state); } }, 60);
});

const urlTheme = new URLSearchParams(location.search).get('theme');
applyTheme(urlTheme === 'dark' || urlTheme === 'light' ? urlTheme : (localStorage.getItem('sp_theme') || 'light'));
syncToggles();
fetchMeta();
pollLoop();
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in ('/', '/index.html'):
            body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
        elif path == '/meta':
            nca_path = getattr(self.server, 'nca_path', None)
            meta = dict(
                has_live = bool(self.server.live_path and self.server.live_path.exists()),
                has_nca = bool(nca_path and nca_path.exists())
            )
            body = json.dumps(meta).encode()
            content_type = 'application/json; charset=utf-8'
        elif path == '/state':
            view = params.get('view', ['live'])[0]
            nca_path = getattr(self.server, 'nca_path', None)
            target = nca_path if (view == 'nca' and nca_path) else self.server.live_path
            body = target.read_bytes() if (target and target.exists()) else b'{}'
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

def make_server(live, host = '127.0.0.1', port = 0, nca = None):
    server = ThreadingHTTPServer((host, port), Handler)
    server.live_path = Path(live) if live else None
    server.nca_path = Path(nca) if nca else None
    return server

def serve(
    live = 'experiments/runs/default/live.json',
    nca = None,
    host = '127.0.0.1',
    port = 8000,
    label = None,
    open_browser = False
):
    server = make_server(live, host, port, nca = nca)
    server.label = default(label, server.live_path.parent.name if server.live_path else 'monitor')

    url = f'http://{host}:{server.server_address[1]}'

    extra = f' + nca {server.nca_path}' if server.nca_path else ''
    print(f'monitor on {url} - watching {server.live_path}{extra} (label {server.label})')

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    Fire(serve)

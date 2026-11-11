// main.ts — roblox-up-and-coming-forecast-01: empirical rank-kernel forecaster for the
// Up-and-Coming chart. The chart ranks games by user growth, so place dynamics are driven by
// each game's growth relative to its neighbours: a game growing faster than the game above it
// tends to swap places with it, and a game published 28 days ago ages out deterministically.
// The kernel is learned from consecutive read pairs (recency-weighted), tilted by growth
// differentials, and averaged over the 72-hour horizon with a light sticky prior.
// Standard library, deterministic, no wall clock.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Snapshot {
  at: string;
  ok: boolean;
  up_and_coming: Array<{ rank: number; universe_id: number; playing: number | null }> | null;
  details: Record<string, { created: string | null; playing: number | null }> | null;
}

interface Task {
  history_until: string;
  window: { from: string; until: string; slots: number };
  places: number;
  output: string;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const PLACES = task.places;

const snaps: Snapshot[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Snapshot)
  .filter((s) => s.ok && s.up_and_coming !== null && s.up_and_coming.length > 0)
  .sort((a, b) => a.at.localeCompare(b.at));

const at = (s: string): number => Date.parse(s);
const HOUR = 3_600_000;
const DAY = 24 * HOUR;
const AGE_OUT_DAYS = 28;

function chartOf(s: Snapshot): number[] {
  const ids: number[] = [];
  for (const g of s.up_and_coming!) {
    if (g.rank >= 1 && g.rank <= PLACES) ids[g.rank - 1] = g.universe_id;
  }
  for (let r = 0; r < PLACES; r++) if (ids[r] === undefined) ids[r] = -1;
  return ids;
}

// Per-hour transition kernel over places 1..PLACES plus an exit state (index PLACES).
const K: number[][] = Array.from({ length: PLACES }, () => new Array<number>(PLACES + 1).fill(0));

for (let i = 1; i < snaps.length; i++) {
  const a = chartOf(snaps[i - 1]!);
  const b = chartOf(snaps[i]!);
  const g = (at(snaps[i]!.at) - at(snaps[i - 1]!.at)) / HOUR;
  if (!(g > 0) || g > 48) continue;
  // Recency: recent pairs describe the current regime.
  const ageH = (at(snaps[snaps.length - 1]!.at) - at(snaps[i]!.at)) / HOUR;
  const recency = Math.exp(-ageH / 96);
  const w = recency / Math.max(g, 1);
  const posB = new Map<number, number>();
  for (let r = 0; r < PLACES; r++) if (b[r] !== -1) posB.set(b[r]!, r + 1);
  for (let r = 0; r < PLACES; r++) {
    if (a[r] === -1) continue;
    const to = posB.get(a[r]!);
    if (to === undefined) K[r]![PLACES]! += w;
    else K[r]![to - 1]! += w;
  }
}

// Shrink each kernel row toward a mobile-chart prior: this chart churns faster than Top
// Trending, so stay mass is lower and neighbour moves larger.
const Ks: number[][] = [];
for (let r = 0; r < PLACES; r++) {
  const n = K[r]!.reduce((x, y) => x + y, 0);
  const c = n / (n + 10);
  const row: number[] = [];
  const priorStay = r === 0 ? 0.55 : 0.45;
  for (let r2 = 0; r2 < PLACES + 1; r2++) {
    const d = r2 - r;
    const prior = r2 === PLACES ? 0.05 : d === 0 ? priorStay : d === -1 ? 0.16 : d === 1 ? 0.16 : d === -2 ? 0.05 : d === 2 ? 0.05 : 0.0;
    const smoothed = (K[r]![r2]! + 0.15 * prior * 10) / (n + 1.5);
    row.push(c * smoothed + (1 - c) * prior);
  }
  const s = row.reduce((x, y) => x + y, 0);
  Ks.push(row.map((v) => v / s));
}

// The last read's chart and per-game growth (playing change per hour over the recent reads).
const last = snaps[snaps.length - 1]!;
const lastChart = chartOf(last);
const playingHist = new Map<number, Array<{ t: number; p: number }>>();
for (const s of snaps) {
  const t = at(s.at);
  for (const g of s.up_and_coming!) {
    if (g.playing === null) continue;
    let h = playingHist.get(g.universe_id);
    if (h === undefined) {
      h = [];
      playingHist.set(g.universe_id, h);
    }
    h.push({ t, p: g.playing });
  }
}
// Growth as the recent log-slope of playing (per hour), robust to gaps.
function growthOf(id: number): number {
  const h = playingHist.get(id);
  if (h === undefined || h.length < 2) return 0;
  const recent = h.slice(-6);
  const t0 = recent[0]!.t;
  const lp = recent.map((x) => Math.log(Math.max(x.p, 1)));
  const n = recent.length;
  const mt = recent.reduce((s, x) => s + (x.t - t0) / HOUR, 0) / n;
  const mp = lp.reduce((s, x) => s + x, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const dt = (recent[i]!.t - t0) / HOUR - mt;
    num += dt * (lp[i]! - mp);
    den += dt * dt;
  }
  if (den <= 1e-9) return 0;
  return num / den;
}

// Aging out: a game's chart life ends 28 days after it was created. Details (with created
// dates) come only on house-recorder reads; the practice history has none, so this only
// bites when the data is there.
const createdOf = new Map<number, number>();
if (last.details) {
  for (const [idStr, d] of Object.entries(last.details)) {
    if (d && d.created) {
      const t = Date.parse(d.created);
      if (!Number.isNaN(t)) createdOf.set(Number(idStr), t);
    }
  }
}
function hoursToAgeOut(id: number, nowT: number): number | null {
  const c = createdOf.get(id);
  if (c === undefined) return null;
  return (c + AGE_OUT_DAYS * DAY - nowT) / HOUR;
}

// Start distributions: point mass at the current place.
type Dist = number[]; // length PLACES+1, last = exit
const dists = new Map<number, Dist>();
for (let r = 0; r < PLACES; r++) {
  if (lastChart[r] === -1) continue;
  const d = new Array<number>(PLACES + 1).fill(0);
  d[r] = 1;
  dists.set(lastChart[r]!, d);
}

// Growth tilt: a game growing faster than the game directly above it tends to swap up. Blend
// its point mass with a point mass one seat up (or down), in proportion to the differential.
// ROBLOX_NOTILT=1 turns the tilt off (the sparse archive's playing snapshots are noisy).
const TILT_OFF = process.env.ROBLOX_NOTILT === '1';
const blended = new Map<number, Dist>();
for (const [id, d] of dists) {
  const myPlace = lastChart.indexOf(id) + 1;
  const myGrowth = growthOf(id);
  const aboveId = myPlace >= 2 ? lastChart[myPlace - 2]! : -1;
  const belowId = myPlace < PLACES ? lastChart[myPlace]! : -1;
  let donorPlace = myPlace;
  if (aboveId !== -1 && myGrowth > growthOf(aboveId) + 0.002) donorPlace = myPlace - 1;
  else if (belowId !== -1 && myGrowth < growthOf(belowId) - 0.002) donorPlace = myPlace + 1;
  const donorId = lastChart[donorPlace - 1]!;
  const donor = new Array<number>(PLACES + 1).fill(0);
  donor[donorPlace - 1] = 1;
  let w = 0;
  if (!TILT_OFF && donorId !== -1 && donorId !== id) {
    const diff = Math.abs(myGrowth - (donorId === aboveId ? growthOf(aboveId) : growthOf(belowId)));
    w = Math.min(0.5, diff * 40);
  }
  blended.set(id, d.map((x, i) => (1 - w) * x + w * donor[i]!));
}

// Evolve the chain over the window's slots, averaging each game's place distribution.
const HORIZON = task.window.slots;
const nowT = at(last.at);
const gameAcc = new Map<number, number[]>();
let state = new Map<number, Dist>();
for (const [id, d] of blended) state.set(id, d.slice());
for (let h = 1; h <= HORIZON; h++) {
  const next = new Map<number, Dist>();
  for (const [id, d] of state) {
    const nd = new Array<number>(PLACES + 1).fill(0);
    for (let r = 0; r < PLACES + 1; r++) {
      const m = d[r]!;
      if (m <= 0) continue;
      if (r === PLACES) nd[PLACES]! += m;
      else for (let r2 = 0; r2 < PLACES + 1; r2++) nd[r2]! += m * Ks[r]![r2]!;
    }
    // Aging out: once the game is past day 28 it leaves for good.
    const left = hoursToAgeOut(id, nowT + h * HOUR);
    if (left !== null && left < 1) {
      nd[PLACES]! += nd[PLACES]! + nd.reduce((s, x, i) => (i < PLACES ? s + x : s), 0) - nd[PLACES]!;
      for (let r = 0; r < PLACES; r++) nd[r] = 0;
    }
    next.set(id, nd);
  }
  state = next;
  for (const [id, d] of state) {
    let a = gameAcc.get(id);
    if (a === undefined) {
      a = new Array<number>(PLACES).fill(0);
      gameAcc.set(id, a);
    }
    for (let r = 0; r < PLACES; r++) a[r]! += d[r]! / HORIZON;
  }
}

// Sticky prior with sharpening: the scored reads arrive hourly, so blend each game's averaged
// distribution back toward its current place and concentrate it on the peaks.
const STICKY = Number(process.env.ROBLOX_STICKY ?? 0.3);
const GAMMA = Number(process.env.ROBLOX_GAMMA ?? 1.7);
for (const [id, a] of gameAcc) {
  const myPlace = lastChart.indexOf(id);
  if (myPlace === -1) continue;
  let s = 0;
  for (let r = 0; r < PLACES; r++) s += Math.pow(Math.max(a[r]!, 0), GAMMA);
  if (s <= 0) continue;
  for (let r = 0; r < PLACES; r++) a[r] = Math.pow(Math.max(a[r]!, 0), GAMMA) / s;
  a[myPlace] = (1 - STICKY) * a[myPlace]! + STICKY;
}

// Rows: per place, the modeled games with the largest probability, plus the unknown-mass tail:
// a share of each place is spread over the REST of the current chart's games, because new
// entrants and deep movers are real and a floored miss costs -9.21 while a named tail row
// costs about ln(u/97).
const UNKNOWN = Number(process.env.ROBLOX_UNKNOWN ?? 0.45);
const rows: Array<{ universe_id: number; place: number; p: number }> = [];
const perPlace: Array<Array<{ id: number; p: number }>> = Array.from({ length: PLACES }, () => []);
for (const [id, a] of gameAcc) {
  for (let r = 0; r < PLACES; r++) {
    if (a[r]! > 0.0001) perPlace[r]!.push({ id, p: a[r]! });
  }
}
for (let r = 0; r < PLACES; r++) {
  perPlace[r]!.sort((a, b) => b.p - a.p || a.id - b.id);
  perPlace[r] = perPlace[r]!.slice(0, 20);
}
// The unknown tail spreads over EVERY game on the current chart (new entrants and deep movers
// can hold any place later; a floored miss costs -9.21 while a named tail row costs ln(u/N)).
const fullChartIds: number[] = [];
for (const g of last.up_and_coming!) {
  if (!fullChartIds.includes(g.universe_id)) fullChartIds.push(g.universe_id);
}
const currentChartIds = fullChartIds;
for (let r = 0; r < PLACES; r++) {
  const s = perPlace[r]!.reduce((x, y) => x + y.p, 0);
  const scale = s > 0 ? (1 - UNKNOWN) / s : 0;
  for (const item of perPlace[r]!) {
    const p = Math.min(0.999, Math.max(0.0002, item.p * scale));
    rows.push({ universe_id: item.id, place: r + 1, p });
  }
  // No tail rows: the scorer's unnamed share (1 - named sum, split over the read's unnamed
  // chart games) already covers new entrants and deep movers. Naming the whole chart would
  // strip that coverage and turn newcomer hits into -9.21 floors.
}

// top10: greedy by place probability, a game used once, filled from the last chart's order.
const top10: number[] = [];
const used = new Set<number>();
for (let r = 0; r < PLACES; r++) {
  let best = -1;
  let bestP = -1;
  for (const item of perPlace[r]!) {
    if (used.has(item.id)) continue;
    if (item.p > bestP) {
      bestP = item.p;
      best = item.id;
    }
  }
  if (best >= 0) {
    top10.push(best);
    used.add(best);
  }
}
for (let r = 0; r < PLACES && top10.length < PLACES; r++) {
  const id = lastChart[r]!;
  if (id !== -1 && !used.has(id)) {
    top10.push(id);
    used.add(id);
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities: rows, top10 })}\n`);

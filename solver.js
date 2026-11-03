/* Smart Academic Timetable Management System - AI engine (CSP + Genetic Algorithm)
   Pure JavaScript, no dependencies. Works in the browser and in Node.
   Time index t = day * periods + period. A domain entry = t * NR + roomIndex. */
(function (root) {
  'use strict';

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  /* ------------------------------------------------------------ CSP model */
  function compile(data) {
    const P = data.periods, D = data.days, T = D * P, L = data.lunchAfter;
    const NR = data.rooms.length, errors = [];
    const fidx = {}, sidx = {};
    data.faculty.forEach((f, i) => (fidx[f.id] = i));
    data.sections.forEach((s, i) => (sidx[s.id] = i));
    const unav = data.faculty.map(f => {
      const u = new Uint8Array(T);
      (f.unavailable || []).forEach(t => { if (t >= 0 && t < T) u[t] = 1; });
      return u;
    });
    const vars = [];
    data.courses.forEach(c => {
      const fi = fidx[c.faculty], si = sidx[c.section];
      if (fi === undefined || si === undefined) { errors.push('Course "' + c.name + '": faculty ya section select nahi hai'); return; }
      const dur = c.block === 2 ? 2 : 1;
      for (let k = 0; k < (c.sessions | 0); k++)
        vars.push({ id: vars.length, cid: c.id, course: c.name, sec: si, fac: fi, dur,
                    type: c.type === 'lab' ? 'lab' : 'class', group: c.group || '', k });
    });
    const maxPerDay = data.maxPerDay !== false;
    const domains = vars.map(v => {
      const d = [];
      for (let t = 0; t < T; t++) {
        const s = t % P;
        if (s + v.dur > P) continue;
        if (v.dur > 1 && s < L && s + v.dur > L) continue;       // a block cannot cross the lunch break
        let free = true;
        for (let x = 0; x < v.dur; x++) if (unav[v.fac][t + x]) free = false;
        if (!free) continue;
        for (let r = 0; r < NR; r++) if (data.rooms[r].type === v.type) d.push(t * NR + r);
      }
      return d;
    });
    domains.forEach((d, i) => { if (!d.length) errors.push('"' + vars[i].course + '" (' + data.sections[vars[i].sec].name + ') ke liye koi valid slot/room nahi hai'); });
    const perSec = {};
    vars.forEach(v => { perSec[v.sec] = (perSec[v.sec] || 0) + v.dur; });
    Object.keys(perSec).forEach(s => { if (perSec[s] > T) errors.push('Section "' + data.sections[s].name + '" ko ' + perSec[s] + ' periods chahiye par week mein sirf ' + T + ' hain'); });
    if (maxPerDay) data.courses.forEach(c => { if ((c.sessions | 0) > D) errors.push('"' + c.name + '": sessions (' + c.sessions + ') days (' + D + ') se zyada hain, "ek din mein ek baar" rule ke saath possible nahi'); });
    return { data, P, D, T, L, NR, vars, domains, unav, maxPerDay, errors };
  }

  const tOf = (p, e) => (e / p.NR) | 0;
  const rOf = (p, e) => e % p.NR;

  function conflictPair(p, va, ea, vb, eb) {
    const A = p.vars[va], B = p.vars[vb], ta = tOf(p, ea), tb = tOf(p, eb);
    if (ta < tb + B.dur && tb < ta + A.dur) {
      if (A.sec === B.sec || A.fac === B.fac || rOf(p, ea) === rOf(p, eb) || (A.group && A.group === B.group)) return true;
    }
    if (p.maxPerDay && A.sec === B.sec && A.cid === B.cid && ((ta / p.P) | 0) === ((tb / p.P) | 0)) return true;
    return false;
  }
  function hasConflict(p, v, e, a) {
    for (let u = 0; u < a.length; u++) if (u !== v && a[u] >= 0 && conflictPair(p, v, e, u, a[u])) return true;
    return false;
  }

  /* ---------------------------------------------------- Search strategies */
  function solve(p, opts) {
    const o = Object.assign({ strategy: 'fch', nodeLimit: 200000, rng: null, fixed: null }, opts || {});
    const n = p.vars.length, a = new Int32Array(n).fill(-1);
    const st = { nodes: 0, backtracks: 0, capped: false };
    const t0 = now();
    if (o.fixed) for (let v = 0; v < n; v++) if (o.fixed[v] >= 0) a[v] = o.fixed[v];
    let ok = false;
    if (o.strategy === 'plain') ok = plain(p, a, o, st); else ok = fc(p, a, o, st, o.strategy === 'fch');
    return { assign: ok ? a : null, nodes: st.nodes, backtracks: st.backtracks, capped: st.capped, ms: now() - t0 };
  }

  function plain(p, a, o, st) {            // chronological backtracking: fixed order, no propagation
    const order = [];
    for (let v = 0; v < a.length; v++) if (a[v] < 0) order.push(v);
    function rec(i) {
      if (i === order.length) return true;
      const v = order[i];
      for (const e of p.domains[v]) {
        if (++st.nodes > o.nodeLimit) { st.capped = true; return false; }
        if (!hasConflict(p, v, e, a)) {
          a[v] = e;
          if (rec(i + 1)) return true;
          a[v] = -1;
          if (st.capped) return false;
        }
      }
      st.backtracks++;
      return false;
    }
    return rec(0);
  }

  function fc(p, a, o, st, heur) {         // backtracking + forward checking (+ MRV, Degree, LCV)
    const n = a.length, NR = p.NR, P = p.P;
    const doms = p.domains.slice(), trail = [], done = new Uint8Array(n);
    let remaining = n;

    function prune(v, e) {
      const A = p.vars[v], t = tOf(p, e), r = rOf(p, e);
      for (let u = 0; u < n; u++) {
        if (done[u] || u === v) continue;
        const B = p.vars[u];
        const same = A.sec === B.sec || A.fac === B.fac || (A.group && A.group === B.group);
        const sameCD = p.maxPerDay && A.sec === B.sec && A.cid === B.cid;
        const d = doms[u], keep = [];
        for (let i = 0; i < d.length; i++) {
          const e2 = d[i], t2 = (e2 / NR) | 0;
          let bad = false;
          if (t2 < t + A.dur && t < t2 + B.dur && (same || e2 % NR === r)) bad = true;
          if (!bad && sameCD && ((t2 / P) | 0) === ((t / P) | 0)) bad = true;
          if (!bad) keep.push(e2);
        }
        if (keep.length !== d.length) { trail.push(u, d); doms[u] = keep; if (!keep.length) return false; }
      }
      return true;
    }
    function undo(mark) { while (trail.length > mark) { const d = trail.pop(); const u = trail.pop(); doms[u] = d; } }

    // pre-assigned (fixed) variables
    for (let v = 0; v < n; v++) if (a[v] >= 0) { done[v] = 1; remaining--; }
    for (let v = 0; v < n; v++) if (a[v] >= 0 && !prune(v, a[v])) return false;

    function pick() {
      let best = -1, bl = Infinity, bd = -1;
      for (let v = 0; v < n; v++) {
        if (done[v]) continue;
        if (!heur) return v;
        const l = doms[v].length;
        if (l > bl) continue;
        let deg = 0; const A = p.vars[v];
        for (let u = 0; u < n; u++) if (!done[u] && u !== v && (p.vars[u].sec === A.sec || p.vars[u].fac === A.fac)) deg++;
        if (l < bl || deg > bd) { best = v; bl = l; bd = deg; }
      }
      return best;
    }
    function order(v) {
      const vals = doms[v].slice();
      if (!heur) return vals;
      const A = p.vars[v], cT = new Int32Array(p.T), cTR = new Int32Array(p.T * NR);
      for (let u = 0; u < n; u++) {
        if (done[u] || u === v) continue;
        const B = p.vars[u], same = B.sec === A.sec || B.fac === A.fac || (A.group && A.group === B.group);
        for (const e2 of doms[u]) {
          const t2 = (e2 / NR) | 0, r2 = e2 % NR;
          for (let x = 0; x < B.dur; x++) { if (same) cT[t2 + x]++; cTR[(t2 + x) * NR + r2]++; }
        }
      }
      if (o.rng) for (let i = vals.length - 1; i > 0; i--) { const j = (o.rng() * (i + 1)) | 0; const tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp; }
      const sc = new Map();
      for (const e of vals) {
        const t = (e / NR) | 0, r = e % NR; let s = 0;
        for (let x = 0; x < A.dur; x++) s += cT[t + x] + cTR[(t + x) * NR + r];
        sc.set(e, s);
      }
      return vals.sort((x, y) => sc.get(x) - sc.get(y));       // Least Constraining Value first
    }
    function rec() {
      if (remaining === 0) return true;
      const v = pick();
      const vals = order(v);
      done[v] = 1; remaining--;
      for (const e of vals) {
        if (++st.nodes > o.nodeLimit) { st.capped = true; break; }
        const mark = trail.length;
        a[v] = e;
        if (prune(v, e) && rec()) return true;
        undo(mark);
        a[v] = -1;
        if (st.capped) break;
      }
      done[v] = 0; remaining++; st.backtracks++;
      return false;
    }
    return rec();
  }

  /* ------------------------------------------------- Rule checker + score */
  function hardViolations(p, a) {
    let c = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] < 0) { c++; continue; }
      for (let j = i + 1; j < a.length; j++) if (a[j] >= 0 && conflictPair(p, i, a[i], j, a[j])) c++;
    }
    return c;
  }
  function hardList(p, a) {            // human-readable clashes (for the checker panel)
    const out = [];
    for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) {
      if (a[i] < 0 || a[j] < 0 || !conflictPair(p, i, a[i], j, a[j])) continue;
      const A = p.vars[i], B = p.vars[j];
      out.push(A.course + ' (' + p.data.sections[A.sec].name + ') vs ' + B.course + ' (' + p.data.sections[B.sec].name + ')');
    }
    return out;
  }
  const DEFAULT_W = { pref: 1, gap: 2, load: 1 };
  function penaltyParts(p, a, w) {
    w = Object.assign({}, DEFAULT_W, w || {});
    const P = p.P, D = p.D, L = p.L, F = p.data.faculty.length, S = p.data.sections.length;
    let pref = 0, gap = 0, load = 0;
    const occ = []; for (let s = 0; s < S; s++) { occ.push([]); for (let d = 0; d < D; d++) occ[s].push(new Uint8Array(P)); }
    const per = []; for (let f = 0; f < F; f++) per.push(new Array(D).fill(0));
    for (let i = 0; i < a.length; i++) {
      if (a[i] < 0) continue;
      const v = p.vars[i], t = tOf(p, a[i]), s0 = t % P, d = (t / P) | 0, fp = p.data.faculty[v.fac].pref;
      if ((fp === 'AM' && s0 >= L) || (fp === 'PM' && s0 < L)) pref++;
      for (let x = 0; x < v.dur; x++) { occ[v.sec][d][s0 + x] = 1; per[v.fac][d]++; }
    }
    for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) {
      let first = -1, last = -1, cnt = 0;
      for (let k = 0; k < P; k++) if (occ[s][d][k]) { if (first < 0) first = k; last = k; cnt++; }
      if (cnt) gap += (last - first + 1) - cnt;
    }
    for (let f = 0; f < F; f++) {
      const m = per[f].reduce((x, y) => x + y, 0) / D;
      load += per[f].reduce((x, y) => x + (y - m) * (y - m), 0);
    }
    return { pref, gap, load, total: Math.round((w.pref * pref + w.gap * gap + w.load * load) * 100) / 100 };
  }
  const penalty = (p, a, w) => penaltyParts(p, a, w).total;

  /* ---------------------------------------------------- Genetic Algorithm */
  async function runGA(p, opts, onProgress) {
    const o = Object.assign({ pop: 30, gens: 150, seed: 1, pc: 0.8, pm: 0.6, tour: 3, elite: 2, weights: null, seedAssign: null }, opts || {});
    const rng = mulberry32(o.seed), S = p.data.sections.length, n = p.vars.length;
    const fit = a => penalty(p, a, o.weights);
    let pop = [];
    if (o.seedAssign) pop.push(Int32Array.from(o.seedAssign));
    let tries = 0;
    while (pop.length < o.pop && tries++ < o.pop * 6) {
      const r = solve(p, { strategy: 'fch', rng, nodeLimit: 60000 });
      if (r.assign) pop.push(r.assign);
    }
    if (!pop.length) return null;
    let scored = pop.map(a => ({ f: fit(a), a })).sort((x, y) => x.f - y.f);
    const initialBest = scored[0].f, initialAvg = scored.reduce((s, x) => s + x.f, 0) / scored.length;
    const history = [initialBest];
    const tourney = () => { let b = null; for (let i = 0; i < o.tour; i++) { const c = scored[(rng() * scored.length) | 0]; if (!b || c.f < b.f) b = c; } return b.a; };

    function mutate(a) {
      const v = (rng() * n) | 0, old = a[v], d = p.domains[v];
      a[v] = -1;
      for (let k = 0; k < 40; k++) { const e = d[(rng() * d.length) | 0]; if (!hasConflict(p, v, e, a)) { a[v] = e; return; } }
      a[v] = old;
    }
    function crossover(A, B) {
      const child = Int32Array.from(A), secs = new Set();
      for (let s = 0; s < S; s++) if (rng() < 0.5) secs.add(s);
      const moved = []; for (let v = 0; v < n; v++) if (secs.has(p.vars[v].sec)) { moved.push(v); child[v] = -1; }
      const pending = [];
      for (const v of moved) { if (!hasConflict(p, v, B[v], child)) child[v] = B[v]; else pending.push(v); }
      for (const v of pending) {                                  // repair step
        const cand = p.domains[v].slice();
        for (let i = cand.length - 1; i > 0; i--) { const j = (rng() * (i + 1)) | 0; const t = cand[i]; cand[i] = cand[j]; cand[j] = t; }
        let placed = false;
        for (const e of cand) if (!hasConflict(p, v, e, child)) { child[v] = e; placed = true; break; }
        if (!placed) return Int32Array.from(A);
      }
      return child;
    }
    for (let g = 1; g <= o.gens; g++) {
      const next = scored.slice(0, o.elite).map(x => x.a);
      while (next.length < o.pop) {
        const pa = tourney(), pb = tourney();
        const ch = rng() < o.pc ? crossover(pa, pb) : Int32Array.from(pa);
        const m = 1 + ((rng() * 3) | 0);
        for (let i = 0; i < m; i++) if (rng() < o.pm) mutate(ch);
        next.push(ch);
      }
      scored = next.map(a => ({ f: fit(a), a })).sort((x, y) => x.f - y.f);
      history.push(scored[0].f);
      if (onProgress) await onProgress(g, o.gens, scored[0].f);
    }
    return { assign: scored[0].a, initialBest, initialAvg, best: scored[0].f, history };
  }

  /* ------------------------------------------ Rescheduling (faculty absent) */
  function reschedule(p0, a0, req) {
    const data = JSON.parse(JSON.stringify(p0.data));
    const f = data.faculty.find(x => x.id === req.faculty);
    const P = data.periods, blocked = [];
    for (let s = req.from; s <= req.to; s++) blocked.push(req.day * P + s);
    f.unavailable = Array.from(new Set((f.unavailable || []).concat(blocked)));
    const p1 = compile(data);
    if (p1.errors.length) return { ok: false, error: p1.errors[0] };
    const fi = p1.data.faculty.indexOf(f) >= 0 ? p1.data.faculty.indexOf(f) : p1.data.faculty.findIndex(x => x.id === req.faculty);
    const affected = [];
    for (let v = 0; v < a0.length; v++) {
      if (p1.vars[v].fac !== fi || a0[v] < 0) continue;
      const t = tOf(p1, a0[v]);
      for (let x = 0; x < p1.vars[v].dur; x++) if (blocked.indexOf(t + x) >= 0) { affected.push(v); break; }
    }
    if (!affected.length) return { ok: true, level: 0, affected: [], moved: [], assign: Int32Array.from(a0), problem: p1, note: 'Us faculty ki us time par koi class nahi thi, kuch badalna nahi pada.' };
    const secs = new Set(affected.map(v => p1.vars[v].sec));
    for (let level = 0; level < 3; level++) {
      const fixed = Int32Array.from(a0);
      for (let v = 0; v < fixed.length; v++) {
        if (affected.indexOf(v) >= 0 || level === 2 || (level === 1 && secs.has(p1.vars[v].sec))) fixed[v] = -1;
      }
      const r = solve(p1, { strategy: 'fch', fixed, rng: mulberry32(7), nodeLimit: 100000 });
      if (r.assign) {
        const moved = [];
        for (let v = 0; v < a0.length; v++) if (r.assign[v] !== a0[v]) moved.push({ v, from: a0[v], to: r.assign[v] });
        return { ok: true, level, affected, moved, assign: r.assign, problem: p1, stats: r };
      }
    }
    return { ok: false, error: 'Valid reschedule nahi mila. Kisi aur slot ya room ki availability badhao.' };
  }

  /* --------------------------------------------------------- Demo datasets */
  function pad(n) { return n; }
  function demoReport(seed, nUnavail) {
    nUnavail = nUnavail || 8;
    const rng = mulberry32((seed | 0) + 1000);
    const P = 6, D = 5, T = P * D;
    const faculty = [];
    for (let f = 0; f < 8; f++) {
      const s = new Set(); while (s.size < nUnavail) s.add((rng() * T) | 0);
      faculty.push({ id: 'F' + (f + 1), name: 'Faculty ' + (f + 1), pref: rng() < 0.5 ? 'AM' : 'PM', unavailable: Array.from(s).sort((x, y) => x - y) });
    }
    const rooms = [];
    for (let i = 1; i <= 4; i++) rooms.push({ id: 'C' + i, name: 'Room C' + i, type: 'class' });
    for (let i = 1; i <= 2; i++) rooms.push({ id: 'L' + i, name: 'Lab L' + i, type: 'lab' });
    const sections = ['A', 'B', 'C', 'D'].map(x => ({ id: 'S' + x, name: 'CSE-' + x }));
    const names = ['AI', 'DSA', 'DBMS', 'OS', 'CN', 'MATH', 'DSA-Lab', 'DBMS-Lab'];
    const courses = [];
    sections.forEach((sec, si) => names.forEach((nm, ci) => {
      const lab = ci >= 6;
      courses.push({ id: sec.id + '-' + nm, section: sec.id, name: nm, type: lab ? 'lab' : 'theory', sessions: lab ? 2 : 3, block: 1, faculty: 'F' + (((ci + si) % 8) + 1), group: '' });
    }));
    return { name: 'Report dataset (88 sessions)', days: 5, periods: 6, lunchAfter: 3, maxPerDay: true, faculty, rooms, sections, courses };
  }
  function demoCollege() {
    const faculty = [
      { id: 'F1', name: 'Dr. Sharma', pref: 'AM', unavailable: [] },
      { id: 'F2', name: 'Prof. Verma', pref: 'AM', unavailable: [24, 25, 26] },
      { id: 'F3', name: 'Dr. Gupta', pref: 'PM', unavailable: [0, 1, 2] },
      { id: 'F4', name: 'Prof. Singh', pref: 'ANY', unavailable: [] },
      { id: 'F5', name: 'Dr. Khan', pref: 'AM', unavailable: [12, 13, 14, 15, 16, 17] },
      { id: 'F6', name: 'Prof. Mishra', pref: 'PM', unavailable: [] },
      { id: 'F7', name: 'Dr. Rao', pref: 'ANY', unavailable: [] }
    ];
    const rooms = [
      { id: 'R101', name: 'Room 101', type: 'class' }, { id: 'R102', name: 'Room 102', type: 'class' },
      { id: 'R103', name: 'Room 103', type: 'class' }, { id: 'LAB1', name: 'Lab 1', type: 'lab' }, { id: 'LAB2', name: 'Lab 2', type: 'lab' }
    ];
    const sections = [{ id: 'A', name: 'CSE-A' }, { id: 'B', name: 'CSE-B' }, { id: 'C', name: 'CSE-C' }];
    const theory = ['AI', 'DSA', 'DBMS', 'OS'];
    const courses = [];
    sections.forEach((sec, si) => {
      theory.forEach((nm, ci) => courses.push({ id: sec.id + nm, section: sec.id, name: nm, type: 'theory', sessions: 3, block: 1, faculty: 'F' + (((ci + si) % 6) + 1), group: '' }));
      courses.push({ id: sec.id + 'DSAL', section: sec.id, name: 'DSA-Lab', type: 'lab', sessions: 1, block: 2, faculty: 'F' + (((si + 4) % 6) + 1), group: '' });
      courses.push({ id: sec.id + 'AIL', section: sec.id, name: 'AI-Lab', type: 'lab', sessions: 1, block: 2, faculty: 'F' + (((si + 5) % 6) + 1), group: '' });
    });
    courses.push({ id: 'ELA', section: 'A', name: 'Cloud Computing (Elective)', type: 'theory', sessions: 2, block: 1, faculty: 'F7', group: 'EL1' });
    courses.push({ id: 'ELB', section: 'B', name: 'Cyber Security (Elective)', type: 'theory', sessions: 2, block: 1, faculty: 'F7', group: 'EL1' });
    courses.push({ id: 'ELC', section: 'C', name: 'Data Mining (Elective)', type: 'theory', sessions: 2, block: 1, faculty: 'F6', group: 'EL1' });
    return { name: 'Mini college (2-period labs + electives)', days: 5, periods: 6, lunchAfter: 3, maxPerDay: true, faculty, rooms, sections, courses };
  }
  function emptyData() {
    return { name: 'My college', days: 5, periods: 6, lunchAfter: 3, maxPerDay: true, faculty: [], rooms: [], sections: [], courses: [] };
  }

  /* ------------------------------------------------------------- Benchmark */
  async function benchmark(nInst, nGA, onProgress, nUnavail) {
    const rows = [], strategies = [['plain', 'Plain backtracking'], ['fc', 'Backtracking + Forward Checking'], ['fch', 'FC + MRV + Degree + LCV']];
    for (let s = 0; s < nInst; s++) {
      const p = compile(demoReport(s, nUnavail)), r = { seed: s, p };
      for (const [k] of strategies) r[k] = solve(p, { strategy: k });
      rows.push(r);
      if (onProgress) await onProgress('search', s + 1, nInst);
    }
    const summary = strategies.map(([k, label]) => {
      const solved = rows.filter(r => r[k].assign).length;
      const avg = f => rows.reduce((x, r) => x + f(r[k]), 0) / rows.length;
      return { key: k, label, solved, total: rows.length, nodes: avg(x => x.nodes), backtracks: avg(x => x.backtracks), ms: avg(x => x.ms) };
    });
    const ga = [];
    for (let s = 0; s < nGA; s++) {
      const r = rows[s], first = r.fch.assign;
      const base = penalty(r.p, first);
      const g = await runGA(r.p, { seed: s + 1 });
      ga.push({ seed: s, base, final: g.best, hard: hardViolations(r.p, g.assign), gain: 100 * (base - g.best) / base });
      if (onProgress) await onProgress('ga', s + 1, nGA);
    }
    return { rows, summary, ga };
  }

  const api = { mulberry32, compile, solve, hardViolations, hardList, penalty, penaltyParts, runGA, reschedule, demoReport, demoCollege, emptyData, benchmark, tOf, rOf, DEFAULT_W };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.TT = api;
})(typeof window !== 'undefined' ? window : globalThis);

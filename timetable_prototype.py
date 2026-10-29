"""
Smart Academic Timetable Management System - AI PBL (CCSAI0301)
Prototype for Review 2: CSP formulation + search strategies + Genetic Algorithm.

Run:  python timetable_prototype.py
"""
import random
import time
import statistics

DAYS, SLOTS = 5, 6                      # Mon-Fri, 6 teaching periods/day (lunch break fixed, not a slot)
T = DAYS * SLOTS
CLASSROOMS = ["C1", "C2", "C3", "C4"]
LABS = ["L1", "L2"]
THEORY = ["AI", "DSA", "DBMS", "OS", "CN", "MATH"]      # 3 sessions / week
LABCRS = ["DSA-Lab", "DBMS-Lab"]                          # 2 sessions / week
N_SECTIONS, N_FAC = 4, 8
NODE_LIMIT = 200_000


def day(t):
    return t // SLOTS


def build_instance(seed):
    """Return (variables, domains, faculty_unavailable, faculty_pref)."""
    rng = random.Random(seed)
    courses = THEORY + LABCRS
    variables = []   # (section, course, k, faculty, is_lab)
    for s in range(N_SECTIONS):
        for ci, c in enumerate(courses):
            fac = (ci + s) % N_FAC
            is_lab = c in LABCRS
            for k in range(2 if is_lab else 3):
                variables.append((s, c, k, fac, is_lab))
    unavail = {f: set(rng.sample(range(T), 8)) for f in range(N_FAC)}      # hard: faculty not available
    pref = {f: rng.choice(["AM", "PM"]) for f in range(N_FAC)}              # soft: preferred half of day
    domains = []
    for (s, c, k, f, is_lab) in variables:
        rooms = LABS if is_lab else CLASSROOMS
        domains.append({(t, r) for t in range(T) if t not in unavail[f] for r in rooms})
    return variables, domains, unavail, pref


# ---------------------------------------------------------------- constraints
def conflicts(vi, val, variables, assign):
    """True if giving variable vi the value `val` breaks a hard constraint w.r.t. `assign`."""
    s, c, k, f, _ = variables[vi]
    t, r = val
    for vj, (t2, r2) in assign.items():
        if vj == vi:
            continue
        s2, c2, _, f2, _ = variables[vj]
        if t == t2 and (s == s2 or f == f2 or r == r2):
            return True
        if s == s2 and c == c2 and day(t) == day(t2):      # same course twice a day for a section
            return True
    return False


# ---------------------------------------------------------------- solvers
def solve_plain(variables, domains):
    """Chronological backtracking: fixed variable order, fixed value order, no propagation."""
    n = len(variables)
    assign, stats = {}, {"nodes": 0, "backtracks": 0}
    vals = [sorted(d) for d in domains]

    def rec(i):
        if i == n:
            return True
        for val in vals[i]:
            stats["nodes"] += 1
            if stats["nodes"] > NODE_LIMIT:
                return False
            if not conflicts(i, val, variables, assign):
                assign[i] = val
                if rec(i + 1):
                    return True
                del assign[i]
        stats["backtracks"] += 1
        return False

    ok = rec(0) and len(assign) == n
    return (dict(assign) if ok else None), stats


def prune(vi, val, variables, doms, unassigned):
    """Forward checking. Returns list of (var, removed_values) or None on wipe-out."""
    s, c, _, f, _ = variables[vi]
    t, r = val
    removed = []
    for vj in unassigned:
        s2, c2, _, f2, _ = variables[vj]
        bad = set()
        for (t2, r2) in doms[vj]:
            if t2 == t and (s == s2 or f == f2 or r == r2):
                bad.add((t2, r2))
            elif s == s2 and c == c2 and day(t2) == day(t):
                bad.add((t2, r2))
        if bad:
            doms[vj] -= bad
            removed.append((vj, bad))
            if not doms[vj]:
                return removed, False
    return removed, True


def solve_fc(variables, domains, heuristics, rng=None):
    """Backtracking + forward checking. heuristics=True adds MRV, Degree, LCV."""
    n = len(variables)
    doms = [set(d) for d in domains]
    assign, stats = {}, {"nodes": 0, "backtracks": 0}
    unassigned = set(range(n))

    def pick():
        if not heuristics:
            return min(unassigned)
        # MRV, tie-break by Degree (most constraints with other unassigned vars)
        def deg(v):
            s, _, _, f, _ = variables[v]
            return sum(1 for u in unassigned if u != v and (variables[u][0] == s or variables[u][3] == f))
        return min(unassigned, key=lambda v: (len(doms[v]), -deg(v)))

    def order(v):
        vals = sorted(doms[v])
        if not heuristics:
            return vals
        s, _, _, f, _ = variables[v]
        cnt_t, cnt_tr = {}, {}
        for u in unassigned:
            if u == v:
                continue
            same = variables[u][0] == s or variables[u][3] == f
            for (t2, r2) in doms[u]:
                if same:
                    cnt_t[t2] = cnt_t.get(t2, 0) + 1
                cnt_tr[(t2, r2)] = cnt_tr.get((t2, r2), 0) + 1
        # Least Constraining Value: fewest options removed from neighbours
        key = lambda val: cnt_t.get(val[0], 0) + cnt_tr.get(val, 0)
        if rng is not None:
            rng.shuffle(vals)
        return sorted(vals, key=key)

    def rec():
        if not unassigned:
            return True
        v = pick()
        unassigned.remove(v)
        for val in order(v):
            stats["nodes"] += 1
            if stats["nodes"] > NODE_LIMIT:
                unassigned.add(v)
                return False
            removed, ok = prune(v, val, variables, doms, unassigned)
            if ok:
                assign[v] = val
                if rec():
                    return True
                del assign[v]
            for vj, bad in removed:
                doms[vj] |= bad
        stats["backtracks"] += 1
        unassigned.add(v)
        return False

    ok = rec()
    return (dict(assign) if ok else None), stats


# ---------------------------------------------------------------- checks + fitness
def hard_violations(tt, variables):
    cnt = 0
    items = list(tt.items())
    for a in range(len(items)):
        va, (ta, ra) = items[a]
        sa, ca, _, fa, _ = variables[va]
        for b in range(a + 1, len(items)):
            vb, (tb, rb) = items[b]
            sb, cb, _, fb, _ = variables[vb]
            if ta == tb and (sa == sb or fa == fb or ra == rb):
                cnt += 1
            if sa == sb and ca == cb and day(ta) == day(tb):
                cnt += 1
    return cnt


def penalty(tt, variables, unavail, pref):
    """Soft-constraint cost (lower is better)."""
    p = 0
    # 1) faculty time preference
    for v, (t, _) in tt.items():
        f = variables[v][3]
        slot = t % SLOTS
        if (pref[f] == "AM") != (slot < SLOTS // 2):
            p += 1
    # 2) student idle gaps (empty periods between first and last class of a day)
    for s in range(N_SECTIONS):
        for d in range(DAYS):
            ts = sorted(t % SLOTS for v, (t, _) in tt.items() if variables[v][0] == s and day(t) == d)
            if ts:
                p += 2 * ((ts[-1] - ts[0] + 1) - len(ts))
    # 3) faculty workload balance across days
    for f in range(N_FAC):
        per = [0] * DAYS
        for v, (t, _) in tt.items():
            if variables[v][3] == f:
                per[day(t)] += 1
        m = sum(per) / DAYS
        p += sum((x - m) ** 2 for x in per)
    return round(p, 2)


# ---------------------------------------------------------------- Genetic Algorithm
def mutate(tt, variables, rng, tries=40):
    v = rng.randrange(len(variables))
    old = tt[v]
    del tt[v]
    cand = sorted(DOMS_CACHE[v])
    for _ in range(tries):
        val = rng.choice(cand)
        if not conflicts(v, val, variables, tt):
            tt[v] = val
            return
    tt[v] = old


def crossover(a, b, variables, rng):
    child = dict(a)
    secs = [s for s in range(N_SECTIONS) if rng.random() < 0.5]
    moved = [v for v in b if variables[v][0] in secs]
    for v in moved:
        del child[v]
    pending = []
    for v in moved:                       # copy from parent B when no clash, else repair later
        if not conflicts(v, b[v], variables, child):
            child[v] = b[v]
        else:
            pending.append(v)
    for v in pending:                     # repair: random feasible re-assignment
        cand = sorted(DOMS_CACHE[v])
        rng.shuffle(cand)
        for val in cand:
            if not conflicts(v, val, variables, child):
                child[v] = val
                break
        else:
            return dict(a)                # repair failed -> keep parent A
    return child


def run_ga(variables, domains, unavail, pref, seed, pop_size=30, gens=150):
    global DOMS_CACHE
    DOMS_CACHE = domains
    rng = random.Random(seed)
    pop = []
    while len(pop) < pop_size:            # feasible population from randomised CSP solver
        sol, _ = solve_fc(variables, domains, True, rng)
        if sol:
            pop.append(sol)
    fit = lambda tt: penalty(tt, variables, unavail, pref)
    scored = sorted(((fit(t), t) for t in pop), key=lambda x: x[0])
    initial_best, initial_avg = scored[0][0], statistics.mean(s for s, _ in scored)
    for _ in range(gens):
        new = [dict(t) for _, t in scored[:2]]                       # elitism
        while len(new) < pop_size:
            pa = min(rng.sample(scored, 3), key=lambda x: x[0])[1]   # tournament selection
            pb = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            ch = crossover(pa, pb, variables, rng) if rng.random() < 0.8 else dict(pa)
            for _ in range(rng.choice([1, 2, 3])):
                if rng.random() < 0.6:
                    mutate(ch, variables, rng)
            new.append(ch)
        scored = sorted(((fit(t), t) for t in new), key=lambda x: x[0])
    return initial_best, initial_avg, scored[0][0], scored[0][1]


# ---------------------------------------------------------------- experiments
def main(n_instances=10):
    rows = []
    for seed in range(n_instances):
        variables, domains, unavail, pref = build_instance(seed)
        r = {"seed": seed}
        for name, fn in [
            ("plain", lambda: solve_plain(variables, domains)),
            ("fc", lambda: solve_fc(variables, domains, False)),
            ("fc+h", lambda: solve_fc(variables, domains, True)),
        ]:
            t0 = time.time()
            sol, st = fn()
            r[name] = dict(ok=sol is not None, nodes=st["nodes"], bt=st["backtracks"],
                           sec=round(time.time() - t0, 2), sol=sol)
        rows.append((variables, domains, unavail, pref, r))
        print(seed, {k: (v["ok"], v["nodes"], v["bt"], v["sec"]) for k, v in r.items() if k != "seed"})

    print("\n=== Search comparison over", n_instances, "instances ===")
    for name in ["plain", "fc", "fc+h"]:
        okc = sum(1 for *_, r in rows if r[name]["ok"])
        nodes = [r[name]["nodes"] for *_, r in rows]
        bts = [r[name]["bt"] for *_, r in rows]
        secs = [r[name]["sec"] for *_, r in rows]
        print(f"{name:6s} solved {okc}/{n_instances}  avg nodes {statistics.mean(nodes):.0f}  "
              f"avg backtracks {statistics.mean(bts):.0f}  avg time {statistics.mean(secs):.2f}s")

    print("\n=== GA optimisation (pop 30, 150 generations) ===")
    gains, hv = [], 0
    for variables, domains, unavail, pref, r in rows[:5]:
        seed = r["seed"]
        base = penalty(r["fc+h"]["sol"], variables, unavail, pref)
        ib, ia, fb, best = run_ga(variables, domains, unavail, pref, seed)
        v = hard_violations(best, variables)
        hv += v
        gains.append(100 * (base - fb) / base)
        print(f"seed {seed}: first-solution penalty {base}, pop-best {ib}, pop-avg {ia:.1f}, "
              f"GA best {fb}, hard violations {v}, improvement vs first solution {gains[-1]:.1f}%")
    print(f"mean improvement {statistics.mean(gains):.1f}%  total hard violations {hv}")


if __name__ == "__main__":
    main()

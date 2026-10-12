/**
 * TraceSource — the single trace data model the renderer sees.
 *
 * Before this module, viz.js owned `ranksEvents: Array<Array<{start,end,fn}>>` and ran nine
 * whole-trace precompute passes that turned percentiles into per-event `THREE.Color` tables
 * before the first frame could be drawn. That is what made a large trace unviewable: the work
 * is O(intervals) and eager, so nothing renders until all of it finishes, and the colour
 * tables are several times the size of the trace itself.
 *
 * A TraceSource replaces both. It exposes:
 *
 *   - a *columnar* view of the trace: per-interval `starts`/`ends`/`regions`, plus a fixed set
 *     of named derived columns (percentiles, iteration index, comp prefix) that are pure
 *     functions of the trace and therefore independent of any colour scheme;
 *   - a *cursor*: `sampleAll(t, out)` resolves, for every rank at once, which interval brackets
 *     time `t` — with zero allocation per frame.
 *
 * Colours are then a table lookup at render time instead of a table build at load time, and
 * the three colour schemes of Dual View share one set of columns instead of building their own.
 *
 * Two implementations produce the same named columns:
 *
 *   - `TextSource` (here)  — the legacy `.txt` format, parsed and derived in the browser.
 *   - `Pv3dSource`         — `.pv3dbin`, where the converter already stored every column and
 *                            the browser fetches one block at a time.
 *
 * Identical column names on both sides make them directly diffable, which is the strongest
 * verification available for the new format.
 *
 * ---------------------------------------------------------------------------
 * Addressing: (chunk, row)
 * ---------------------------------------------------------------------------
 * An interval is addressed by a pair. `TextSource` uses `chunk = rank index` and `row = index
 * within that rank`; `Pv3dSource` will use `chunk = block index` and `row = row within that
 * block`. The renderer never interprets either number — it gets them from `sampleAll()` and
 * hands them straight back to `cell()`. That is what lets one renderer drive both formats.
 */

// ---------------------------------------------------------------------------
// Column names
// ---------------------------------------------------------------------------
// Every derived column is computed on the *whole* interval sequence of a rank, so index `row`
// means the same interval whatever detail level is in use. Each replaces one of viz.js's old
// build*Colors() passes; the mapping is noted per entry.
export const COL = {
  /** Region (function) id, index into `regionNames`. Replaces `event.fn`. */
  REGION: 'regions',
  /** MPI_Wait/MPI_Waitall count before this interval on this rank. (buildIterationIndices) */
  ITER: 'iter_index',
  /** Percentile (0-100) of this interval's start among the same column across ranks. (buildHeatmapColors) */
  PCT_ENTER: 'pct_enter',
  /** Percentile of this interval's end among the same column across ranks. (buildHeatmapColors) */
  PCT_LEAVE: 'pct_leave',
  /** Percentile of this interval's duration among the same column. (buildHeatmapDurationColors) */
  PCT_DUR_COL: 'pct_dur_col',
  /** Percentile of this interval's duration among every interval of the same region, run-wide.
   *  (buildSpeedHeatmapRunColors) */
  PCT_DUR_REGION: 'pct_dur_region',
  /** The four above again, for the Comp gap *following* this interval. Defined for every
   *  interval except a rank's last. (buildCompHeatmapColors / buildCompHeatmapDurationColors) */
  GAP_PCT_ENTER: 'gap_pct_enter',
  GAP_PCT_LEAVE: 'gap_pct_leave',
  GAP_PCT_DUR_COL: 'gap_pct_dur_col',
  /** Percentile of the following gap's duration among *every* gap in the run — one flat pool,
   *  not per column. (buildCompHeatmapRunColors) */
  GAP_PCT_DUR_FLAT: 'gap_pct_dur_flat',
  /** Cumulative comp (non-MPI) time before this interval's start. (buildSpeedHeatmapPrefixes) */
  COMP_PREFIX: 'comp_prefix',
};

/** Enter/leave column pair for each `scheme.heatmapAppliesTo` value. */
export function heatmapColumns(appliesTo) {
  switch (appliesTo) {
    case 'comp': return [COL.GAP_PCT_ENTER, COL.GAP_PCT_LEAVE];
    case 'mpiDuration': return [COL.PCT_DUR_COL, COL.PCT_DUR_COL];
    case 'compDuration': return [COL.GAP_PCT_DUR_COL, COL.GAP_PCT_DUR_COL];
    case 'compRun': return [COL.GAP_PCT_DUR_FLAT, COL.GAP_PCT_DUR_FLAT];
    default: return [COL.PCT_ENTER, COL.PCT_LEAVE];
  }
}

/** True for the Comp-family heatmaps, whose enter/leave columns are indexed by *gap*. */
export function isGapHeatmap(appliesTo) {
  return appliesTo === 'comp' || appliesTo === 'compDuration' || appliesTo === 'compRun';
}

// ---------------------------------------------------------------------------
// Per-frame sample buffer
// ---------------------------------------------------------------------------

/**
 * Allocates the struct-of-arrays that `sampleAll()` fills. One per (scheme-independent) cursor,
 * reused for the life of a visualization so a frame allocates nothing.
 *
 * Per rank `i`, after `sampleAll(t, out)`:
 *   curRow[i] >= 0   this rank has an interval bracketing or ahead of `t` (the old `pointer`);
 *                    -1 means `t` is past this rank's last interval.
 *   prevRow[i] >= 0  the interval before that one exists (the old `pointer > 0`).
 *   inside[i]        1 when `t` falls inside [curStart, curEnd].
 *   curIsLast[i]     1 when the current interval is this rank's last, so the gap that follows
 *                    it does not exist (replaces the old `pointer < len - 1` tests).
 *   curStart/curEnd  extents of the current interval, valid when curRow >= 0.
 *   prevEnd          end of the previous interval, valid when prevRow >= 0.
 */
export function makeSample(n) {
  return {
    n,
    curChunk: new Int32Array(n),
    curRow: new Int32Array(n),
    prevChunk: new Int32Array(n),
    prevRow: new Int32Array(n),
    curStart: new Float64Array(n),
    curEnd: new Float64Array(n),
    prevEnd: new Float64Array(n),
    inside: new Uint8Array(n),
    curIsLast: new Uint8Array(n),
  };
}

/** First index in [0, n) whose end is >= t; n when there is none.
 *
 *  This is exactly the fixed point of viz.js's old two-pointer bracket ("advance while
 *  t > ends[p]; retreat while t < ends[p-1]"), reached in ~log2(n) comparisons from any
 *  starting position — which is why the incremental `eventPointers` cursor, and the three
 *  invalidation paths a seek/level-switch/block-eviction would need, are gone. */
export function bracketIndex(ends, n, t) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ends[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Parses a decimal-seconds field into whole microseconds, exactly.
 *
 * `parseFloat(s) * 1e6` is not exact — 0.1 is not representable in binary, so the product can
 * land a ULP off and the trace's own six decimals stop being integers. Reading the digits
 * directly keeps every timestamp an exact integer, which is the point of working in ticks at
 * all. Falls back to the lenient float path for anything unusual (exponent form, trailing
 * junk), matching what parseFloat() used to accept.
 */
export function secondsTextToTicks(s) {
  const n = s.length;
  let i = 0, neg = false;
  if (n > 0 && (s.charCodeAt(0) === 43 || s.charCodeAt(0) === 45)) { neg = s.charCodeAt(0) === 45; i = 1; }
  let intPart = 0, seenDigit = false;
  for (; i < n; i++) {
    const c = s.charCodeAt(i);
    if (c < 48 || c > 57) break;
    intPart = intPart * 10 + (c - 48);
    seenDigit = true;
  }
  let frac = 0;
  if (i < n && s.charCodeAt(i) === 46) {
    i++;
    for (let d = 0; d < 6; d++) {
      const c = i < n ? s.charCodeAt(i) : -1;
      if (c >= 48 && c <= 57) { frac = frac * 10 + (c - 48); i++; seenDigit = true; }
      else frac = frac * 10;                       // fewer than six decimals: pad
    }
    const c = i < n ? s.charCodeAt(i) : -1;        // seventh decimal: round, then discard
    if (c >= 48 && c <= 57) {
      if (c - 48 >= 5) frac++;
      while (i < n && s.charCodeAt(i) >= 48 && s.charCodeAt(i) <= 57) i++;
    }
  }
  if (!seenDigit || i !== n) {
    const f = parseFloat(s);
    return Number.isNaN(f) ? NaN : Math.round(f * 1e6);
  }
  const v = intPart * 1000000 + frac;
  return neg ? -v : v;
}

/** First index in [0, n) whose start is > t; n when there is none. */
function firstStartAfter(starts, n, t) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Percentile helpers
// ---------------------------------------------------------------------------
// Deliberately identical, expression for expression, to viz.js's computeColumnPercentiles()
// and computeFlatPercentiles() that they replace: same stable sort, same `(i / (n-1)) * 100`,
// same "a lone entry is 100". Any drift here shows up as a changed pixel.

/**
 * Ranks `values[0..n)` against each other and writes percentiles into `out`.
 * Order of equal values follows input order (Array.prototype.sort is stable), matching the
 * ascending-rank enumeration the old builders used.
 */
export function rankPercentiles(values, n, out) {
  if (n === 0) return;
  if (n === 1) { out[0] = 100; return; }
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => values[a] - values[b]);
  for (let pos = 0; pos < n; pos++) out[order[pos]] = (pos / (n - 1)) * 100;
}

// ---------------------------------------------------------------------------
// TraceSource — the interface
// ---------------------------------------------------------------------------

/**
 * Shape of every trace loader. Documented here rather than enforced; JS has no interfaces and
 * the methods are called on the hot path, so `TextSource` and `Pv3dSource` each define them
 * directly instead of inheriting stubs that would deoptimize the call sites.
 *
 *   nRanks            number of ranks (locations) — one rendered box each
 *   ticksPerSecond    integer ticks per second used by starts/ends (1 = seconds, step 0)
 *   t0                absolute time of timeline zero, for the wall-clock display
 *   t1                timeline span (the old `maxTime`)
 *   regionNames       string[] — region (function) names, indexed by region id
 *   regionId(name)    -> id, or -1
 *   rowCount(rank)
 *   ensureColumns(names) -> Promise, makes those columns readable
 *   hasColumn(name)
 *   cell(name, chunk, row) -> number, already scaled (percentiles come out as 0-100)
 *   ensure(t) -> Promise / ready(t) -> bool   (always true for TextSource: it is all in memory)
 *   sampleAll(t, out)
 *   compTimeAt(rank, t)  -> cumulative comp time at an arbitrary t
 *   runIndex(regionId)   -> Array<Int32Array> of rows matching that region, per rank
 *   sampleRun(index, t, out)
 *   nextEventAfter(t)    -> smallest interval start strictly after t, or Infinity
 *   overview()           -> columnar per-rank arrays for the timeline HUD
 */
export class TraceSource {}

// ---------------------------------------------------------------------------
// TextSource — the legacy `.txt` loader
// ---------------------------------------------------------------------------

/**
 * Parses the `# Rank` / `function;starttime;endtime` text format into the columnar model, and
 * computes the derived columns itself (lazily, only the ones an active scheme asks for).
 *
 * This reproduces viz.js's parseTraceData() exactly, including its quirks — a `# Rank` header
 * pushes the previous rank even when that rank had no parsable rows, while the final rank is
 * pushed only if non-empty — because the rank count drives `instanceCount` and therefore the
 * grid layout.
 *
 * Times are integer *ticks*, here one microsecond — the exact resolution the text format
 * writes (`%.6f` seconds), so every timestamp is a whole number and stays one through the
 * shift to timeline zero. `Pv3dSource` uses the trace's own timer resolution instead, making
 * a tick one cycle there; either way the renderer works in integers and converts to seconds
 * only where a clock is displayed.
 */
export class TextSource {
  constructor() {
    this.nRanks = 0;
    this.ticksPerSecond = 1e6;   // one tick = 1 µs
    this.t0 = 0;
    this.t1 = 0;
    this.regionNames = [];
    this._regionIds = new Map();
    this._starts = [];         // Float64Array per rank
    this._ends = [];           // Float64Array per rank
    this._regions = [];        // Uint16Array per rank
    this._counts = null;       // Int32Array
    this._cols = new Map();    // name -> Array<TypedArray> per rank
    this._runIndexCache = new Map();
    //: The legacy text format is `function;starttime;endtime` — it records no hardware
    //: counters at all, so every PAPI query has no answer rather than a zero one.
    this.papiCounters = [];
    this.format = 'legacy-txt';
    this.levelInfo = null;     // no detail levels in the text format
  }

  /**
   * @param {string} rawData    file contents
   * @param {number} timeskip   "Begin time" — seconds to skip past the trace's own start
   * @param {boolean} ignoreMpiInit  drop MPI_Init intervals before anything else
   * @returns {TextSource|null} null when the file contains no `# Rank` block at all
   */
  static parse(rawData, timeskip = 0, ignoreMpiInit = false) {
    const src = new TextSource();

    // --- pass 1: split into ranks, keeping absolute times ---------------------------------
    const absStarts = [], absEnds = [], absRegions = [];
    let minStartTicks = Infinity;
    let curS = null, curE = null, curR = null;
    let isParsingRank = false;

    const lines = rawData.trim().split('\n');
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li].trim();
      if (line.startsWith('# Rank')) {
        if (curS) { absStarts.push(curS); absEnds.push(curE); absRegions.push(curR); }
        curS = []; curE = []; curR = [];
        isParsingRank = true;
        continue;
      }
      if (!isParsingRank) continue;
      if (line.toLowerCase() === 'function;starttime;endtime') continue;

      // Hand-rolled split: `String.split(';')` allocates an array per line, which at a few
      // million lines dominates the parse.
      const a = line.indexOf(';');
      if (a < 0) continue;
      const b = line.indexOf(';', a + 1);
      if (b < 0) continue;
      if (line.indexOf(';', b + 1) >= 0) continue;   // more than 3 parts — old code skipped it

      const fn = line.slice(0, a);
      const startTime = secondsTextToTicks(line.slice(a + 1, b));
      const endTime = secondsTextToTicks(line.slice(b + 1));
      if (isNaN(startTime) || isNaN(endTime) || !curS) continue;
      if (ignoreMpiInit && fn === 'MPI_Init') continue;

      curS.push(startTime);
      curE.push(endTime);
      curR.push(src._internRegion(fn));
      if (startTime < minStartTicks) minStartTicks = startTime;
    }
    if (curS && curS.length > 0) { absStarts.push(curS); absEnds.push(curE); absRegions.push(curR); }

    if (absStarts.length === 0) return null;
    if (minStartTicks === Infinity) minStartTicks = 0;

    // --- pass 2: shift to timeline zero and drop what falls before it ---------------------
    const timelineStartTicks = minStartTicks + Math.round(timeskip * 1e6);
    let maxTime = 0;
    const n = absStarts.length;
    src.nRanks = n;
    src._counts = new Int32Array(n);

    for (let r = 0; r < n; r++) {
      const aS = absStarts[r], aE = absEnds[r], aR = absRegions[r];
      let kept = 0;
      for (let p = 0; p < aS.length; p++) if (aE[p] > timelineStartTicks) kept++;

      const S = new Float64Array(kept), E = new Float64Array(kept), R = new Uint16Array(kept);
      let w = 0;
      for (let p = 0; p < aS.length; p++) {
        if (aE[p] <= timelineStartTicks) continue;
        const newEnd = aE[p] - timelineStartTicks;
        S[w] = Math.max(0, aS[p] - timelineStartTicks);
        E[w] = newEnd;
        R[w] = aR[p];
        w++;
        if (newEnd > maxTime) maxTime = newEnd;
      }
      src._starts.push(S); src._ends.push(E); src._regions.push(R);
      src._counts[r] = kept;
    }

    src.t0 = timelineStartTicks;
    src.t1 = maxTime;
    src._cols.set(COL.REGION, src._regions);
    return src;
  }

  _internRegion(name) {
    let id = this._regionIds.get(name);
    if (id === undefined) {
      id = this.regionNames.length;
      this.regionNames.push(name);
      this._regionIds.set(name, id);
    }
    return id;
  }

  regionId(name) {
    const id = this._regionIds.get(name);
    return id === undefined ? -1 : id;
  }

  papiColumnFor() { return null; }
  papiIsConstant() { return false; }
  papiDelta() { return NaN; }

  rowCount(rank) { return this._counts[rank]; }

  // --- columns ---------------------------------------------------------------------------

  hasColumn(name) { return this._cols.has(name); }

  /** Everything is already in memory, so this only has to materialize what was asked for. */
  async ensureColumns(names) {
    for (const name of names) if (!this._cols.has(name)) this._buildColumn(name);
  }

  cell(name, chunk, row) {
    const col = this._cols.get(name);
    return col ? col[chunk][row] : 0;
  }

  _buildColumn(name) {
    switch (name) {
      case COL.ITER: return this._buildIterIndex();
      case COL.PCT_ENTER:
      case COL.PCT_LEAVE:
      case COL.PCT_DUR_COL: return this._buildIntervalColumnPercentiles();
      case COL.GAP_PCT_ENTER:
      case COL.GAP_PCT_LEAVE:
      case COL.GAP_PCT_DUR_COL: return this._buildGapColumnPercentiles();
      case COL.GAP_PCT_DUR_FLAT: return this._buildGapFlatPercentiles();
      case COL.PCT_DUR_REGION: return this._buildRegionDurationPercentiles();
      case COL.COMP_PREFIX: return this._buildCompPrefix();
      default: throw new Error(`TextSource: unknown column "${name}"`);
    }
  }

  _alloc(perRankLength) {
    const out = new Array(this.nRanks);
    for (let r = 0; r < this.nRanks; r++) out[r] = new Float64Array(perRankLength(r));
    return out;
  }

  /** MPI_Wait/MPI_Waitall count before each interval. (was buildIterationIndices) */
  _buildIterIndex() {
    const wait = this.regionId('MPI_Wait'), waitall = this.regionId('MPI_Waitall');
    const out = new Array(this.nRanks);
    for (let r = 0; r < this.nRanks; r++) {
      const R = this._regions[r], n = this._counts[r];
      // Uint32 rather than the file format's u16: a legacy text trace can hold more than
      // 65,535 iterations, and `iterIdx % iterationColors.length` would then wrap wrongly.
      const col = new Uint32Array(n);
      let iteration = 0;
      for (let p = 0; p < n; p++) {
        col[p] = iteration;
        if (R[p] === wait || R[p] === waitall) iteration++;
      }
      out[r] = col;
    }
    this._cols.set(COL.ITER, out);
  }

  /**
   * Per-column percentiles of start, end and duration. "The same call" across ranks is matched
   * by the same interval index, the standard case for SPMD codes where ranks execute symmetric
   * call sequences. All three come from one pass because they share the column enumeration.
   * (was buildHeatmapColors + buildHeatmapDurationColors)
   */
  _buildIntervalColumnPercentiles() {
    const n = this.nRanks;
    let maxRows = 0;
    for (let r = 0; r < n; r++) maxRows = Math.max(maxRows, this._counts[r]);

    const enter = this._alloc(r => this._counts[r]);
    const leave = this._alloc(r => this._counts[r]);
    const dur = this._alloc(r => this._counts[r]);

    const ranksIn = new Int32Array(n);
    const vals = new Float64Array(n);
    const pct = new Float64Array(n);

    for (let p = 0; p < maxRows; p++) {
      let m = 0;
      for (let r = 0; r < n; r++) if (p < this._counts[r]) ranksIn[m++] = r;
      if (m === 0) continue;
      for (let which = 0; which < 3; which++) {
        for (let k = 0; k < m; k++) {
          const r = ranksIn[k];
          vals[k] = which === 0 ? this._starts[r][p]
                  : which === 1 ? this._ends[r][p]
                                : this._ends[r][p] - this._starts[r][p];
        }
        rankPercentiles(vals, m, pct);
        const out = which === 0 ? enter : which === 1 ? leave : dur;
        for (let k = 0; k < m; k++) out[ranksIn[k]][p] = pct[k];
      }
    }

    this._cols.set(COL.PCT_ENTER, enter);
    this._cols.set(COL.PCT_LEAVE, leave);
    this._cols.set(COL.PCT_DUR_COL, dur);
  }

  /**
   * The same three, for the Comp gap following each interval. Gap `g` on a rank runs from
   * interval `g`'s end to interval `g+1`'s start, so a rank with N intervals has N-1 of them;
   * the time before the first interval and after the last is not a call-bounded gap and falls
   * back to the flat colour at render time.
   * (was buildCompHeatmapColors + buildCompHeatmapDurationColors)
   */
  _buildGapColumnPercentiles() {
    const n = this.nRanks;
    const gapCount = r => Math.max(0, this._counts[r] - 1);
    let maxGaps = 0;
    for (let r = 0; r < n; r++) maxGaps = Math.max(maxGaps, gapCount(r));

    const enter = this._alloc(gapCount);
    const leave = this._alloc(gapCount);
    const dur = this._alloc(gapCount);

    const ranksIn = new Int32Array(n);
    const vals = new Float64Array(n);
    const pct = new Float64Array(n);

    for (let g = 0; g < maxGaps; g++) {
      let m = 0;
      for (let r = 0; r < n; r++) if (g < gapCount(r)) ranksIn[m++] = r;
      if (m === 0) continue;
      for (let which = 0; which < 3; which++) {
        for (let k = 0; k < m; k++) {
          const r = ranksIn[k];
          const gs = this._ends[r][g], ge = this._starts[r][g + 1];
          vals[k] = which === 0 ? gs : which === 1 ? ge : ge - gs;
        }
        rankPercentiles(vals, m, pct);
        const out = which === 0 ? enter : which === 1 ? leave : dur;
        for (let k = 0; k < m; k++) out[ranksIn[k]][g] = pct[k];
      }
    }

    this._cols.set(COL.GAP_PCT_ENTER, enter);
    this._cols.set(COL.GAP_PCT_LEAVE, leave);
    this._cols.set(COL.GAP_PCT_DUR_COL, dur);
  }

  /**
   * Ranks every Comp gap, on every rank, at every position, against every other gap in the run
   * — one flat pool rather than per-column. (was buildCompHeatmapRunColors)
   */
  _buildGapFlatPercentiles() {
    const n = this.nRanks;
    let total = 0;
    for (let r = 0; r < n; r++) total += Math.max(0, this._counts[r] - 1);

    const flat = new Float64Array(total);
    let w = 0;
    for (let r = 0; r < n; r++) {
      const S = this._starts[r], E = this._ends[r], c = this._counts[r];
      for (let g = 0; g < c - 1; g++) flat[w++] = S[g + 1] - E[g];
    }
    const pct = new Float64Array(total);
    rankPercentiles(flat, total, pct);

    const out = this._alloc(r => Math.max(0, this._counts[r] - 1));
    let k = 0;
    for (let r = 0; r < n; r++) {
      const col = out[r];
      for (let g = 0; g < col.length; g++) col[g] = pct[k++];
    }
    this._cols.set(COL.GAP_PCT_DUR_FLAT, out);
  }

  /**
   * Ranks each interval's duration against every occurrence of the *same region* anywhere in
   * the run — across ranks and across loop iterations, one flat pool per region.
   * (was buildSpeedHeatmapRunColors, which did this for one region at a time)
   */
  _buildRegionDurationPercentiles() {
    const n = this.nRanks;
    const nRegions = this.regionNames.length;

    // Count per region, then fill pools in (rank, row) order so ties break exactly as the old
    // builder's rank-major enumeration did.
    const counts = new Int32Array(nRegions);
    for (let r = 0; r < n; r++) {
      const R = this._regions[r], c = this._counts[r];
      for (let p = 0; p < c; p++) counts[R[p]]++;
    }
    const pools = new Array(nRegions);
    const refRank = new Array(nRegions);
    const refRow = new Array(nRegions);
    for (let g = 0; g < nRegions; g++) {
      pools[g] = new Float64Array(counts[g]);
      refRank[g] = new Int32Array(counts[g]);
      refRow[g] = new Int32Array(counts[g]);
    }
    const fill = new Int32Array(nRegions);
    for (let r = 0; r < n; r++) {
      const R = this._regions[r], S = this._starts[r], E = this._ends[r], c = this._counts[r];
      for (let p = 0; p < c; p++) {
        const g = R[p], w = fill[g]++;
        pools[g][w] = E[p] - S[p];
        refRank[g][w] = r;
        refRow[g][w] = p;
      }
    }

    const out = this._alloc(r => this._counts[r]);
    let widest = 1;
    for (let g = 0; g < nRegions; g++) if (counts[g] > widest) widest = counts[g];
    const scratch = new Float64Array(widest);
    for (let g = 0; g < nRegions; g++) {
      const m = counts[g];
      if (m === 0) continue;
      rankPercentiles(pools[g], m, scratch);
      const rr = refRank[g], rw = refRow[g];
      for (let k = 0; k < m; k++) out[rr[k]][rw[k]] = scratch[k];
    }
    this._cols.set(COL.PCT_DUR_REGION, out);
  }

  /**
   * Cumulative comp (non-MPI) time at each interval's start. The piecewise-linear function this
   * describes has slope 1 during comp gaps and slope 0 during MPI calls, which is what lets
   * compTimeAt() answer for an arbitrary timestamp. (was buildSpeedHeatmapPrefixes)
   */
  _buildCompPrefix() {
    const out = this._alloc(r => this._counts[r]);
    for (let r = 0; r < this.nRanks; r++) {
      const S = this._starts[r], E = this._ends[r], c = this._counts[r], col = out[r];
      let cum = 0, prevEnd = 0;
      for (let p = 0; p < c; p++) {
        cum += Math.max(0, S[p] - prevEnd);
        col[p] = cum;
        prevEnd = E[p];
      }
    }
    this._cols.set(COL.COMP_PREFIX, out);
  }

  // --- cursor ----------------------------------------------------------------------------

  async ensure(_t) { /* all in memory */ }
  ready(_t) { return true; }

  sampleAll(t, out) {
    const n = this.nRanks;
    for (let i = 0; i < n; i++) {
      const S = this._starts[i], E = this._ends[i], c = this._counts[i];
      const p = bracketIndex(E, c, t);
      out.curChunk[i] = i;
      out.prevChunk[i] = i;
      out.prevRow[i] = p > 0 ? p - 1 : -1;
      out.prevEnd[i] = p > 0 ? E[p - 1] : 0;
      if (p < c) {
        out.curRow[i] = p;
        out.curStart[i] = S[p];
        out.curEnd[i] = E[p];
        out.inside[i] = (t >= S[p] && t <= E[p]) ? 1 : 0;
        out.curIsLast[i] = (p === c - 1) ? 1 : 0;
      } else {
        out.curRow[i] = -1;
        out.curStart[i] = 0;
        out.curEnd[i] = 0;
        out.inside[i] = 0;
        out.curIsLast[i] = 0;
      }
    }
  }

  /**
   * Cumulative comp time for `rank` from timeline zero up to an arbitrary `t` — needed by
   * "Heatmap instantaneous speed", which measures pace over a trailing window and so has to
   * query a timestamp behind the playhead, not just the current one.
   */
  compTimeAt(rank, t) {
    const prefix = this._cols.get(COL.COMP_PREFIX);
    const S = this._starts[rank], E = this._ends[rank], c = this._counts[rank];
    if (!prefix || c === 0) return Math.max(0, t);
    const col = prefix[rank];

    const lo = bracketIndex(E, c, t);
    if (lo >= c) return col[c - 1] + Math.max(0, t - E[c - 1]);
    if (t < S[lo]) {
      const prevEnd = lo > 0 ? E[lo - 1] : 0;
      const prevPrefix = lo > 0 ? col[lo - 1] : 0;
      return prevPrefix + Math.max(0, t - prevEnd);
    }
    return col[lo];   // inside the call itself: flat, no comp time accrues
  }

  /** Rows of each rank whose region is `regionId`, in order. Cached per region. */
  runIndex(regionId) {
    let cached = this._runIndexCache.get(regionId);
    if (cached) return cached;
    cached = new Array(this.nRanks);
    for (let r = 0; r < this.nRanks; r++) {
      const R = this._regions[r], c = this._counts[r];
      let m = 0;
      for (let p = 0; p < c; p++) if (R[p] === regionId) m++;
      const idxs = new Int32Array(m);
      let w = 0;
      for (let p = 0; p < c; p++) if (R[p] === regionId) idxs[w++] = p;
      cached[r] = idxs;
    }
    this._runIndexCache.set(regionId, cached);
    return cached;
  }

  /**
   * The same bracket as sampleAll(), but restricted to one region's occurrences — "MPI Heatmap
   * instantaneous speed (run)" colours by the surrounding occurrences of one named call, which
   * may be many ordinary intervals away.
   */
  sampleRun(index, t, out) {
    for (let i = 0; i < this.nRanks; i++) {
      const idxs = index[i], m = idxs.length;
      const E = this._ends[i], S = this._starts[i];
      // Binary search over the filtered subsequence's ends.
      let lo = 0, hi = m;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (E[idxs[mid]] < t) lo = mid + 1;
        else hi = mid;
      }
      out.curChunk[i] = i;
      out.prevChunk[i] = i;
      out.prevRow[i] = lo > 0 ? idxs[lo - 1] : -1;
      out.prevEnd[i] = lo > 0 ? E[idxs[lo - 1]] : 0;
      if (lo < m) {
        const row = idxs[lo];
        out.curRow[i] = row;
        out.curStart[i] = S[row];
        out.curEnd[i] = E[row];
        out.inside[i] = (t >= S[row] && t <= E[row]) ? 1 : 0;
        out.curIsLast[i] = (lo === m - 1) ? 1 : 0;
      } else {
        out.curRow[i] = -1;
        out.curStart[i] = 0;
        out.curEnd[i] = 0;
        out.inside[i] = 0;
        out.curIsLast[i] = 0;
      }
    }
  }

  /** Smallest interval start strictly after `t` across all ranks; Infinity if there is none. */
  nextEventAfter(t) {
    let best = Infinity;
    for (let r = 0; r < this.nRanks; r++) {
      const S = this._starts[r], c = this._counts[r];
      const j = firstStartAfter(S, c, t);
      if (j < c && S[j] < best) best = S[j];
    }
    return best;
  }

  /** Columnar per-rank arrays for the timeline HUD. */
  overview() {
    return {
      nRanks: this.nRanks,
      counts: this._counts,
      starts: this._starts,
      ends: this._ends,
      regions: this._regions,
      regionNames: this.regionNames,
      t1: this.t1,
    };
  }
}

// ---------------------------------------------------------------------------
// Pv3dSource — the `.pv3dbin` loader
// ---------------------------------------------------------------------------

/** Chunk id meaning "this row is in the level's resident long-interval table". */
export const RESIDENT = -1;

/**
 * Serves the same columnar model out of a `.pv3dbin`, one block at a time.
 *
 * Where `TextSource` parses a whole file and computes every derived column in the browser,
 * this one reads them: the converter already stored `pct_enter`, `iter_index`, `comp_prefix`
 * and the rest, so opening a trace costs a block directory (24 bytes per block) plus the
 * resident table, and each frame costs the block covering that instant.
 *
 * ### Units
 * A tick here is one timer cycle, so `ticksPerSecond` is the trace's own `timer_resolution`
 * and no scaling happens between the file and the renderer's clock. Renderer time runs from 0
 * at the level's first interval; the level's quantised absolute time is
 * `floor(t / quant) + t0_q`.
 *
 * ### The three-block window
 * "What is every rank doing at t" is exactly one block — that is what a block is. But the
 * *previous* interval of a rank may have ended in the block before, and its *next* may not
 * start until the block after, and the gradient in-between modes need both. So the cursor
 * consults blocks b-1, b and b+1, which are the ones the cache holds anyway (b is fetched,
 * b-1 pinned, b+1 prefetched). A rank with nothing in any of the three renders as though it
 * were past the end of the trace — the same fallback the text path uses at the trace edges.
 */
export class Pv3dSource {
  constructor(file, level) {
    this.file = file;
    this.lvl = level;
    this.format = 'pv3dbin';
    this.nRanks = file.nRanks;
    this.regionNames = file.regions;
    this.ticksPerSecond = file.timerResolution;
    this.quant = level.quant;
    this.t0q = level.meta.t0_q ?? 0;
    this.t1q = level.meta.t1_q ?? 0;
    this.t0 = this.t0q * this.quant;                    // cycles since ProgramBegin
    this.t1 = Math.max(0, (this.t1q - this.t0q) * this.quant);
    this.dropped = level.dropped;
    this.nSource = level.meta.n_source ?? level.nRows;
    this.levelInfo = { level: level.level, quant: level.quant,
                       dropped: level.dropped, nRows: level.nRows,
                       nSource: this.nSource };

    this._regionIds = new Map();
    this.regionNames.forEach((n, i) => { if (!this._regionIds.has(n)) this._regionIds.set(n, i); });

    /** Hardware counters this file carries, in header order; empty when PAPI was not stored. */
    this.papiCounters = (file.header.papi && file.header.papi.counters) || [];

    // Always-present columns: the bracket needs them every frame. `depth` is free — it sits
    // between `regions` and `flags` in the block layout, so the one byte range that fetches
    // the others already covers it — and it disambiguates spanning chains.
    this._columns = new Set(['starts', 'ends', 'regions', 'depth', 'flags']);
    //: How many block boundaries an interval may cross before the converter moves it to the
    //: resident table instead of duplicating it — the bound on a chain walk.
    this.spanLimit = (file.header.options && file.header.options.span_duplicate_limit) || 4;
    this._runCache = new Map();
    this._sample = null;
  }

  static async open(transport, levelNumber) {
    const { Pv3dFile } = await import('./pv3d.js');
    const file = await Pv3dFile.open(transport);
    const lvl = file.level(levelNumber);
    await lvl.load();
    return new Pv3dSource(file, lvl);
  }

  /**
   * Why a file cannot be drawn, or null when it can.
   *
   * A version-1 file, or one converted with `--no-derived`, has no percentile columns and no
   * `t0_q`: the viewer would show a correctly-timed but entirely uncoloured trace. Saying so
   * is more useful than rendering that.
   */
  unsupportedReason() {
    const missing = ['pct_enter', 'iter_index', 'comp_prefix'].filter(c => !this.lvl.hasColumn(c));
    if (missing.length) {
      return `This .pv3dbin has no derived columns (missing ${missing.join(', ')}), so the `
           + `viewer cannot colour it. Re-convert with otf2_to_pv3d.py without --no-derived `
           + `(format version ${this.file.header.version}; version 2 is required).`;
    }
    if (this.lvl.meta.t0_q === undefined) {
      return 'This .pv3dbin predates format version 2 and does not record the trace\'s visible '
           + 'time span. Re-convert it with the current otf2_to_pv3d.py.';
    }
    return null;
  }

  regionId(name) {
    const id = this._regionIds.get(name);
    return id === undefined ? -1 : id;
  }

  hasColumn(name) { return this.lvl.hasColumn(name); }

  async ensureColumns(names) {
    for (const n of names) if (this.lvl.hasColumn(n)) this._columns.add(n);
  }

  get columnList() { return [...this._columns]; }

  // --- time conversions -------------------------------------------------------------------

  /** Renderer ticks -> this level's absolute quantised time. */
  absQ(t) { return Math.floor(t / this.quant) + this.t0q; }
  /** Absolute quantised time -> renderer ticks. */
  ticks(q) { return (q - this.t0q) * this.quant; }

  // --- block window -----------------------------------------------------------------------

  _window(tq) {
    const b = this.lvl.blockOf(tq);
    const out = [b];
    if (b > 0) out.push(b - 1);
    if (b + 1 < this.lvl.nBlocks) out.push(b + 1);
    return out;
  }

  async ensure(t) {
    const cols = this.columnList;
    const win = this._window(this.absQ(t));
    // The centre block gates rendering; the neighbours are warmed but not waited on.
    await this.lvl.block(win[0], cols);
    for (let i = 1; i < win.length; i++) this.lvl.block(win[i], cols).catch(() => {});
  }

  ready(t) { return this.lvl.ready(this.lvl.blockOf(this.absQ(t)), this.columnList); }

  // --- column access ----------------------------------------------------------------------

  /**
   * Scale from a stored value to what the renderer expects: percentiles come back as 0-100,
   * comp time as ticks, everything else as itself.
   */
  _scale(name) {
    if (name === 'comp_prefix') return this.quant;
    return name.startsWith('pct_') || name.startsWith('gap_pct_') ? (100 / 255) : 1;
  }

  /**
   * Column holding one counter's increment over a call (`'call'`) or over the Comp gap that
   * follows it (`'gap'`), or null when that counter has no column.
   *
   * A counter whose increment is zero everywhere in the trace is marked `constant` and no
   * column is written for it — single-precision counters on a double-precision code, say.
   * The caller must distinguish that ("the answer is zero") from an absent counter ("there is
   * no answer"), which is why `constant` is reported separately rather than folded in here.
   */
  papiColumnFor(counterName, scope) {
    const cols = this.lvl.meta.papi_columns || [];
    for (const c of cols) {
      if (c.counter === counterName && c.scope === scope) {
        return c.constant ? null : c.name;
      }
    }
    return null;
  }

  /** True when the counter exists but never moves, so its increment is known to be zero. */
  papiIsConstant(counterName, scope) {
    const cols = this.lvl.meta.papi_columns || [];
    for (const c of cols) if (c.counter === counterName && c.scope === scope) return !!c.constant;
    return false;
  }

  /**
   * One counter increment, or NaN when there is no answer — a row in a block that is not
   * cached, or a column this file does not carry.
   *
   * NaN rather than 0 deliberately: 0 is a perfectly ordinary increment (it is the *correct*
   * answer for a single-precision counter on this code), so it cannot also mean "unknown".
   * PAPI must never go through `cell()`, which returns 0 for a missing column.
   */
  papiDelta(name, chunk, row) {
    if (!name || row < 0) return NaN;
    const src = chunk === RESIDENT
      ? this.lvl.longIntervals.columns[name]
      : (this.lvl.cached(chunk) || { columns: {} }).columns[name];
    if (!src || row >= src.length) return NaN;
    return Number(src[row]);
  }

  cell(name, chunk, row) {
    if (row < 0) return 0;
    const src = chunk === RESIDENT
      ? this.lvl.longIntervals.columns[name]
      : (this.lvl.cached(chunk) || { columns: {} }).columns[name];
    if (!src || row >= src.length) return 0;
    return src[row] * this._scale(name);
  }

  // --- cursor -----------------------------------------------------------------------------

  /**
   * Resolves, for every rank, the interval bracketing `t` and the one before it, searching
   * the resident table and the three-block window together. Both sequences are sorted by
   * start and neither overlaps itself, so the merge is two binary searches and a comparison.
   */
  sampleAll(t, out) {
    const tq = this.absQ(t);
    const blocks = [];
    for (const b of this._window(tq)) {
      const blk = this.lvl.cached(b);
      if (blk && blk.nRows > 0) blocks.push({ b, blk, base: b * this.lvl.blockLenQ });
    }
    const longs = this.lvl.longIntervals;
    const lStart = longs.columns.starts, lEnd = longs.columns.ends, lRankStart = longs.rankStart;
    const lFlags = longs.columns.flags;

    for (let i = 0; i < this.nRanks; i++) {
      // --- the interval bracketing tq: the earliest whose end reaches it ------------------
      let curEnd = Infinity, curStart = 0, curChunk = 0, curRow = -1, curLast = 0;

      if (lRankStart) {
        const lo = lRankStart[i], hi = lRankStart[i + 1];
        if (hi > lo) {
          const j = lowerBoundNumeric(lEnd, lo, hi, tq);
          if (j < hi && Number(lEnd[j]) < curEnd) {
            curEnd = Number(lEnd[j]); curStart = Number(lStart[j]);
            curChunk = RESIDENT; curRow = j;
            curLast = (lFlags && (lFlags[j] & 4)) ? 1 : 0;
          }
        }
      }
      for (const { b, blk, base } of blocks) {
        const lo = blk.rankOffsets[i], hi = blk.rankOffsets[i + 1];
        if (hi <= lo) continue;
        const E = blk.columns.ends;
        const j = lowerBoundNumeric(E, lo, hi, tq - base);
        if (j < hi) {
          const e = E[j] + base;
          if (e < curEnd) {
            curEnd = e; curStart = blk.columns.starts[j] + base; curChunk = b; curRow = j;
            curLast = (blk.columns.flags && (blk.columns.flags[j] & 4)) ? 1 : 0;
          }
        }
      }

      // An interval that crosses a block boundary is stored clipped in each block it touches,
      // so the extents just read are what is *visible in that block*, not the call's own. Using
      // them for `progress` would replay the whole gradient once per block. Walk the chain.
      if (curRow >= 0 && curChunk !== RESIDENT) {
        const flags = this.lvl.cached(curChunk).columns.flags[curRow];
        if (flags & 3) {
          const ext = this._chainExtents(i, curChunk, curRow, curStart, curEnd);
          curStart = ext[0];
          curEnd = ext[1];
        }
      }

      // --- the interval before it -----------------------------------------------------------
      // Bounded by the current interval's *true* start rather than by tq, which is what keeps
      // an earlier fragment of the current interval from being mistaken for the previous one.
      const before = curRow >= 0 ? curStart : tq;
      let prevEnd = -Infinity, prevChunk = 0, prevRow = -1;
      if (lRankStart) {
        const lo = lRankStart[i], hi = lRankStart[i + 1];
        if (hi > lo) {
          const j = lowerBoundNumeric(lEnd, lo, hi, before);
          if (j > lo && Number(lEnd[j - 1]) > prevEnd) {
            prevEnd = Number(lEnd[j - 1]); prevChunk = RESIDENT; prevRow = j - 1;
          }
        }
      }
      for (const { b, blk, base } of blocks) {
        const lo = blk.rankOffsets[i], hi = blk.rankOffsets[i + 1];
        if (hi <= lo) continue;
        const E = blk.columns.ends;
        const j = lowerBoundNumeric(E, lo, hi, before - base);
        if (j > lo) {
          const e = E[j - 1] + base;
          if (e > prevEnd) { prevEnd = e; prevChunk = b; prevRow = j - 1; }
        }
      }

      out.curChunk[i] = curChunk;
      out.prevChunk[i] = prevChunk;
      out.prevRow[i] = prevRow;
      out.prevEnd[i] = prevRow >= 0 ? this.ticks(prevEnd) : 0;
      if (curRow >= 0) {
        out.curRow[i] = curRow;
        out.curStart[i] = this.ticks(curStart);
        out.curEnd[i] = this.ticks(curEnd);
        out.inside[i] = (tq >= curStart && tq <= curEnd) ? 1 : 0;
        out.curIsLast[i] = curLast;
      } else {
        out.curRow[i] = -1;
        out.curStart[i] = 0;
        out.curEnd[i] = 0;
        out.inside[i] = 0;
        out.curIsLast[i] = 0;
      }
    }
  }

  /**
   * True extents of a clipped interval, by following its copies into neighbouring blocks.
   *
   * Consecutive copies satisfy `next.start == prev.end + 1` and share region and depth
   * (PV3D_FORMAT.md §7), which is unambiguous because two intervals of one rank at one depth
   * cannot overlap. Only blocks already in the cache are consulted: a chain reaching past the
   * prefetch window is rare, and one frame of clipped extents is a better failure than a
   * synchronous fetch in the middle of a frame.
   */
  _chainExtents(rank, b, j, start, end) {
    const lvl = this.lvl;
    let blk = lvl.cached(b);
    //: Every column the walk dereferences. A block cached by a whole-level scan may hold only
    //: some of them, and a half-walked chain is worse than none: fall back to the clipped
    //: extents, which are at least self-consistent.
    const usable = (x) => x && x.columns.starts && x.columns.ends && x.columns.flags
                        && x.columns.regions && x.columns.depth;
    if (!usable(blk)) return [start, end];
    let curB = b, curJ = j, lo, hi;

    // forward, while this copy is cut at its block's end
    let steps = 0;
    while ((blk.columns.flags[curJ] & 2) && steps++ <= this.spanLimit) {
      const nb = curB + 1;
      const nblk = lvl.cached(nb);
      if (!usable(nblk) || nblk.nRows === 0) break;
      const base = nb * lvl.blockLenQ;
      lo = nblk.rankOffsets[rank]; hi = nblk.rankOffsets[rank + 1];
      let found = -1;
      for (let k = lo; k < hi; k++) {
        const st = nblk.columns.starts[k] + base;
        if (st > end + 1) break;                       // rows are start-ordered
        if (st === end + 1 && (nblk.columns.flags[k] & 1)
            && nblk.columns.regions[k] === blk.columns.regions[curJ]
            && nblk.columns.depth[k] === blk.columns.depth[curJ]) { found = k; break; }
      }
      if (found < 0) break;
      end = nblk.columns.ends[found] + base;
      curB = nb; curJ = found; blk = nblk;
    }

    // backward, while the first copy we have is cut at its block's start
    blk = lvl.cached(b); curB = b; curJ = j; steps = 0;
    while ((blk.columns.flags[curJ] & 1) && steps++ <= this.spanLimit) {
      const pb = curB - 1;
      const pblk = pb >= 0 ? lvl.cached(pb) : null;
      if (!usable(pblk) || pblk.nRows === 0) break;
      const base = pb * lvl.blockLenQ;
      lo = pblk.rankOffsets[rank]; hi = pblk.rankOffsets[rank + 1];
      let found = -1;
      for (let k = hi - 1; k >= lo; k--) {
        const en = pblk.columns.ends[k] + base;
        if (en < start - 1) break;                     // rows are start-ordered
        if (en === start - 1 && (pblk.columns.flags[k] & 2)
            && pblk.columns.regions[k] === blk.columns.regions[curJ]
            && pblk.columns.depth[k] === blk.columns.depth[curJ]) { found = k; break; }
      }
      if (found < 0) break;
      start = pblk.columns.starts[found] + base;
      curB = pb; curJ = found; blk = pblk;
    }

    return [start, end];
  }

  /**
   * Cumulative comp time at an arbitrary `t`, from the stored `comp_prefix`. Same piecewise
   * reconstruction as the text path: flat inside a call, slope 1 in the gap before it.
   */
  compTimeAt(rank, t) {
    const tq = this.absQ(t);
    const blocks = [];
    for (const b of this._window(tq)) {
      const blk = this.lvl.cached(b);
      if (blk && blk.nRows > 0) blocks.push({ b, blk, base: b * this.lvl.blockLenQ });
    }
    let curPrefix = null, curStart = 0;
    let prevEnd = null, prevPrefix = 0;
    for (const { b, blk, base } of blocks) {
      const lo = blk.rankOffsets[rank], hi = blk.rankOffsets[rank + 1];
      if (hi <= lo) continue;
      const S = blk.columns.starts, E = blk.columns.ends;
      const j = lowerBoundNumeric(E, lo, hi, tq - base);
      if (j < hi && (curPrefix === null || S[j] + base < curStart)) {
        curStart = S[j] + base;
        curPrefix = this.cell(COL.COMP_PREFIX, b, j);
      }
      if (j > lo && (prevEnd === null || E[j - 1] + base > prevEnd)) {
        prevEnd = E[j - 1] + base;
        prevPrefix = this.cell(COL.COMP_PREFIX, b, j - 1);
      }
    }
    if (curPrefix !== null && tq >= curStart) return curPrefix;   // inside the call: flat
    const base = prevEnd === null ? 0 : this.ticks(prevEnd);
    return prevPrefix + Math.max(0, t - base);
  }

  /** Smallest interval start strictly after `t`, within the block window. */
  nextEventAfter(t) {
    const tq = this.absQ(t);
    let best = Infinity;
    for (const b of this._window(tq)) {
      const blk = this.lvl.cached(b);
      if (!blk || blk.nRows === 0) continue;
      const base = b * this.lvl.blockLenQ;
      const S = blk.columns.starts;
      for (let i = 0; i < this.nRanks; i++) {
        const lo = blk.rankOffsets[i], hi = blk.rankOffsets[i + 1];
        if (hi <= lo) continue;
        let a = lo, z = hi;
        while (a < z) { const m = (a + z) >> 1; if (S[m] + base <= tq) a = m + 1; else z = m; }
        if (a < hi) { const v = this.ticks(S[a] + base); if (v < best) best = v; }
      }
    }
    return best;
  }

  /**
   * Per-rank index of one region's occurrences across the *whole* level, for "instantaneous
   * speed (run)". This is the one metric that needs a global per-rank sequence rather than
   * the current block, so it pays for one whole-level pass over three columns — u8 regions
   * plus the two time columns, ~4 MB at level 0 on a 220k-interval trace — done once per
   * Visualize click rather than per frame.
   */
  async runIndex(regionId) {
    const cached = this._runCache.get(regionId);
    if (cached) return cached;

    const perRank = Array.from({ length: this.nRanks }, () => []);
    if (regionId >= 0) {
      for (let b = 0; b < this.lvl.nBlocks; b++) {
        if (this.lvl.dir.nRows[b] === 0) continue;
        const blk = await this.lvl.block(b, [...this._columns, COL.PCT_DUR_REGION]);
        const base = b * this.lvl.blockLenQ;
        const R = blk.columns.regions, S = blk.columns.starts, E = blk.columns.ends;
        for (let i = 0; i < this.nRanks; i++) {
          const lo = blk.rankOffsets[i], hi = blk.rankOffsets[i + 1];
          for (let j = lo; j < hi; j++) {
            if (R[j] !== regionId) continue;
            const list = perRank[i];
            // A spanning interval appears clipped in each block it touches; keep only the
            // copy carrying the true start, so an occurrence is counted once.
            if (blk.columns.flags && (blk.columns.flags[j] & 1)) continue;
            list.push(S[j] + base, E[j] + base, b, j);
          }
        }
      }
      const lr = this.lvl.longIntervals;
      if (lr.count && lr.columns.regions) {
        for (let k = 0; k < lr.count; k++) {
          if (lr.columns.regions[k] !== regionId) continue;
          perRank[lr.columns.rank[k]].push(Number(lr.columns.starts[k]), Number(lr.columns.ends[k]), RESIDENT, k);
        }
      }
    }

    const index = perRank.map(flat => {
      const n = flat.length / 4;
      const order = Array.from({ length: n }, (_, k) => k).sort((a, z) => flat[a * 4] - flat[z * 4]);
      const starts = new Float64Array(n), ends = new Float64Array(n);
      const chunks = new Int32Array(n), rowsA = new Int32Array(n);
      order.forEach((k, pos) => {
        starts[pos] = flat[k * 4]; ends[pos] = flat[k * 4 + 1];
        chunks[pos] = flat[k * 4 + 2]; rowsA[pos] = flat[k * 4 + 3];
      });
      return { n, starts, ends, chunks, rows: rowsA };
    });
    this._runCache.set(regionId, index);
    return index;
  }

  sampleRun(index, t, out) {
    const tq = this.absQ(t);
    for (let i = 0; i < this.nRanks; i++) {
      const ix = index[i];
      const j = lowerBoundNumeric(ix.ends, 0, ix.n, tq);
      out.prevChunk[i] = j > 0 ? ix.chunks[j - 1] : 0;
      out.prevRow[i] = j > 0 ? ix.rows[j - 1] : -1;
      out.prevEnd[i] = j > 0 ? this.ticks(ix.ends[j - 1]) : 0;
      if (j < ix.n) {
        out.curChunk[i] = ix.chunks[j];
        out.curRow[i] = ix.rows[j];
        out.curStart[i] = this.ticks(ix.starts[j]);
        out.curEnd[i] = this.ticks(ix.ends[j]);
        out.inside[i] = (tq >= ix.starts[j] && tq <= ix.ends[j]) ? 1 : 0;
        out.curIsLast[i] = (j === ix.n - 1) ? 1 : 0;
      } else {
        out.curChunk[i] = 0; out.curRow[i] = -1;
        out.curStart[i] = 0; out.curEnd[i] = 0;
        out.inside[i] = 0; out.curIsLast[i] = 0;
      }
    }
  }

  /** Whole-level columnar view for the timeline HUD. */
  async overview() {
    const perRank = Array.from({ length: this.nRanks }, () => []);
    for (let b = 0; b < this.lvl.nBlocks; b++) {
      if (this.lvl.dir.nRows[b] === 0) continue;
      // The cursor's own column set, not a subset: a block cached here with fewer columns
      // would later be found by sampleAll() and dereferenced for one it does not hold.
      const blk = await this.lvl.block(b, [...this._columns]);
      const base = b * this.lvl.blockLenQ;
      const S = blk.columns.starts, E = blk.columns.ends, R = blk.columns.regions, F = blk.columns.flags;
      for (let i = 0; i < this.nRanks; i++) {
        for (let j = blk.rankOffsets[i]; j < blk.rankOffsets[i + 1]; j++) {
          if (F && (F[j] & 1)) continue;            // keep one copy of a spanning interval
          perRank[i].push(this.ticks(S[j] + base), this.ticks(E[j] + base), R[j]);
        }
      }
    }
    const lr = this.lvl.longIntervals;
    if (lr.count) {
      for (let k = 0; k < lr.count; k++) {
        perRank[lr.columns.rank[k]].push(this.ticks(Number(lr.columns.starts[k])),
                                         this.ticks(Number(lr.columns.ends[k])),
                                         lr.columns.regions[k]);
      }
    }
    const counts = new Int32Array(this.nRanks);
    const starts = [], ends = [], regions = [];
    for (let i = 0; i < this.nRanks; i++) {
      const flat = perRank[i];
      const n = flat.length / 3;
      counts[i] = n;
      const order = Array.from({ length: n }, (_, k) => k).sort((a, z) => flat[a * 3] - flat[z * 3]);
      const S = new Float64Array(n), E = new Float64Array(n), R = new Uint16Array(n);
      order.forEach((k, pos) => { S[pos] = flat[k * 3]; E[pos] = flat[k * 3 + 1]; R[pos] = flat[k * 3 + 2]; });
      starts.push(S); ends.push(E); regions.push(R);
    }
    return { nRanks: this.nRanks, counts, starts, ends, regions,
             regionNames: this.regionNames, t1: this.t1 };
  }
}

/** First index in [lo, hi) of `arr` whose value is >= t. Works for BigInt64 columns too. */
function lowerBoundNumeric(arr, lo, hi, t) {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(arr[mid]) < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * pv3d.js — reader for the PV3D trace format (`.pv3dbin`), as specified in PV3D_FORMAT.md.
 *
 * Pure: no Three.js, no DOM, no reference to the viewer's config. The only thing it needs from
 * its host is a *transport* — an object that can report a file's size and read a byte range —
 * so the same code serves an HTTP URL (Range requests), a local `File` picked through an
 * `<input type="file">` (`File.slice`), and Node's `fs` (which is how it is tested against the
 * Python encoder's own `.pv3dtxt` dump).
 *
 * The central idea is that **the viewer never loads a level**. A block covers every rank over
 * one span of time, so "what is every rank doing at t" is always exactly one block; the reader
 * fetches that block, and within it only the columns the active metric reads. A 3.4 GB level 0
 * would not fit in an ArrayBuffer (~1074 MB measured) let alone the JS heap (~2330 MB), so
 * block-wise fetching is not an optimisation here, it is the only thing that works.
 *
 *   const file = await Pv3dFile.open(httpTransport(url));
 *   const lvl  = file.level(1);
 *   const blk  = await lvl.block(7, ['starts', 'ends', 'regions']);
 *   blk.rankOffsets[r] .. blk.rankOffsets[r+1]   // rank r's rows, sorted by start
 */

const MAGIC_BIN = 'PV3DBIN1';
const MAGIC_TXT = 'PV3DTXT1';

const DTYPE_BYTES = { u8: 1, u16: 2, u32: 4, u64: 8 };
const DTYPE_ARRAY = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, u64: BigUint64Array };

/** Thrown when a server ignores `Range` and answers 200 with the whole file. */
export class RangeUnsupportedError extends Error {
  constructor(url) {
    super(`The server at ${url} ignored the Range request and returned the whole file. ` +
          `python3 -m http.server does this; start serve.py instead.`);
    this.name = 'RangeUnsupportedError';
  }
}

// ---------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------

/** Reads byte ranges over HTTP. Requires a server that answers 206 (see serve.py). */
export function httpTransport(url) {
  let total = null;
  return {
    kind: 'http',
    name: url,
    async size() {
      if (total === null) {
        const r = await fetch(url, { method: 'HEAD' });
        if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
        total = Number(r.headers.get('Content-Length'));
      }
      return total;
    },
    async read(start, end) {              // [start, end)
      const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      // A 200 here means the whole file came back; using it would silently buffer gigabytes.
      if (r.status !== 206) throw new RangeUnsupportedError(url);
      return r.arrayBuffer();
    },
  };
}

/** Reads byte ranges out of a local File chosen in the page — no server at all. */
export function fileTransport(file) {
  return {
    kind: 'file',
    name: file.name,
    async size() { return file.size; },
    async read(start, end) { return file.slice(start, end).arrayBuffer(); },
  };
}

/**
 * Identifies a file from its first 8 bytes, so the viewer can keep opening legacy `.txt`
 * traces without the user having to say which format a file is in.
 */
export async function sniff(transport) {
  // Deliberately no size() call: over HTTP that is a HEAD round trip whose only use would be
  // to rule out an 8-byte file, and a short read answers the same question.
  let head;
  try {
    head = new Uint8Array(await transport.read(0, 8));
  } catch {
    return 'legacy-txt';
  }
  if (head.length < 8) return 'legacy-txt';
  const tag = String.fromCharCode(...head);
  if (tag === MAGIC_BIN) return 'pv3dbin';
  if (tag === MAGIC_TXT) return 'pv3dtxt';
  return 'legacy-txt';
}

// ---------------------------------------------------------------------------
// Pv3dFile
// ---------------------------------------------------------------------------

export class Pv3dFile {
  constructor(transport, header, binOffset) {
    this.transport = transport;
    this.header = header;
    this.binOffset = binOffset;
    this.nRanks = header.n_ranks;
    this.regions = header.regions || [];
    this.timerResolution = header.clock.timer_resolution;
    this.globalOffset = header.clock.global_offset;
    this.papiPairs = false;
    this._levels = new Map();
  }

  /** Three small reads and the file is usable: the container head, the JSON, nothing else. */
  static async open(transport) {
    const head = new DataView(await transport.read(0, 20));
    const magic = String.fromCharCode(...new Uint8Array(head.buffer, 0, 8));
    if (magic !== MAGIC_BIN) throw new Error(`not a .pv3dbin file (magic "${magic}")`);
    const version = head.getUint32(8, true);
    // The versions differ only by additions a reader can detect: 2 records core_bytes in the
    // directory's last field (zero in 1), omits empty blocks and empty other-event sections,
    // and stores the derived columns; 3 stores PAPI as two increment columns per counter
    // (one entry per row) instead of one absolute column (two entries per row).
    if (version < 1 || version > 3) throw new Error(`unsupported PV3D version ${version}`);
    const papiPairs = version < 3;
    const jsonLength = head.getUint32(12, true);
    const binOffset = head.getUint32(16, true);
    const json = new TextDecoder().decode(await transport.read(20, 20 + jsonLength));
    const file = new Pv3dFile(transport, JSON.parse(json), binOffset);
    file.papiPairs = papiPairs;
    return file;
  }

  get levels() { return this.header.levels; }

  level(n) {
    let lvl = this._levels.get(n);
    if (!lvl) {
      const meta = this.header.levels.find(l => l.level === n);
      if (!meta) throw new Error(`no level ${n} in this file`);
      lvl = new Pv3dLevel(this, meta);
      this._levels.set(n, lvl);
    }
    return lvl;
  }

  /**
   * Coarsest level whose quantum still resolves a frame-to-frame jump of `frameStepCycles`.
   * Below that, a finer level would only redraw the same picture with more data fetched.
   */
  levelForStep(frameStepCycles) {
    const levels = [...this.header.levels].sort((a, b) => b.quant - a.quant);
    for (const l of levels) if (l.quant <= frameStepCycles) return l.level;
    return levels[levels.length - 1].level;   // finer than the finest level we have
  }
}

// ---------------------------------------------------------------------------
// Pv3dLevel
// ---------------------------------------------------------------------------

export class Pv3dLevel {
  constructor(file, meta) {
    this.file = file;
    this.meta = meta;
    this.level = meta.level;
    this.quant = meta.quant;
    this.blockLenQ = meta.block_len_q;
    this.nBlocks = meta.n_blocks;
    this.nRows = meta.n_rows;
    this.dropped = meta.dropped ?? 0;

    this.dir = null;        // { offset, length, intervalBytes, coreBytes, nRows } arrays
    this.longIntervals = null;
    this._cache = new Map();   // blockIndex -> Block
    this._pending = new Map(); // blockIndex -> Promise<Block>
    this._lru = [];
    this.cacheCapacity = 12;
    this.bytesFetched = 0;
    this.requests = 0;
  }

  blockOf(tq) {
    const b = Math.floor(tq / this.blockLenQ);
    return b < 0 ? 0 : (b >= this.nBlocks ? this.nBlocks - 1 : b);
  }

  /**
   * Block directory plus the resident long-interval table — the one-off cost of opening a
   * level. 24 bytes per block and a handful of rows per rank; everything else stays on disk.
   */
  async load() {
    if (this.dir) return;
    const { transport, binOffset } = this.file;
    const d = this.meta.directory;
    const base = binOffset + d.byte_offset;
    const buf = await transport.read(base, base + d.row_bytes * d.n_blocks);
    this.file.bytesFetched = (this.file.bytesFetched || 0) + buf.byteLength;

    const dv = new DataView(buf);
    const n = d.n_blocks;
    const dir = {
      offset: new Float64Array(n),       // u64 in the file; Number is exact to 2^53
      length: new Uint32Array(n),
      intervalBytes: new Uint32Array(n),
      coreBytes: new Uint32Array(n),
      nRows: new Uint32Array(n),
    };
    for (let i = 0; i < n; i++) {
      const o = i * d.row_bytes;
      // Read the u64 as two u32s: BigInt on this path would be exact but slow, and a block
      // offset cannot exceed 2^53 for any file that fits on a disk.
      dir.offset[i] = dv.getUint32(o, true) + dv.getUint32(o + 4, true) * 4294967296;
      dir.length[i] = dv.getUint32(o + 8, true);
      dir.intervalBytes[i] = dv.getUint32(o + 12, true);
      dir.nRows[i] = dv.getUint32(o + 16, true);
      // Field 20 was padding in version 1 of the writer and becomes `core_bytes` (bytes
      // through `flags`, i.e. before the derived columns) once the converter writes it. Zero
      // means "not present": fall back to the full interval prefix.
      const core = dv.getUint32(o + 20, true);
      dir.coreBytes[i] = core || dir.intervalBytes[i];
    }
    this.dir = dir;
    this.blocksBase = binOffset + this.meta.blocks_byte_offset;

    await this._loadLongIntervals();
  }

  /**
   * The resident table: intervals too long to duplicate into every block they touch (one
   * MPI_Init per rank, typically). Stored once with *absolute* quanta and loaded whole, since
   * it is what answers "this rank is inside a call that started long before this block".
   */
  async _loadLongIntervals() {
    const li = this.meta.long_intervals;
    this.longIntervals = { count: 0, rankStart: null, columns: {} };
    if (!li || !li.count) return;

    const specs = li.columns.map(s => {
      const [name, dtype] = s.split(':');
      return { name, dtype, size: DTYPE_BYTES[dtype] };
    });
    // This table's columns are aligned against their *absolute* file offset, unlike a block's,
    // which are aligned against the block start. Aligning relative to the table start instead
    // would land on the wrong byte whenever the table does not begin on an 8-byte boundary.
    const base = this.file.binOffset + li.byte_offset;
    let pos = base;
    for (const s of specs) { pos = alignUp(pos, s.size); s.abs = pos; pos += s.size * li.count; }

    const buf = await this.file.transport.read(base, pos);
    this.file.bytesFetched = (this.file.bytesFetched || 0) + buf.byteLength;

    const cols = {};
    for (const s of specs) cols[s.name] = readColumn(buf, s.abs - base, li.count, s.dtype);
    // The table has no rank_offsets of its own; build the index once so lookups are a slice
    // rather than a scan of every long interval on every rank on every frame.
    const nRanks = this.file.nRanks;
    const counts = new Int32Array(nRanks + 1);
    for (let i = 0; i < li.count; i++) counts[cols.rank[i] + 1]++;
    for (let r = 0; r < nRanks; r++) counts[r + 1] += counts[r];
    this.longIntervals = { count: li.count, rankStart: counts, columns: cols };
  }

  /**
   * Byte offset of one column inside a block — a direct port of BlockLayout.__init__ in
   * otf2_to_pv3d.py. Because the layout is a pure function of `n_rows`, `n_ranks` and the
   * level schema, per-column fetching needs no format support at all: the reader simply
   * computes where a column is and asks for those bytes.
   */
  columnOffset(name, nRows) {
    const nRanks = this.file.nRanks;
    let off = 4 * (nRanks + 1);
    for (const col of this.meta.block_columns) {
      const size = DTYPE_BYTES[col.dtype];
      off = alignUp(off, size);
      if (col.name === name) return { offset: off, dtype: col.dtype, size };
      off += size * nRows;
    }
    // Before version 3 a PAPI column held the counter's absolute value at each row's start
    // and end — two entries per row. From version 3 it holds one increment per row.
    const per = this.file.papiPairs ? 2 : 1;
    for (const col of this.meta.papi_columns || []) {
      if (col.constant) continue;
      const size = DTYPE_BYTES[col.dtype];
      off = alignUp(off, size);
      if (col.name === name) return { offset: off, dtype: col.dtype, size, count: nRows * per };
      off += size * nRows * per;
    }
    return null;
  }

  hasColumn(name) {
    return this.meta.block_columns.some(c => c.name === name)
        || (this.meta.papi_columns || []).some(c => c.name === name && !c.constant);
  }

  /**
   * Fetches block `b`, making sure at least `columns` are present. A cached block **grows**:
   * switching metric fetches only the column the new metric added, not the block again.
   */
  async block(b, columns) {
    await this.load();
    const cached = this._cache.get(b);
    if (cached && columns.every(c => cached.columns[c] !== undefined)) {
      this._touch(b);
      return cached;
    }
    const key = b + '|' + columns.join(',');
    let p = this._pending.get(key);
    if (!p) {
      p = this._fetchBlock(b, columns).finally(() => this._pending.delete(key));
      this._pending.set(key, p);
    }
    return p;
  }

  /** True when block `b` is already in the cache with every requested column. */
  ready(b, columns) {
    const c = this._cache.get(b);
    return !!c && columns.every(n => c.columns[n] !== undefined);
  }

  cached(b) { return this._cache.get(b) || null; }

  async _fetchBlock(b, columns) {
    const nRows = this.dir.nRows[b];
    const base = this.blocksBase + this.dir.offset[b];
    let blk = this._cache.get(b);
    if (!blk) {
      blk = { index: b, nRows, rankOffsets: null, columns: {} };
    }

    // An empty block stores nothing at all — not even its rank_offsets, which would be
    // 4*(n_ranks+1) bytes of zeros. `n_rows == 0` in the directory already says every rank
    // has nothing here, so synthesise the offsets and fetch no bytes.
    if (nRows === 0) {
      if (!blk.rankOffsets) blk.rankOffsets = new Uint32Array(this.file.nRanks + 1);
      for (const name of columns) if (blk.columns[name] === undefined) blk.columns[name] = EMPTY;
      this._store(b, blk);
      return blk;
    }

    // Which byte ranges are actually missing? rank_offsets is always needed; the requested
    // columns are contiguous runs in the block, so adjacent ones coalesce into one request.
    const wanted = [];
    if (!blk.rankOffsets) wanted.push({ offset: 0, bytes: 4 * (this.file.nRanks + 1), kind: 'ranks' });
    for (const name of columns) {
      if (blk.columns[name] !== undefined) continue;
      const spec = this.columnOffset(name, nRows);
      if (!spec) { blk.columns[name] = null; continue; }   // not in this file (e.g. no PAPI)
      const count = spec.count ?? nRows;
      wanted.push({ offset: spec.offset, bytes: spec.size * count, kind: 'col', name, spec, count });
    }
    if (wanted.length === 0) { this._store(b, blk); return blk; }

    wanted.sort((x, y) => x.offset - y.offset);
    const runs = [];
    for (const w of wanted) {
      const last = runs[runs.length - 1];
      // Merge anything separated by less than a TCP-round-trip's worth of bytes: one larger
      // request beats two small ones over the network by a wide margin.
      if (last && w.offset - (last.offset + last.bytes) <= 4096) {
        last.bytes = w.offset + w.bytes - last.offset;
        last.items.push(w);
      } else {
        runs.push({ offset: w.offset, bytes: w.bytes, items: [w] });
      }
    }

    const buffers = await Promise.all(runs.map(async (run) => {
      const buf = await this.file.transport.read(base + run.offset, base + run.offset + run.bytes);
      this.bytesFetched += buf.byteLength;
      this.requests++;
      return buf;
    }));

    runs.forEach((run, i) => {
      const buf = buffers[i];
      for (const item of run.items) {
        const rel = item.offset - run.offset;
        if (item.kind === 'ranks') {
          blk.rankOffsets = new Uint32Array(buf.slice(rel, rel + item.bytes));
        } else {
          blk.columns[item.name] = readColumn(buf, rel, item.count, item.spec.dtype);
        }
      }
    });

    this._store(b, blk);
    return blk;
  }

  _store(b, blk) {
    this._cache.set(b, blk);
    this._touch(b);
    while (this._lru.length > this.cacheCapacity) {
      const evict = this._lru.shift();
      if (evict !== b) this._cache.delete(evict);
    }
  }

  _touch(b) {
    const i = this._lru.indexOf(b);
    if (i >= 0) this._lru.splice(i, 1);
    this._lru.push(b);
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Stand-in for any column of an empty block; shared, never written to. */
const EMPTY = new Uint8Array(0);

function alignUp(offset, alignment) {
  const rem = offset % alignment;
  return rem === 0 ? offset : offset + (alignment - rem);
}

/**
 * Views `count` values of `dtype` at `byteOffset` in `buf`.
 *
 * A typed-array view is zero-copy only when its byte offset is a multiple of the item size.
 * Column offsets *within a block* always are (invariant 1), but a byte range fetched from the
 * middle of a file starts wherever it starts, so the check is real and the `slice()` fallback
 * is not dead code.
 */
function readColumn(buf, byteOffset, count, dtype) {
  const Ctor = DTYPE_ARRAY[dtype];
  const size = DTYPE_BYTES[dtype];
  if (byteOffset % size === 0) return new Ctor(buf, byteOffset, count);
  return new Ctor(buf.slice(byteOffset, byteOffset + size * count));
}

export { alignUp, readColumn, DTYPE_BYTES, MAGIC_BIN, MAGIC_TXT };

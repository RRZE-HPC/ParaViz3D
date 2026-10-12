#!/usr/bin/env python3
"""
otf2_to_pv3d.py — convert an OTF2 trace into the PV3D format.

PV3D is a block-fetchable, columnar trace format designed so that a Three.js
viewer can display an arbitrary moment of a very large trace by fetching only
the bytes covering that moment, and can use those bytes *directly* as typed
arrays with no parsing step.

The three properties that shape the whole format, all measured rather than
assumed (see PV3D_FORMAT.md §1 for the numbers):

  1. `new Uint32Array(buffer, byteOffset, n)` is zero-copy and O(1), but throws
     a RangeError unless byteOffset is a multiple of 4.  Every column is
     therefore padded to its own item size, relative to the start of its block.
  2. A single ArrayBuffer caps at ~1 GB in Chromium and the JS heap at ~2.3 GB,
     while a large trace's level-0 data is several GB.  Whole-file loading is
     impossible, so the file is cut into independently fetchable blocks.
  3. Delta-encoded columns cannot be random-accessed: answering "what happened
     at time t" would require decoding every row before t.  Columns are stored
     as block-relative absolutes so they can be binary-searched in place.

The unit of storage is the *interval* (a matched Enter/Leave pair), not the
individual event, because that is what the viewer actually draws and it removes
the need to reconstruct nesting in the browser.

Usage
-----
    otf2_to_pv3d.py TRACE_DIR [--output OUT] [--loss-level 1+2] ...

See `--help`, and PV3D_FORMAT.md for the on-disk specification.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from array import array
from dataclasses import dataclass, field, asdict, fields
from typing import Any

import numpy as np

try:
    import otf2
    import otf2.events as otf2_events
    from otf2.events import Enter, Leave, Metric
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    sys.exit("error: the 'otf2' Python package is required (%s)" % exc)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Format identity and constants
# ─────────────────────────────────────────────────────────────────────────────

MAGIC_BIN = b"PV3DBIN1"
MAGIC_TXT = b"PV3DTXT1"
FORMAT_VERSION = 3

#: Spec minimum and base time-block length, in cycles.  Every level's block
#: length is this times a power-of-two multiplier times the level's quantum.
BASE_BLOCK_CYCLES = 16_777_216  # 2**24

#: Quantisation divisor per loss level.  Level 0 is lossless.
LEVEL_QUANT = (1, 256, 65536)

#: Default "drop intervals shorter than this" threshold per level, in cycles.
LEVEL_DROP = (0, 256, 65536)

#: Because block-relative times are stored quantised, their range is
#: BASE_BLOCK_CYCLES * multiplier regardless of level.  u32 can hold that only
#: while the multiplier stays at or below this value.
MAX_BLOCK_MULTIPLIER = 256

#: Target payload of one block, in bytes.  A 256 KB range request measured
#: 15.4 ms in Chromium, which is a comfortable seek granularity.
TARGET_BLOCK_BYTES = 300_000

#: Interval columns, in the order they are laid out inside a block.  The
#: `count` field says how many entries each row occupies.
INTERVAL_COLUMNS = (
    ("starts", 1),
    ("ends", 1),
    ("regions", 1),
    ("depth", 1),
    ("flags", 1),
)

FLAG_CLIPPED_LEFT = 1
FLAG_CLIPPED_RIGHT = 2
#: Bit 2: this row is its rank's last interval, so the `gap_*` columns — which
#: describe the Comp period *following* an interval — have no meaning for it.
#: Saves the viewer a per-rank interval count it would otherwise have to keep.
FLAG_LAST_INTERVAL = 4

#: Derived columns, in block layout order after `flags`.
#:
#: These hold everything the viewer used to compute for itself in nine eager
#: whole-trace passes before it could draw a single frame.  Each is a pure
#: function of the trace, so computing them here costs the converter one pass
#: and costs the browser nothing.
#:
#: They are computed on the *undropped* interval sequence, before the per-level
#: threshold is applied.  That is what makes a surviving interval carry the same
#: percentile at every level — without it, an automatic level switch would
#: repaint the whole heatmap with different numbers.
#:
#: The order matters twice: the u8 run keeps total padding under 4 bytes per
#: block, and keeping (pct_enter, pct_leave) and (gap_pct_enter, gap_pct_leave)
#: adjacent means no metric ever needs more than two byte ranges per block.
DERIVED_U8_COLUMNS = (
    "pct_enter",        # percentile of this interval's start within its column
    "pct_leave",        # ... of its end
    "pct_dur_col",      # ... of its duration, within its column
    "pct_dur_region",   # ... of its duration among all calls of the same region, run-wide
    "gap_pct_enter",    # the same four, for the Comp gap that follows this interval
    "gap_pct_leave",
    "gap_pct_dur_col",
    "gap_pct_dur_flat",  # ... against every gap in the run, one flat pool
)
#: Full set, in layout order.  `iter_index` is u16 and `comp_prefix` is sized
#: from its own range; everything between them is u8.
DERIVED_COLUMNS = ("iter_index",) + DERIVED_U8_COLUMNS + ("comp_prefix",)

#: Regions whose completion ends a loop iteration, for `iter_index`.
ITERATION_REGIONS = ("MPI_Wait", "MPI_Waitall")

#: Largest number of counted events per cycle treated as physically possible,
#: used to spot a broken PAPI accumulation.  The real ceiling on x86 is 64
#: (AVX-512 single-precision FMA: 16 lanes x 2 flops x 2 ports); 256 leaves
#: generous margin while still catching a baseline shift by orders of magnitude.
MAX_PAPI_EVENTS_PER_CYCLE = 256

#: Slack added to that bound so a zero-duration interval is not flagged.
PAPI_STEP_SLACK = 1 << 20

#: Event class-name prefixes kept by --ignore-non-mpi-events.
MPI_EVENT_PREFIXES = ("Mpi", "Comm", "NonBlocking", "Enter", "Leave")

_NUMPY_DTYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64}
_DTYPE_BYTES = {"u8": 1, "u16": 2, "u32": 4, "u64": 8}


def dtype_for_max(max_value: int) -> str:
    """Smallest unsigned dtype name that can hold `max_value`."""
    if max_value <= 0xFF:
        return "u8"
    if max_value <= 0xFFFF:
        return "u16"
    if max_value <= 0xFFFF_FFFF:
        return "u32"
    return "u64"


def align_up(offset: int, alignment: int) -> int:
    """Round `offset` up to the next multiple of `alignment`."""
    return (offset + alignment - 1) // alignment * alignment


def snake(name: str) -> str:
    """Map an OTF2 `_fields` name (camelCase) to its instance accessor.

    Verified against every field of all 83 event classes in otf2 3.2, including
    the awkward ones: requestID -> request_id, msgTag -> msg_tag,
    collectiveOp -> collective_op, numberOfRequestedThreads -> ....
    """
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


# ─────────────────────────────────────────────────────────────────────────────
# §2  Command line
# ─────────────────────────────────────────────────────────────────────────────

_TIME_UNITS = {
    "": None,  # bare number = cycles
    "c": None,
    "cycles": None,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}


def parse_time_arg(text: str, timer_resolution: int) -> int:
    """Parse a block length given in cycles or with a us/ms/s suffix.

    Returns a whole number of cycles.  Raises ValueError below the spec minimum.
    """
    text = text.strip()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(µs|us|ms|s|c|cycles)?", text)
    if not m:
        raise ValueError("cannot parse time %r" % text)
    value, unit = float(m.group(1)), (m.group(2) or "")
    factor = _TIME_UNITS.get(unit, "bad")
    if factor == "bad":
        raise ValueError("unknown time unit %r" % unit)
    cycles = int(round(value if factor is None else value * factor * timer_resolution))
    if cycles < BASE_BLOCK_CYCLES:
        min_ms = BASE_BLOCK_CYCLES / (0.001 * timer_resolution)
        raise ValueError(
            "time block length %s = %d cycles is below the minimum of %d cycles "
            "(%.4f ms at this trace's timer resolution of %d Hz); try %.4gms or more"
            % (text, cycles, BASE_BLOCK_CYCLES, min_ms, timer_resolution,
               math.ceil(min_ms * 1000) / 1000)
        )
    return cycles


def parse_offset_arg(text: str, timer_resolution: int) -> int:
    """Parse a time *offset* in cycles or with a us/ms/s suffix.

    Same grammar as parse_time_arg() but without its minimum: that floor is the
    spec's minimum block *length*, which is the right rule for a block and the
    wrong one for "skip 3 ms into the trace".
    """
    text = text.strip()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(µs|us|ms|s|c|cycles)?", text)
    if not m:
        raise ValueError("cannot parse time %r" % text)
    value, unit = float(m.group(1)), (m.group(2) or "")
    factor = _TIME_UNITS.get(unit, "bad")
    if factor == "bad":
        raise ValueError("unknown time unit %r" % unit)
    return max(0, int(round(value if factor is None else value * factor * timer_resolution)))


def parse_loss_levels(text: str) -> tuple[int, ...]:
    """'1+2' -> (1, 2).  Validates that every part is 0, 1 or 2."""
    parts = [p.strip() for p in text.split("+")]
    levels = []
    for p in parts:
        if p not in ("0", "1", "2"):
            raise ValueError("loss level parts must be 0, 1 or 2, got %r" % p)
        levels.append(int(p))
    if not levels:
        raise ValueError("no loss level given")
    return tuple(sorted(set(levels)))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="otf2_to_pv3d.py",
        description="Convert an OTF2 trace to the PV3D block-fetchable columnar format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("trace_dir", help="OTF2 trace folder (or its traces.otf2 anchor file)")
    p.add_argument("--output", default=None,
                   help="output file path (default: derived from the trace folder name)")
    p.add_argument("--loss-level", default="1+2",
                   help="which loss levels to emit: 0, 1, 2, 0+1, 0+2, 1+2 or 0+1+2")
    p.add_argument("--papi-loss", choices=("yes", "no"), default="no",
                   help="divide the stored counter increments by the loss level's quantum. "
                        "Off by default: counters are stored as increments, not absolute "
                        "values, and dividing an increment by a *time* quantum destroys the "
                        "small ones outright — PAPI_DP_OPS gains at most 18 during an MPI "
                        "call on the project's traces, which //256 flattens to zero. The "
                        "space it was meant to save is already taken by sizing each column "
                        "from its own range")
    p.add_argument("--time-block-length", default="auto",
                   help="level-0 block length in cycles or with a us/ms/s suffix; "
                        "'auto' targets ~%d bytes per block" % TARGET_BLOCK_BYTES)
    p.add_argument("--ignore-non-mpi-events", action="store_true",
                   help="drop event classes outside Mpi*/Comm*/NonBlocking*/Enter/Leave, "
                        "and Enter/Leave whose region name does not start with 'MPI'")
    p.add_argument("--ignore-papi-presets", action="store_true",
                   help="do not store PAPI counters")
    p.add_argument("--ignore-non-enter-leave", action="store_true",
                   help="store only intervals; drop every other event type")
    p.add_argument("--region-list", default=None,
                   help="comma-separated region names, or @file with one name per line")
    p.add_argument("--region-list-mode", choices=("ignore", "keep"), default="ignore",
                   help="treat --region-list as a blacklist or a whitelist")
    p.add_argument("--begin-time", default=None,
                   help="skip this much past the trace's first interval (cycles, or with a "
                        "us/ms/s suffix); intervals ending before the cut are dropped and "
                        "one that straddles it is clamped. Replaces the viewer's own "
                        "'Begin time' field, which cannot work once percentiles are "
                        "precomputed against the full sequence")
    p.add_argument("--no-derived", action="store_true",
                   help="do not store the precomputed per-interval columns (percentiles, "
                        "iteration index, comp prefix); smaller file, but the viewer must "
                        "then compute them itself and cannot render block-at-a-time")
    p.add_argument("--format", choices=("bin", "txt"), default="bin",
                   help="write .pv3dbin or its exact .pv3dtxt mirror")
    # Implementation knobs
    p.add_argument("--workers", type=int, default=0,
                   help="reader processes (0 = auto, 1 = serial)")
    p.add_argument("--quantize-mode", choices=("absolute-floor", "delta-ceil"),
                   default="absolute-floor",
                   help="absolute-floor keeps the error bounded; delta-ceil reproduces "
                        "the original specification and drifts forward")
    p.add_argument("--ll1-drop-threshold", type=int, default=None,
                   help="override the level-1 short-interval threshold, in cycles")
    p.add_argument("--ll2-drop-threshold", type=int, default=None,
                   help="override the level-2 short-interval threshold, in cycles")
    p.add_argument("--span-duplicate-limit", type=int, default=4,
                   help="intervals crossing at most this many block boundaries are "
                        "duplicated into each block; longer ones go to the resident table")
    p.add_argument("--quiet", action="store_true", help="suppress the statistics report")
    return p


@dataclass
class Options:
    trace_dir: str
    anchor: str
    output: str
    levels: tuple[int, ...]
    papi_loss: bool
    block_length_arg: str
    ignore_non_mpi: bool
    ignore_papi: bool
    ignore_non_enter_leave: bool
    region_list: tuple[str, ...]
    region_list_mode: str
    begin_time_arg: str | None
    derived: bool
    fmt: str
    workers: int
    quantize_mode: str
    drop_thresholds: tuple[int, int, int]
    span_duplicate_limit: int
    quiet: bool

    def as_json(self) -> dict:
        d = asdict(self)
        d["levels"] = list(self.levels)
        d["region_list"] = list(self.region_list)
        d["drop_thresholds"] = list(self.drop_thresholds)
        return d


def resolve_options(argv: list[str] | None = None) -> Options:
    args = build_argparser().parse_args(argv)

    trace_dir = os.path.abspath(args.trace_dir)
    if os.path.isdir(trace_dir):
        anchor = os.path.join(trace_dir, "traces.otf2")
    else:
        anchor, trace_dir = trace_dir, os.path.dirname(trace_dir)
    if not os.path.isfile(anchor):
        sys.exit("error: no OTF2 anchor file at %s" % anchor)

    region_list: tuple[str, ...] = ()
    if args.region_list:
        if args.region_list.startswith("@"):
            with open(args.region_list[1:], encoding="utf-8") as fh:
                region_list = tuple(line.strip() for line in fh if line.strip())
        else:
            region_list = tuple(s.strip() for s in args.region_list.split(",") if s.strip())

    ext = ".pv3dbin" if args.format == "bin" else ".pv3dtxt"
    output = args.output or os.path.join(
        os.path.dirname(trace_dir) or ".", os.path.basename(trace_dir) + ext)
    if not output.endswith(ext):
        output += ext

    drops = list(LEVEL_DROP)
    if args.ll1_drop_threshold is not None:
        drops[1] = args.ll1_drop_threshold
    if args.ll2_drop_threshold is not None:
        drops[2] = args.ll2_drop_threshold

    return Options(
        trace_dir=trace_dir,
        anchor=anchor,
        output=output,
        levels=parse_loss_levels(args.loss_level),
        papi_loss=(args.papi_loss == "yes"),
        block_length_arg=args.time_block_length,
        ignore_non_mpi=args.ignore_non_mpi_events,
        ignore_papi=args.ignore_papi_presets,
        ignore_non_enter_leave=args.ignore_non_enter_leave,
        region_list=region_list,
        region_list_mode=args.region_list_mode,
        begin_time_arg=args.begin_time,
        derived=not args.no_derived,
        fmt=args.format,
        workers=args.workers,
        quantize_mode=args.quantize_mode,
        drop_thresholds=(drops[0], drops[1], drops[2]),
        span_duplicate_limit=max(0, args.span_duplicate_limit),
        quiet=args.quiet,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3  OTF2 schema introspection
#
# Every event class carries a `_fields` tuple of (camelCaseName, typeString,
# doc).  We turn that into a per-class list of parameters we know how to reduce
# to an integer, once, before any I/O.
# ─────────────────────────────────────────────────────────────────────────────

# How to reduce a parameter value to an int, by declared type.
PARAM_INT = "int"        # already an integer
PARAM_REF = "ref"        # definition object, use ._ref
PARAM_ENUM = "enum"      # enum member, use .value
PARAM_STR = "str"        # resolves to a Python str, use the string dictionary
PARAM_SKIP = "skip"      # array / attributes / anything we do not store inline

_REF_SUFFIX = "Ref"
_INT_TYPES = {"ctypes.c_uint8", "ctypes.c_uint32", "ctypes.c_uint64",
              "ctypes.c_int64", "ctypes.c_int32", "TimeStamp", "int"}
_ENUM_TYPES = {"CollectiveOp", "Paradigm", "LockType", "MeasurementMode",
               "RmaSyncLevel", "RmaSyncType", "RmaAtomicType", "IoAccessMode",
               "IoCreationFlag", "IoStatusFlag", "IoOperationMode",
               "IoOperationFlag", "IoSeekOption"}
_SIGNED_TYPES = {"ctypes.c_int64", "ctypes.c_int32"}


@dataclass(frozen=True)
class ParamSpec:
    otf2_name: str
    accessor: str
    decl_type: str
    kind: str
    signed: bool


@dataclass(frozen=True)
class EventSpec:
    name: str
    cls: type
    params: tuple[ParamSpec, ...]


def classify_param(decl_type: str) -> tuple[str, bool]:
    """Map an OTF2 declared type to (reduction kind, is_signed)."""
    if decl_type == "StringRef":
        return PARAM_STR, False
    if decl_type.endswith(_REF_SUFFIX):
        return PARAM_REF, False
    if decl_type in _ENUM_TYPES:
        return PARAM_ENUM, False
    if decl_type in _INT_TYPES:
        return PARAM_INT, decl_type in _SIGNED_TYPES
    # 'array', 'dict' (attributes) and anything unrecognised are not stored
    # inline; see PV3D_FORMAT.md for how ProgramBegin's argument list is kept.
    return PARAM_SKIP, False


def build_event_specs() -> dict[type, EventSpec]:
    """One EventSpec per OTF2 event class, keyed by the class for `is` dispatch."""
    specs: dict[type, EventSpec] = {}
    for name in dir(otf2_events):
        if name.startswith("_"):
            continue
        cls = getattr(otf2_events, name)
        if not inspect.isclass(cls):
            continue
        fields = getattr(cls, "_fields", None)
        if not fields:
            continue
        params = []
        for fname, ftype, *_ in fields:
            if fname in ("time", "attributes"):
                continue
            kind, signed = classify_param(ftype)
            if kind == PARAM_SKIP:
                continue
            params.append(ParamSpec(fname, snake(fname), ftype, kind, signed))
        specs[cls] = EventSpec(name, cls, tuple(params))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# §4  Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceMeta:
    timer_resolution: int
    global_offset: int
    trace_length: int
    location_refs: list[int]
    location_rows: dict[int, int]          # otf2 location ref -> dense row index
    location_info: list[dict]
    region_names: list[str]                # indexed by otf2 region ref
    papi_names: list[str]
    n_events_total: int
    rank_parse_warning: bool


def _mpi_rank_from_group(group_name: str, fallback: int) -> tuple[int, bool]:
    m = re.search(r"(\d+)\s*$", group_name or "")
    if m:
        return int(m.group(1)), False
    return fallback, True


def read_definitions(opts: Options) -> TraceMeta:
    """Harvest everything we need before touching the event streams."""
    with otf2.reader.open(opts.anchor) as trace:
        defs = trace.definitions
        clock = defs.clock_properties
        locations = list(defs.locations)

        location_refs, location_rows, location_info = [], {}, []
        warn = False
        for row, loc in enumerate(locations):
            ref = loc._ref
            group_name = getattr(loc.group, "name", "") if loc.group else ""
            rank, missed = _mpi_rank_from_group(group_name, row)
            warn = warn or missed
            location_refs.append(ref)
            location_rows[ref] = row
            location_info.append({
                "row": row,
                "otf2_ref": int(ref),
                "mpi_rank": int(rank),
                "name": str(loc.name),
                "group": str(group_name),
                "n_events": int(loc.number_of_events),
            })

        max_region = -1
        region_map: dict[int, str] = {}
        for reg in defs.regions:
            region_map[reg._ref] = str(reg.name)
            max_region = max(max_region, reg._ref)
        region_names = [region_map.get(i, "<region %d>" % i) for i in range(max_region + 1)]

        papi_names = [str(m.name) for m in defs.metric_members]

    return TraceMeta(
        timer_resolution=int(clock.timer_resolution),
        global_offset=int(clock.global_offset),
        trace_length=int(clock.trace_length),
        location_refs=location_refs,
        location_rows=location_rows,
        location_info=location_info,
        region_names=region_names,
        papi_names=papi_names,
        n_events_total=sum(i["n_events"] for i in location_info),
        rank_parse_warning=warn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §5  Reading
#
# The reader is CFFI-bound at ~250k events/s with only a few percent of
# headroom for Python work, so the hot loop does the minimum possible: unpack,
# read .time, dispatch on type(ev) identity, append to typed array.array
# accumulators (8 bytes per value, no per-value object overhead).  Raw event
# objects are never retained.
# ─────────────────────────────────────────────────────────────────────────────

class LocationBuffers:
    """Typed accumulators for one location's event stream."""

    __slots__ = ("el_time", "el_kind", "el_region", "papi_time", "papi_vals",
                 "other", "n_events", "skipped_arrays")

    def __init__(self) -> None:
        self.el_time = array("q")
        self.el_kind = array("b")      # +1 Enter, -1 Leave
        self.el_region = array("i")
        self.papi_time = array("q")
        self.papi_vals = array("q")    # flattened, n_counters per sample
        self.other: dict[str, tuple[array, list[array]]] = {}
        self.n_events = 0
        self.skipped_arrays: list[dict] = []


def read_locations(opts: Options, meta: TraceMeta, specs: dict[type, EventSpec],
                   location_subset: list[int]) -> dict[int, LocationBuffers]:
    """Read the given locations (by OTF2 ref) in a single pass over their files."""
    buffers: dict[int, LocationBuffers] = {ref: LocationBuffers() for ref in location_subset}
    want = set(location_subset)

    with otf2.reader.open(opts.anchor) as trace:
        locs = [l for l in trace.definitions.locations if l._ref in want]
        events = trace.events(locs) if len(locs) != len(trace.definitions.locations) \
            else trace.events

        for loc, ev in events:
            buf = buffers.get(loc._ref)
            if buf is None:
                continue
            cls = type(ev)
            buf.n_events += 1

            if cls is Enter:
                buf.el_time.append(ev.time)
                buf.el_kind.append(1)
                buf.el_region.append(ev.region._ref)
            elif cls is Leave:
                buf.el_time.append(ev.time)
                buf.el_kind.append(-1)
                buf.el_region.append(ev.region._ref)
            elif cls is Metric:
                if not opts.ignore_papi:
                    buf.papi_time.append(ev.time)
                    buf.papi_vals.extend(ev.values)
            else:
                if opts.ignore_non_enter_leave:
                    continue
                spec = specs.get(cls)
                if spec is None:
                    continue
                if opts.ignore_non_mpi and not spec.name.startswith(MPI_EVENT_PREFIXES):
                    continue
                slot = buf.other.get(spec.name)
                if slot is None:
                    slot = (array("q"), [array("q") for _ in spec.params])
                    buf.other[spec.name] = slot
                times, cols = slot
                times.append(ev.time)
                for i, p in enumerate(spec.params):
                    cols[i].append(_reduce_param(getattr(ev, p.accessor, None), p))
    return buffers


def plan_chunks(meta: TraceMeta, n_workers: int) -> list[list[int]]:
    """Split locations into `n_workers` chunks balanced by event count.

    `Location.number_of_events` is known from the definitions, before any event
    is read, so the split can be balanced up front rather than by location count.
    """
    if n_workers <= 1:
        return [list(meta.location_refs)]
    order = sorted(meta.location_info, key=lambda i: -i["n_events"])
    chunks: list[list[int]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for info in order:                       # longest-processing-time first
        k = loads.index(min(loads))
        chunks[k].append(info["otf2_ref"])
        loads[k] += info["n_events"]
    return [c for c in chunks if c]


#: Set in the parent before forking so workers inherit it without pickling.
_WORKER_CTX: dict = {}


def _worker_read(chunk: list[int]) -> dict:
    """Read and transform one chunk of locations in a forked worker."""
    opts, meta, specs = _WORKER_CTX["opts"], _WORKER_CTX["meta"], _WORKER_CTX["specs"]
    n_counters = _WORKER_CTX["n_counters"]
    buffers = read_locations(opts, meta, specs, chunk)
    anomalies = Anomalies()
    out_iv, out_other, events = {}, {}, 0
    for ref, buf in buffers.items():
        row = meta.location_rows[ref]
        events += buf.n_events
        out_iv[row] = build_intervals(buf, n_counters, anomalies)
        for name, (times, cols) in buf.other.items():
            out_other.setdefault(name, {})[row] = (
                np.frombuffer(times, dtype=np.int64).copy(),
                [np.frombuffer(c, dtype=np.int64).copy() for c in cols],
            )
    return {"intervals": out_iv, "other": out_other, "events": events,
            "anomalies": anomalies, "strings": list(_string_dict.items)}


def _reduce_param(value: Any, spec: ParamSpec) -> int:
    """Reduce one event parameter to an integer we can store in a column."""
    if value is None:
        return 0
    if spec.kind == PARAM_INT:
        return int(value)
    if spec.kind == PARAM_REF:
        return int(getattr(value, "_ref", 0))
    if spec.kind == PARAM_ENUM:
        return int(getattr(value, "value", 0))
    if spec.kind == PARAM_STR:
        return _string_dict.intern(str(value))
    return 0


class _StringDict:
    """Interning table for StringRef-typed parameters (e.g. ProgramBegin.programName)."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self._index: dict[str, int] = {}

    def intern(self, s: str) -> int:
        idx = self._index.get(s)
        if idx is None:
            idx = len(self.items)
            self._index[s] = idx
            self.items.append(s)
        return idx


_string_dict = _StringDict()

#: Event type name -> EventSpec, filled in by convert() so the serialiser can
#: recover each type's parameter names without threading the table through.
_spec_by_name: dict[str, "EventSpec"] = {}


# ─────────────────────────────────────────────────────────────────────────────
# §6  Per-location transform: pairing, PAPI alignment, filtering, blocks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Intervals:
    """Matched Enter/Leave pairs for one location, in start order."""
    start: np.ndarray     # int64 absolute cycles
    end: np.ndarray       # int64 absolute cycles
    region: np.ndarray    # int64 otf2 region ref
    depth: np.ndarray     # int64 nesting depth
    #: Counter increments, (n, n_counters) int64, or None when PAPI is off.
    #: `papi_call[j]` accumulates over interval j itself; `papi_gap[j]` over the
    #: Comp gap that follows it (0 for a rank's last interval).
    #:
    #: Differences rather than the absolute samples, because an absolute PAPI
    #: value is a per-core total since boot: it differs by billions between
    #: ranks, so a block-relative encoding of it overflows u32 and wraps, while
    #: the increments are small and exact.  Nothing downstream wants the
    #: absolute value — every metric is a rate or a ratio of increments.
    papi_call: np.ndarray | None
    papi_gap: np.ndarray | None
    #: name -> int64 array, one entry per interval; see DERIVED_COLUMNS.
    #: Empty when --no-derived.  Filled by compute_derived() once the interval
    #: set is final, and carried unchanged through filtering and block placement
    #: so that the value always describes the whole trace, not one level of it.
    derived: dict = field(default_factory=dict)


@dataclass
class Anomalies:
    unmatched: int = 0
    papi_misaligned: int = 0
    #: Counter increments spanning a break in the accumulation (a reset, an
    #: overflow, or a baseline shift such as the first sample of a Score-P
    #: trace sitting on a different base).  Stored as 0.
    papi_broken_step: int = 0
    negative_duration: int = 0
    papi_negative: int = 0
    #: Pairs whose Leave carried a different region ref than its Enter.  Normal
    #: for compiler-instrumented traces; informational only.
    region_mismatch: int = 0


def pair_intervals(el_time: np.ndarray, el_kind: np.ndarray, el_region: np.ndarray,
                   anomalies: Anomalies) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match Enter/Leave events into intervals, vectorised.

    `level = cumsum(kind) - (kind == +1)` gives each event the nesting depth of
    the region it opens or closes.  Within one depth, events alternate
    Enter, Leave, Enter, Leave..., so sorting by (depth, original index) puts
    every Enter at an even position and its matching Leave immediately after.

    Returns (enter_index, leave_index, depth) into the original event arrays.
    Verified bit-identical to a sequential stack on the project's traces,
    including nesting depth 9.
    """
    if el_time.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, z

    depth = np.cumsum(el_kind) - (el_kind == 1)
    order = np.lexsort((np.arange(el_time.size), depth))
    sorted_depth = depth[order]

    enter_idx, leave_idx, out_depth = [], [], []
    # Each depth forms a contiguous run in `order`; within it, pair adjacent.
    boundaries = np.flatnonzero(np.diff(sorted_depth)) + 1
    for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, sorted_depth.size]):
        run = order[lo:hi]
        if run.size % 2:
            anomalies.unmatched += 1
            run = run[:run.size // 2 * 2]
        if run.size == 0:
            continue
        e, l = run[0::2], run[1::2]
        # The bracket structure is authoritative: guard that every even position
        # really is an Enter and every odd one a Leave, and drop only pairs that
        # violate that.
        ok = (el_kind[e] == 1) & (el_kind[l] == -1)
        if not ok.all():
            anomalies.unmatched += int((~ok).sum())
            e, l = e[ok], l[ok]
        # A Leave whose region ref differs from its Enter's is common in
        # compiler-instrumented traces (measured: 15.9% of pairs on walberla,
        # where the stream is nonetheless perfectly balanced — 912,609 pairs,
        # no underflow, depth returning to 0).  The interval's region is the one
        # that was *entered*, so this is recorded as a statistic, not a defect,
        # and never causes an interval to be dropped.
        if e.size:
            anomalies.region_mismatch += int((el_region[e] != el_region[l]).sum())
        enter_idx.append(e)
        leave_idx.append(l)
        out_depth.append(np.full(e.size, sorted_depth[lo], dtype=np.int64))

    if not enter_idx:
        z = np.empty(0, dtype=np.int64)
        return z, z, z
    return (np.concatenate(enter_idx), np.concatenate(leave_idx),
            np.concatenate(out_depth))


def build_intervals(buf: LocationBuffers, n_counters: int,
                    anomalies: Anomalies) -> Intervals:
    """Turn one location's raw buffers into start-ordered intervals with PAPI."""
    el_time = np.frombuffer(buf.el_time, dtype=np.int64)
    el_kind = np.frombuffer(buf.el_kind, dtype=np.int8).astype(np.int64)
    el_region = np.frombuffer(buf.el_region, dtype=np.int32).astype(np.int64)

    e_idx, l_idx, depth = pair_intervals(el_time, el_kind, el_region, anomalies)

    start = el_time[e_idx] if e_idx.size else np.empty(0, np.int64)
    end = el_time[l_idx] if l_idx.size else np.empty(0, np.int64)
    region = el_region[e_idx] if e_idx.size else np.empty(0, np.int64)

    bad = end < start
    if bad.any():
        anomalies.negative_duration += int(bad.sum())
        keep = ~bad
        start, end, region, depth = start[keep], end[keep], region[keep], depth[keep]
        e_idx, l_idx = e_idx[keep], l_idx[keep]

    papi_call = papi_gap = None
    sample_of_event = None
    papi_vals = None
    if n_counters and len(buf.papi_time):
        papi_time = np.frombuffer(buf.papi_time, dtype=np.int64)
        papi_vals = np.frombuffer(buf.papi_vals, dtype=np.int64)
        if papi_vals.size == papi_time.size * n_counters:
            papi_vals = papi_vals.reshape(-1, n_counters)
            # A Metric event immediately precedes every Enter/Leave with an
            # identical timestamp, so the two streams are 1:1 in order.  Verify
            # rather than assume; fall back to timestamp matching if not.
            if papi_time.size == el_time.size and np.array_equal(papi_time, el_time):
                sample_of_event = np.arange(el_time.size)
            else:
                anomalies.papi_misaligned += 1
                sample_of_event = np.clip(
                    np.searchsorted(papi_time, el_time, side="right") - 1,
                    0, papi_time.size - 1)
        else:
            papi_vals = None

    # Sort by start time so every downstream step can assume start order.
    if start.size:
        order = np.argsort(start, kind="stable")
        start, end, region, depth = start[order], end[order], region[order], depth[order]
        e_idx, l_idx = e_idx[order], l_idx[order]

    if papi_vals is not None and start.size:
        papi_call, papi_gap = _papi_deltas(papi_vals, papi_time, sample_of_event,
                                           e_idx, l_idx, anomalies)

    return Intervals(start, end, region, depth, papi_call, papi_gap)


def _papi_deltas(papi_vals: np.ndarray, papi_time: np.ndarray,
                 sample_of_event: np.ndarray, e_idx: np.ndarray,
                 l_idx: np.ndarray, anomalies: Anomalies) -> tuple[np.ndarray, np.ndarray]:
    """Counter increments per interval and per following gap, in start order.

    Counters are declared ACCUMULATED_START, i.e. monotonic, so a step that goes
    backwards or that implies an impossible event rate is not data: it is a
    reset, a wrap, or a baseline shift.  Score-P writes one of the latter on some
    traces — the first Metric sample of every location sits on a different base
    than the rest, a jump of ~2**47.  Rather than special-casing "sample 0",
    every such break is located once at the sample level and any increment that
    straddles one is stored as 0 and counted.
    """
    step = np.diff(papi_vals, axis=0)                       # (n_samples-1, n_counters)
    # No counter can advance faster than the clock allows, so the wall time
    # between two samples bounds every step between them.
    bound = (np.maximum(np.diff(papi_time), 0) * MAX_PAPI_EVENTS_PER_CYCLE
             + PAPI_STEP_SLACK)
    broken = (step < 0).any(axis=1) | (step > bound[:, None]).any(axis=1)

    # breaks_before[k] = number of broken transitions strictly before sample k,
    # so an increment from sample a to sample b is sound iff the two counts match.
    breaks_before = np.r_[0, np.cumsum(broken.astype(np.int64))]

    def delta(a, b):
        out = papi_vals[b] - papi_vals[a]
        bad = breaks_before[b] != breaks_before[a]
        if bad.any():
            anomalies.papi_broken_step += int(bad.sum())
            out = np.where(bad[:, None], 0, out)
        return out

    s_of = sample_of_event
    call = delta(s_of[e_idx], s_of[l_idx])

    # The gap after interval j runs from j's Leave to j+1's Enter; a rank's last
    # interval has no gap, which FLAG_LAST_INTERVAL also records.
    gap = np.zeros_like(call)
    if call.shape[0] > 1:
        gap[:-1] = delta(s_of[l_idx[:-1]], s_of[e_idx[1:]])
    return call, gap


# ─────────────────────────────────────────────────────────────────────────────
# §7  Level assembly
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# §6b  Derived columns
#
# The viewer used to compute all of this itself, eagerly, over the whole trace,
# before the first frame — nine passes producing per-interval THREE.Color tables
# several times the size of the trace.  Computing it here instead is what lets
# the browser fetch one block and draw.
# ─────────────────────────────────────────────────────────────────────────────

def _group_percentile(group: np.ndarray, value: np.ndarray,
                      tiebreak: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each element within its group.

    Deliberately identical to the viewer's computeColumnPercentiles() /
    computeFlatPercentiles(): a stable sort by value with ties in ascending
    `tiebreak` (which is the rank index, or the flat position, exactly as the
    JavaScript enumerated them), `position / (n - 1) * 100`, and 100 for a group
    of one.  Vectorised as one lexsort rather than a sort per group, so the cost
    is a single O(N log N) pass over the trace instead of one per column.
    """
    n = group.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.lexsort((tiebreak, value, group))
    g = group[order]
    counts = np.bincount(group)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pos = np.arange(n, dtype=np.int64) - starts[g]
    sizes = counts[g]
    pct = np.where(sizes > 1, pos / np.maximum(sizes - 1, 1) * 100.0, 100.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = pct
    return out


def _to_u8_percent(pct: np.ndarray) -> np.ndarray:
    """0-100 float -> 0-255 int.  The viewer multiplies back by 100/255.

    Costs +-0.2 percentile points, which can flip a bucket only for a value
    within 0.2 of a user cutoff; a float column would double the file for that.
    """
    return np.rint(pct * (255.0 / 100.0)).astype(np.int64)


def compute_derived(intervals: dict[int, Intervals], meta: TraceMeta) -> dict:
    """Fill `iv.derived` for every location.  Returns a stats dict.

    Every column is computed across *all* locations at once, because that is
    what the quantities mean: "this call's start, ranked against the same call
    on every other rank".  Interval index within a rank is the matching key —
    the standard case for SPMD codes where ranks execute symmetric sequences.
    """
    rows = sorted(intervals)
    n_per = np.array([intervals[r].start.size for r in rows], dtype=np.int64)
    total = int(n_per.sum())
    stats = {"intervals": total, "max_iter_index": 0, "max_depth": 0,
             "deep_rows": 0, "iter_index_clipped": 0}
    if total == 0:
        return stats

    offsets = np.r_[0, np.cumsum(n_per)]
    rank_flat = np.repeat(np.arange(len(rows), dtype=np.int64), n_per)
    #: Position of each interval within its own rank — the "column" that makes
    #: one call comparable across ranks.
    col_flat = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], n_per)

    start = np.concatenate([intervals[r].start for r in rows])
    end = np.concatenate([intervals[r].end for r in rows])
    region = np.concatenate([intervals[r].region for r in rows])
    depth = np.concatenate([intervals[r].depth for r in rows])
    dur = end - start

    stats["max_depth"] = int(depth.max())
    stats["deep_rows"] = int((depth > 0).sum())

    out: dict[str, np.ndarray] = {}

    # --- per-column percentiles of start, end and duration --------------------
    out["pct_enter"] = _to_u8_percent(_group_percentile(col_flat, start, rank_flat))
    out["pct_leave"] = _to_u8_percent(_group_percentile(col_flat, end, rank_flat))
    out["pct_dur_col"] = _to_u8_percent(_group_percentile(col_flat, dur, rank_flat))

    # --- duration percentile among every call of the same region, run-wide ----
    # Ties break by flat position, i.e. rank-major then interval order, which is
    # the order the viewer's own flat pool was built in.
    _, region_dense = np.unique(region, return_inverse=True)
    flat_index = np.arange(total, dtype=np.int64)
    out["pct_dur_region"] = _to_u8_percent(
        _group_percentile(region_dense.astype(np.int64), dur, flat_index))

    # --- the same four for the Comp gap that *follows* each interval ---------
    # Gap g on a rank runs from interval g's end to interval g+1's start, so a
    # rank with N intervals has N-1 of them and its last interval has none —
    # which is what FLAG_LAST_INTERVAL records.
    has_gap = col_flat < (n_per[rank_flat] - 1)
    gsel = np.flatnonzero(has_gap)
    # What actually breaks these columns is *overlap*, not nesting depth. A gap column is the
    # time between one interval's end and the next one's start; if the next interval begins
    # before this one ends, that "gap" is negative and the ranking is meaningless. Depth alone
    # does not tell you: on a compiler-instrumented C++ trace filtered to MPI calls, every
    # surviving interval reports a depth in the thousands and yet none of them overlap.
    if gsel.size:
        stats["overlapping"] = int((start[gsel + 1] < end[gsel]).sum())
        stats["gap_pairs"] = int(gsel.size)
    for name in ("gap_pct_enter", "gap_pct_leave", "gap_pct_dur_col", "gap_pct_dur_flat"):
        out[name] = np.zeros(total, dtype=np.int64)
    if gsel.size:
        gap_from = end[gsel]
        gap_to = start[gsel + 1]          # same rank by construction of has_gap
        gap_dur = gap_to - gap_from
        gap_col = col_flat[gsel]
        gap_rank = rank_flat[gsel]
        gap_pos = np.arange(gsel.size, dtype=np.int64)
        out["gap_pct_enter"][gsel] = _to_u8_percent(_group_percentile(gap_col, gap_from, gap_rank))
        out["gap_pct_leave"][gsel] = _to_u8_percent(_group_percentile(gap_col, gap_to, gap_rank))
        out["gap_pct_dur_col"][gsel] = _to_u8_percent(_group_percentile(gap_col, gap_dur, gap_rank))
        out["gap_pct_dur_flat"][gsel] = _to_u8_percent(
            _group_percentile(np.zeros(gsel.size, dtype=np.int64), gap_dur, gap_pos))

    # --- iteration index: completed MPI_Wait/MPI_Waitall calls before this one -
    wait_refs = [ref for ref, name in enumerate(meta.region_names)
                 if name in ITERATION_REGIONS]
    is_wait = np.isin(region, wait_refs).astype(np.int64) if wait_refs \
        else np.zeros(total, dtype=np.int64)
    inc = np.cumsum(is_wait)
    excl = inc - is_wait
    rank_first = np.clip(offsets[:-1], 0, total - 1)
    iter_index = excl - excl[rank_first][rank_flat]
    stats["max_iter_index"] = int(iter_index.max())
    if stats["max_iter_index"] > 0xFFFF:
        stats["iter_index_clipped"] = int((iter_index > 0xFFFF).sum())
        iter_index = np.minimum(iter_index, 0xFFFF)
    out["iter_index"] = iter_index

    # --- cumulative comp (non-MPI) time before each interval's start ----------
    # Measured from the trace's first interval, so it matches the viewer's
    # timeline zero.  Stored in cycles here and quantised per level on the way
    # out, since a level's quantum is not known yet.
    t0_cycles = int(start.min())
    prev_end = np.where(col_flat == 0, t0_cycles, np.r_[0, end[:-1]])
    step = np.maximum(0, start - prev_end)
    inc2 = np.cumsum(step)
    excl2 = inc2 - step
    out["comp_prefix"] = inc2 - excl2[rank_first][rank_flat]

    # --- hand the columns back to their locations ----------------------------
    for name, col in out.items():
        pieces = np.split(col, offsets[1:-1]) if len(rows) > 1 else [col]
        for i, r in enumerate(rows):
            intervals[r].derived[name] = pieces[i]
    return stats


@dataclass
class LevelRows:
    """Flat, unsorted rows for one loss level, gathered across all locations."""
    block: list = field(default_factory=list)
    rank: list = field(default_factory=list)
    start: list = field(default_factory=list)
    end: list = field(default_factory=list)
    region: list = field(default_factory=list)
    depth: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    papi_call: list = field(default_factory=list)
    papi_gap: list = field(default_factory=list)
    long_papi_call: list = field(default_factory=list)
    long_papi_gap: list = field(default_factory=list)
    #: name -> list of arrays, parallel to `start`; see DERIVED_COLUMNS.
    derived: dict = field(default_factory=dict)
    # resident tier
    long_rank: list = field(default_factory=list)
    long_start: list = field(default_factory=list)
    long_end: list = field(default_factory=list)
    long_region: list = field(default_factory=list)
    long_depth: list = field(default_factory=list)
    long_flags: list = field(default_factory=list)
    long_derived: dict = field(default_factory=dict)
    # point events: type name -> dict of flat arrays
    other: dict = field(default_factory=dict)
    # bookkeeping
    n_source: int = 0
    n_dropped: int = 0
    n_duplicated: int = 0

    def concat(self, name: str, dtype=np.int64) -> np.ndarray:
        parts = getattr(self, name)
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    def concat_derived(self, store: str, name: str) -> np.ndarray:
        parts = getattr(self, store).get(name)
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts).astype(np.int64, copy=False)


def _place_other_events(other_events: dict, level: int, meta: TraceMeta,
                        block_len_q: int, n_blocks: int) -> dict:
    """Assign every non-Enter/Leave event to a block, quantised for this level.

    Point events cannot span a block, so there is no clipping or resident tier;
    they are simply bucketed.  Nothing is ever dropped, only coarsened.
    """
    quantum = LEVEL_QUANT[level]
    out: dict[str, dict] = {}
    for name, per_loc in other_events.items():
        blocks, ranks, times, params = [], [], [], None
        for row, (t_raw, cols) in per_loc.items():
            if t_raw.size == 0:
                continue
            tq = quantize(t_raw.copy(), meta.global_offset, quantum)
            b = np.minimum(tq // block_len_q, n_blocks - 1)
            blocks.append(b)
            ranks.append(np.full(t_raw.size, row, dtype=np.int64))
            times.append(tq - b * block_len_q)
            if params is None:
                params = [[] for _ in cols]
            for i, c in enumerate(cols):
                params[i].append(c)
        if not blocks:
            continue
        out[name] = {
            "block": np.concatenate(blocks),
            "rank": np.concatenate(ranks),
            "time": np.concatenate(times),
            "params": [np.concatenate(p) for p in (params or [])],
        }
    return out


def quantize(values: np.ndarray, global_offset: int, quantum: int) -> np.ndarray:
    """Absolute-floor quantisation: (t - global_offset) // Q.

    Unlike ceil(delta/Q) this keeps the error bounded by one quantum absolutely
    instead of accumulating it forward, and it keeps PAPI counters at their true
    magnitude (a counter rising by 1 per sample would otherwise be inflated Q-fold).
    """
    rel = values - global_offset
    np.maximum(rel, 0, out=rel)
    return rel // quantum if quantum > 1 else rel


def add_location_to_level(rows: LevelRows, iv: Intervals, rank: int, level: int,
                          meta: TraceMeta, opts: Options, block_len_q: int,
                          n_blocks: int) -> None:
    """Filter, quantise and place one location's intervals into a level."""
    if iv.start.size == 0:
        return
    quantum = LEVEL_QUANT[level]
    threshold = opts.drop_thresholds[level]
    rows.n_source += iv.start.size

    start, end = iv.start, iv.end
    region, depth = iv.region, iv.depth
    papi_call, papi_gap = iv.papi_call, iv.papi_gap
    derived = iv.derived

    if threshold > 0:
        keep = (end - start) >= threshold
        dropped = int((~keep).sum())
        if dropped:
            rows.n_dropped += dropped
            start, end, region, depth = start[keep], end[keep], region[keep], depth[keep]
            if papi_call is not None:
                # Filtered, not recomputed: the gap increment then describes the
                # gap to the next interval in the *full* sequence, not the next
                # visible one.  Same trade as the gap_pct_* columns, for the same
                # reason — recomputing would make a level's values incomparable
                # with every other level's.
                papi_call, papi_gap = papi_call[keep], papi_gap[keep]
            # Filtered, never recomputed: a surviving interval must carry the
            # same percentile at every level, or an automatic level switch would
            # repaint the heatmap with different numbers for the same calls.
            derived = {k: v[keep] for k, v in derived.items()}
        if start.size == 0:
            return

    #: Which surviving row is this rank's last — the gap columns describe the
    #: Comp period after an interval, and the last one has none.
    last_row = start.size - 1

    sq = quantize(start, meta.global_offset, quantum)
    eq = quantize(end, meta.global_offset, quantum)
    if opts.quantize_mode == "delta-ceil" and quantum > 1:
        # Spec-literal alternative: round each interval's extent up.
        eq = sq + -(-(end - start) // quantum)

    b0 = np.minimum(sq // block_len_q, n_blocks - 1)
    b1 = np.minimum(eq // block_len_q, n_blocks - 1)
    span = b1 - b0

    local = span == 0
    short = (span > 0) & (span <= opts.span_duplicate_limit)
    long_ = span > opts.span_duplicate_limit

    def emit(sel_block, sel_row, clip_left, clip_right):
        if sel_row.size == 0:
            return
        blk_start = sel_block * block_len_q
        s = np.maximum(sq[sel_row] - blk_start, 0)
        e = np.minimum(eq[sel_row] - blk_start, block_len_q - 1)
        f = (clip_left * FLAG_CLIPPED_LEFT + clip_right * FLAG_CLIPPED_RIGHT
             + (sel_row == last_row) * FLAG_LAST_INTERVAL)
        rows.block.append(sel_block)
        rows.rank.append(np.full(sel_row.size, rank, dtype=np.int64))
        rows.start.append(s)
        rows.end.append(e)
        rows.region.append(region[sel_row])
        rows.depth.append(depth[sel_row])
        rows.flags.append(f.astype(np.int64))
        # Every clipped copy of a spanning interval inherits the whole
        # interval's derived values: they describe the call, not the fragment.
        for name, col in derived.items():
            rows.derived.setdefault(name, []).append(col[sel_row])
        if papi_call is not None:
            rows.papi_call.append(papi_call[sel_row])
            rows.papi_gap.append(papi_gap[sel_row])

    # Block-local intervals: one row, no clipping.
    idx = np.flatnonzero(local)
    if idx.size:
        emit(b0[idx], idx, np.zeros(idx.size, np.int64), np.zeros(idx.size, np.int64))

    # Short-spanning intervals: duplicated, clipped, into every block they touch.
    idx = np.flatnonzero(short)
    if idx.size:
        counts = (span[idx] + 1).astype(np.int64)
        rep_row = np.repeat(idx, counts)
        offsets = np.arange(counts.sum()) - np.repeat(
            np.r_[0, np.cumsum(counts)[:-1]], counts)
        rep_block = b0[rep_row] + offsets
        emit(rep_block, rep_row,
             (rep_block > b0[rep_row]).astype(np.int64),
             (rep_block < b1[rep_row]).astype(np.int64))
        rows.n_duplicated += int(counts.sum() - idx.size)

    # Long-spanning intervals: stored once in the resident table.
    idx = np.flatnonzero(long_)
    if idx.size:
        rows.long_rank.append(np.full(idx.size, rank, dtype=np.int64))
        rows.long_start.append(sq[idx])
        rows.long_end.append(eq[idx])
        rows.long_region.append(region[idx])
        rows.long_depth.append(depth[idx])
        rows.long_flags.append(((idx == last_row) * FLAG_LAST_INTERVAL).astype(np.int64))
        for name, col in derived.items():
            rows.long_derived.setdefault(name, []).append(col[idx])
        if papi_call is not None:
            rows.long_papi_call.append(papi_call[idx])
            rows.long_papi_gap.append(papi_gap[idx])


# ─────────────────────────────────────────────────────────────────────────────
# §8  Binary writer
# ─────────────────────────────────────────────────────────────────────────────

class BlockLayout:
    """Deterministic byte layout of one block, derived from the level schema.

    A reader reproduces this from `n_rows`, `n_ranks` and the column schema
    alone; no per-block offset table is stored.  The order is fixed:

        rank_offsets, core interval columns      <- everything up to core_bytes
        derived columns                          <- ... up to interval_bytes
        PAPI columns (two per counter: the increment over the call and over
          the gap after it; one entry per interval each)
        u32 section_mask[ceil(n_sections / 32)]
        other-event sections, only the non-empty ones, in schema order

    so a viewer that only draws regions fetches the first `core_bytes` of a
    block, adds one derived column when a metric needs it, and never pays for
    counters or message events.

    Two things are elided rather than written as zeros.  An empty block writes
    nothing at all — not even its rank_offsets, which alone cost 4*(n_ranks+1)
    bytes — and an empty other-event section is skipped, its absence recorded in
    `section_mask`.  Measured on the 576-rank test trace that removed 1.5 MB of
    zeros from a 24 MB file (59 of 72 blocks were empty, at ~25 KB each), and it
    scales with block count: the same waste is ~250 MB at 10,000 blocks.
    """

    def __init__(self, n_ranks: int, n_rows: int, columns: list[dict],
                 papi_columns: list[dict], other_sections: list[dict]):
        self.offsets: dict[str, int] = {}
        off = 0
        core = 0
        if n_rows:
            off = 4 * (n_ranks + 1)                  # rank_offsets u32[n_ranks+1]
            core = off
            for col in columns:
                size = _DTYPE_BYTES[col["dtype"]]
                off = align_up(off, size)
                self.offsets[col["name"]] = off
                off += size * n_rows
                if not col.get("derived"):
                    core = off
        #: Bytes through `flags` — enough to draw regions, without the derived
        #: columns.  Recorded per block in the directory.
        self.core_bytes = core
        self.interval_bytes = off
        if n_rows:
            for col in papi_columns:
                size = _DTYPE_BYTES[col["dtype"]]
                off = align_up(off, size)
                self.offsets[col["name"]] = off
                off += size * n_rows
        #: Where the section bitmask and then the sections begin.
        self.total_before_sections = off
        n_sec = len(other_sections)
        self.mask_words = (n_sec + 31) // 32 if n_sec else 0
        self.mask_offset = None
        self.present = [bool(sec["n_rows"]) for sec in other_sections]
        if self.mask_words:
            off = align_up(off, 4)
            self.mask_offset = off
            off += 4 * self.mask_words
        # One self-contained section per non-empty other-event type.
        for sec, present in zip(other_sections, self.present):
            if not present:
                continue
            off = align_up(off, 4)
            self.offsets[sec["name"] + ".rank_offsets"] = off
            off += 4 * (n_ranks + 1)
            for col in sec["columns"]:
                size = _DTYPE_BYTES[col["dtype"]]
                off = align_up(off, size)
                self.offsets["%s.%s" % (sec["name"], col["name"])] = off
                off += size * sec["n_rows"]
        self.total = align_up(off, 4)

    def mask_value(self) -> list[int]:
        """The section bitmask as `mask_words` u32 values, bit i = section i present."""
        words = [0] * self.mask_words
        for i, present in enumerate(self.present):
            if present:
                words[i // 32] |= 1 << (i % 32)
        return words


def write_column(out: bytearray, base: int, offset: int, values: np.ndarray, dtype: str) -> None:
    """Place one column at `base + offset`, zero-padding any alignment gap."""
    pos = base + offset
    if len(out) < pos:
        out.extend(b"\0" * (pos - len(out)))
    data = np.ascontiguousarray(values, dtype=_NUMPY_DTYPES[dtype]).tobytes()
    out[pos:pos + len(data)] = data


# ─────────────────────────────────────────────────────────────────────────────
# §9  Driver
# ─────────────────────────────────────────────────────────────────────────────

def choose_block_len_q(n_intervals: int, trace_length_q: int, quant: int,
                       bytes_per_interval: float = 12.0) -> int:
    """Pick a block length, in quanta, targeting ~TARGET_BLOCK_BYTES per block.

    Returned as a power-of-two multiple of BASE_BLOCK_CYCLES where possible, but
    allowed to go *below* it at coarser levels: a level-2 block is nominally
    65536x longer than a level-0 one, which on a short trace exceeds the whole
    run and collapses the level into a single unfetchable block.  The two hard
    bounds are kept:

      * block_len_q * quant >= BASE_BLOCK_CYCLES  — the spec's minimum block
        length in cycles still holds at every level;
      * block_len_q <= BASE_BLOCK_CYCLES * MAX_BLOCK_MULTIPLIER — so that
        block-relative quantised times still fit u32.

    Bigger blocks also mean a smaller block directory and fewer duplicated
    spanning rows; smaller blocks mean less transferred per seek.
    """
    lo = max(1, BASE_BLOCK_CYCLES // max(1, quant))
    hi = BASE_BLOCK_CYCLES * MAX_BLOCK_MULTIPLIER
    if n_intervals <= 0:
        return min(max(BASE_BLOCK_CYCLES, lo), hi)
    ideal_blocks = max(1.0, n_intervals * bytes_per_interval / TARGET_BLOCK_BYTES)
    ideal = max(1.0, trace_length_q / ideal_blocks)
    p2 = 1 << max(0, int(round(math.log2(ideal))))
    return int(min(max(p2, lo), hi))


def convert(opts: Options) -> dict:
    t_start = time.time()
    specs = build_event_specs()
    _spec_by_name.clear()
    _spec_by_name.update({s.name: s for s in specs.values()})
    meta = read_definitions(opts)
    t_defs = time.time()

    n_ranks = len(meta.location_info)
    n_counters = 0 if opts.ignore_papi else len(meta.papi_names)

    # ---- read every location, then pair into intervals -------------------
    # `trace.events(subset)` seeks straight to those locations' event files, so
    # disjoint subsets are embarrassingly parallel.  A reader can only be given
    # its location set once, so each worker opens its own.
    anomalies = Anomalies()
    intervals: dict[int, Intervals] = {}
    other_events: dict[str, dict[int, tuple[np.ndarray, list[np.ndarray]]]] = {}
    events_read = 0

    n_workers = opts.workers or min(os.cpu_count() or 1, max(1, n_ranks))
    chunks = plan_chunks(meta, n_workers)
    use_pool = len(chunks) > 1 and hasattr(os, "fork")

    if use_pool:
        import multiprocessing as mp
        _WORKER_CTX.update({"opts": opts, "meta": meta, "specs": specs,
                            "n_counters": n_counters})
        try:
            with mp.get_context("fork").Pool(len(chunks)) as pool:
                results = pool.map(_worker_read, chunks)
        except Exception as exc:                      # fall back rather than fail
            print("warning: parallel read failed (%s); falling back to serial" % exc,
                  file=sys.stderr)
            use_pool = False
        else:
            by_name = {s.name: s for s in specs.values()}
            for res in results:
                intervals.update(res["intervals"])
                events_read += res["events"]
                # Each worker interned strings into its own table, so its stored
                # indices are worker-local; merge the tables and remap.
                remap = np.array([_string_dict.intern(s) for s in res["strings"]],
                                 dtype=np.int64) if res["strings"] else None
                for name, per_loc in res["other"].items():
                    spec = by_name.get(name)
                    str_cols = [i for i, p in enumerate(spec.params)
                                if p.kind == PARAM_STR] if spec else []
                    if remap is not None and str_cols:
                        for _row, (_t, cols) in per_loc.items():
                            for ci in str_cols:
                                if ci < len(cols) and cols[ci].size:
                                    cols[ci] = remap[cols[ci]]
                    other_events.setdefault(name, {}).update(per_loc)
                # Sum every counter by reflection rather than by name: the
                # hand-written list this replaced silently dropped any field
                # added later, so a new anomaly would read as zero on every
                # parallel run and only appear in the serial fallback.
                a = res["anomalies"]
                for f in fields(Anomalies):
                    setattr(anomalies, f.name,
                            getattr(anomalies, f.name) + getattr(a, f.name))

    if not use_pool:
        buffers = read_locations(opts, meta, specs, meta.location_refs)
        for ref, buf in buffers.items():
            row = meta.location_rows[ref]
            events_read += buf.n_events
            intervals[row] = build_intervals(buf, n_counters, anomalies)
            for name, (times, cols) in buf.other.items():
                other_events.setdefault(name, {})[row] = (
                    np.frombuffer(times, dtype=np.int64),
                    [np.frombuffer(c, dtype=np.int64) for c in cols],
                )
        buffers.clear()
    t_read = time.time()

    # ---- region filtering and re-indexing --------------------------------
    keep_region = _region_filter(meta, opts)
    if keep_region is not None:
        for row, iv in intervals.items():
            mask = keep_region[iv.region]
            if not mask.all():
                intervals[row] = Intervals(
                    iv.start[mask], iv.end[mask], iv.region[mask], iv.depth[mask],
                    None if iv.papi_call is None else iv.papi_call[mask],
                    None if iv.papi_gap is None else iv.papi_gap[mask])

    # ---- begin-time cut ---------------------------------------------------
    # Moved here from the viewer: it drops intervals, and every derived column
    # below is a ranking over the intervals that exist, so the two cannot both
    # be true unless the cut happens first.
    begin_cut_cycles = 0
    begin_dropped = 0
    if opts.begin_time_arg:
        offset = parse_offset_arg(opts.begin_time_arg, meta.timer_resolution)
        firsts = [int(iv.start.min()) for iv in intervals.values() if iv.start.size]
        if firsts and offset > 0:
            begin_cut_cycles = min(firsts) + offset
            for row, iv in intervals.items():
                keep = iv.end > begin_cut_cycles
                begin_dropped += int((~keep).sum())
                intervals[row] = Intervals(
                    np.maximum(iv.start[keep], begin_cut_cycles), iv.end[keep],
                    iv.region[keep], iv.depth[keep],
                    None if iv.papi_call is None else iv.papi_call[keep],
                    None if iv.papi_gap is None else iv.papi_gap[keep])
            # Point events before the cut would quantise to a negative block.
            for _name, per_loc in other_events.items():
                for row, (t_raw, cols) in list(per_loc.items()):
                    m = t_raw >= begin_cut_cycles
                    if not m.all():
                        per_loc[row] = (t_raw[m], [c[m] for c in cols])

    # ---- derived columns --------------------------------------------------
    derived_stats = {}
    if opts.derived:
        derived_stats = compute_derived(intervals, meta)
        if derived_stats.get("overlapping"):
            # Column index, gap and comp_prefix are only defined on a sequence whose
            # intervals do not overlap. Filtering to MPI calls normally produces one even
            # from a deeply recursive trace.
            print("warning: %d of %d consecutive interval pairs overlap (max nesting depth "
                  "%d); the gap_* columns and comp_prefix are meaningless for those rows — "
                  "use --ignore-non-mpi-events or --region-list to keep a non-overlapping "
                  "set, or --no-derived"
                  % (derived_stats["overlapping"], derived_stats["gap_pairs"],
                     derived_stats["max_depth"]), file=sys.stderr)
        if derived_stats.get("iter_index_clipped"):
            print("warning: %d intervals have an iteration index above 65535 and were "
                  "clipped; iteration colours repeat there"
                  % derived_stats["iter_index_clipped"], file=sys.stderr)

    used = np.zeros(len(meta.region_names), dtype=bool)
    for iv in intervals.values():
        if iv.region.size:
            used[np.unique(iv.region)] = True
    region_index = np.full(len(meta.region_names), 0, dtype=np.int64)
    region_names_out: list[str] = []
    for ref in np.flatnonzero(used):
        region_index[ref] = len(region_names_out)
        region_names_out.append(meta.region_names[ref])
    if not region_names_out:
        region_names_out = ["<none>"]

    total_intervals = sum(iv.start.size for iv in intervals.values())

    # Estimate the real per-interval cost so the block sizer targets total block
    # bytes, not just the interval columns: 12 B of interval columns, 8 B per
    # PAPI counter (two u32 samples), plus the amortised message-event payload.
    n_other = sum(t.size for per_loc in other_events.values()
                  for t, _ in per_loc.values())
    est_bytes_per_interval = 12.0 + 8.0 * n_counters
    if opts.derived:
        est_bytes_per_interval += 14.0     # u16 + 8*u8 + u32, see DERIVED_COLUMNS
    if total_intervals:
        est_bytes_per_interval += 10.0 * n_other / total_intervals

    # ---- per level: place rows into blocks -------------------------------
    trace_length_q_by_level = {}
    levels_out = []
    level_rows: dict[int, LevelRows] = {}
    for level in opts.levels:
        quantum = LEVEL_QUANT[level]
        trace_length_q = max(1, meta.trace_length // quantum)
        trace_length_q_by_level[level] = trace_length_q

        if opts.block_length_arg == "auto":
            block_len_q = choose_block_len_q(total_intervals, trace_length_q,
                                             quantum, est_bytes_per_interval)
        else:
            # An explicit length is a level-0 length; coarser levels keep the
            # spec's x256 / x65536 relationship, which in quanta is the same value.
            explicit = parse_time_arg(opts.block_length_arg, meta.timer_resolution)
            mult = max(1, min(MAX_BLOCK_MULTIPLIER, explicit // BASE_BLOCK_CYCLES))
            block_len_q = BASE_BLOCK_CYCLES * mult
        n_blocks = max(1, int(math.ceil(trace_length_q / block_len_q)))

        rows = LevelRows()
        for row, iv in intervals.items():
            add_location_to_level(rows, iv, row, level, meta, opts, block_len_q, n_blocks)
        # Point events (everything that is not an Enter/Leave) get the same block
        # placement, quantised the same way; none are ever dropped.
        rows.other = _place_other_events(other_events, level, meta, block_len_q, n_blocks)
        level_rows[level] = rows
        levels_out.append({
            "level": level, "quant": quantum,
            # Expressed in cycles, so "1x" always means the spec's base block.
            "multiplier": (block_len_q * quantum) / BASE_BLOCK_CYCLES,
            "block_len_cycles": block_len_q * quantum, "block_len_q": block_len_q,
            "n_blocks": n_blocks,
            "drop_threshold": opts.drop_thresholds[level],
        })
    t_transform = time.time()

    # ---- serialise --------------------------------------------------------
    blob, header, stats = _serialise(opts, meta, levels_out, level_rows, other_events,
                                     region_index, region_names_out, n_ranks, n_counters)
    t_assemble = time.time()

    if opts.fmt == "bin":
        with open(opts.output, "wb") as fh:
            fh.write(blob)
    else:
        text = render_text(blob)
        with open(opts.output, "w", encoding="utf-8") as fh:
            fh.write(text)

    stats.update({
        "events_read": events_read,
        "intervals": total_intervals,
        "derived": derived_stats,
        "begin_cut_cycles": begin_cut_cycles,
        "begin_dropped": begin_dropped,
        "anomalies": asdict(anomalies),
        "timing": {
            "definitions_s": round(t_defs - t_start, 3),
            "read_s": round(t_read - t_defs, 3),
            "transform_s": round(t_transform - t_read, 3),
            "assemble_s": round(t_assemble - t_transform, 3),
            "total_s": round(time.time() - t_start, 3),
        },
        "header": header,
        "levels": levels_out,
        "level_rows": {lvl: {"source": r.n_source, "dropped": r.n_dropped,
                             "duplicated": r.n_duplicated}
                       for lvl, r in level_rows.items()},
        "rank_parse_warning": meta.rank_parse_warning,
    })
    return stats


def _region_filter(meta: TraceMeta, opts: Options) -> np.ndarray | None:
    """Boolean mask over region refs, or None when nothing is filtered."""
    if not opts.ignore_non_mpi and not opts.region_list:
        return None
    keep = np.ones(len(meta.region_names), dtype=bool)
    if opts.ignore_non_mpi:
        for i, name in enumerate(meta.region_names):
            if not name.startswith("MPI"):
                keep[i] = False
    if opts.region_list:
        listed = set(opts.region_list)
        for i, name in enumerate(meta.region_names):
            in_list = name in listed
            keep[i] &= (not in_list) if opts.region_list_mode == "ignore" else in_list
    return keep


def _serialise(opts, meta, levels_out, level_rows, other_events,
               region_index, region_names_out, n_ranks, n_counters):
    """Build the whole file: JSON header, directories, long tables, blocks."""
    region_dtype = dtype_for_max(max(0, len(region_names_out) - 1))
    body = bytearray()
    stats_levels = []

    for lvl_meta in levels_out:
        level = lvl_meta["level"]
        rows = level_rows[level]
        n_blocks = lvl_meta["n_blocks"]

        block = rows.concat("block")
        rank = rows.concat("rank")
        start = rows.concat("start")
        end = rows.concat("end")
        region = region_index[rows.concat("region")] if rows.region else np.empty(0, np.int64)
        depth = rows.concat("depth")
        flags = rows.concat("flags")
        derived = {name: rows.concat_derived("derived", name)
                   for name in DERIVED_COLUMNS if rows.derived.get(name)}
        # comp_prefix is accumulated in cycles (the level's quantum is not known
        # when it is computed); express it in this level's quanta, like every
        # other time in the file.
        if "comp_prefix" in derived and lvl_meta["quant"] > 1:
            derived["comp_prefix"] = derived["comp_prefix"] // lvl_meta["quant"]
        papi_c = rows.concat("papi_call") if rows.papi_call else None
        papi_g = rows.concat("papi_gap") if rows.papi_gap else None
        if papi_c is not None and papi_c.ndim == 1 and n_counters:
            papi_c = papi_c.reshape(-1, n_counters)
            papi_g = papi_g.reshape(-1, n_counters)
        long_papi_c = rows.concat("long_papi_call") if rows.long_papi_call else None
        long_papi_g = rows.concat("long_papi_gap") if rows.long_papi_gap else None
        if long_papi_c is not None and long_papi_c.ndim == 1 and n_counters:
            long_papi_c = long_papi_c.reshape(-1, n_counters)
            long_papi_g = long_papi_g.reshape(-1, n_counters)
        # Opt-in only: see --papi-loss. Increments are small, and the quantum is
        # a unit of time, so this trades away whole counters for little space.
        if opts.papi_loss and lvl_meta["quant"] > 1:
            for arr in (papi_c, papi_g, long_papi_c, long_papi_g):
                if arr is not None:
                    arr //= lvl_meta["quant"]

        # Group rows by (block, rank), ordered by start within each cell.
        if block.size:
            order = np.lexsort((start, rank, block))
            block, rank, start, end = block[order], rank[order], start[order], end[order]
            region, depth, flags = region[order], depth[order], flags[order]
            derived = {k: v[order] for k, v in derived.items()}
            if papi_c is not None:
                papi_c, papi_g = papi_c[order], papi_g[order]

        cell = block * n_ranks + rank
        counts = np.bincount(cell, minlength=n_blocks * n_ranks)
        cell_offsets = np.r_[0, np.cumsum(counts)]

        # Column dtypes chosen from the data, one per column per level.
        col_schema = [
            {"name": "starts", "dtype": dtype_for_max(int(start.max()) if start.size else 0)},
            {"name": "ends", "dtype": dtype_for_max(int(end.max()) if end.size else 0)},
            {"name": "regions", "dtype": region_dtype},
            {"name": "depth", "dtype": dtype_for_max(int(depth.max()) if depth.size else 0)},
            {"name": "flags", "dtype": "u8"},
        ]
        # Derived columns follow, marked so a reader can tell where core_bytes ends.
        for name in DERIVED_COLUMNS:
            if name not in derived:
                continue
            if name == "iter_index":
                dt = "u16"
            elif name == "comp_prefix":
                col = derived[name]
                dt = dtype_for_max(int(col.max()) if col.size else 0)
            else:
                dt = "u8"
            col_schema.append({"name": name, "dtype": dt, "derived": True})

        # Two columns per counter — the increment over the call and over the
        # gap that follows it — each sized from its own range.  A counter that
        # never moves (single-precision counters on a double-precision code, say)
        # comes out all-zero and is dropped entirely.
        papi_schema = []
        if n_counters and papi_c is not None and papi_c.size:
            for c in range(n_counters):
                for which, blk_src, long_src in (("call", papi_c, long_papi_c),
                                                 ("gap", papi_g, long_papi_g)):
                    # The resident tier holds the longest calls in the trace, so
                    # its increments are the largest: size the column from both
                    # or MPI_Init's counters wrap.
                    parts = [blk_src[:, c]]
                    if long_src is not None and long_src.size:
                        parts.append(long_src[:, c])
                    hi_val = max(int(part.max()) for part in parts if part.size)
                    lo_val = min(int(part.min()) for part in parts if part.size)
                    papi_schema.append({
                        "name": "papi_%d_%s" % (c, which),
                        "counter": meta.papi_names[c],
                        "scope": which,
                        "dtype": dtype_for_max(max(0, hi_val)),
                        "constant": bool(hi_val == 0 and lo_val == 0),
                        "index": c,
                    })

        # Other-event sections: sort each type by (block, rank, time) and pick
        # a dtype per column from its range, exactly as for the interval columns.
        other_sorted: dict[str, dict] = {}
        other_schema = []
        for name, data in sorted(rows.other.items()):
            blk, rnk, tim = data["block"], data["rank"], data["time"]
            o = np.lexsort((tim, rnk, blk))
            blk, rnk, tim = blk[o], rnk[o], tim[o]
            prm = [p[o] for p in data["params"]]
            cells = np.bincount(blk * n_ranks + rnk, minlength=n_blocks * n_ranks)
            spec = _spec_by_name.get(name)
            pnames = [p.otf2_name for p in spec.params] if spec else []
            columns = [{"name": "time", "dtype": dtype_for_max(int(tim.max()) if tim.size else 0)}]
            for i, pn in enumerate(pnames):
                mx = int(prm[i].max()) if i < len(prm) and prm[i].size else 0
                columns.append({"name": pn, "dtype": dtype_for_max(max(0, mx))})
            other_sorted[name] = {"time": tim, "params": prm,
                                  "offsets": np.r_[0, np.cumsum(cells)]}
            other_schema.append({"name": name, "columns": columns,
                                 "count": int(tim.size)})

        level_body = bytearray()
        block_dir = []
        for b in range(n_blocks):
            lo_cell, hi_cell = b * n_ranks, (b + 1) * n_ranks
            lo, hi = int(cell_offsets[lo_cell]), int(cell_offsets[hi_cell])
            n_rows = hi - lo
            rank_off = (cell_offsets[lo_cell:hi_cell + 1] - lo).astype(np.uint32)

            active_papi = [c for c in papi_schema if not c["constant"]]
            sections = []
            for sec in other_schema:
                so = other_sorted[sec["name"]]["offsets"]
                sections.append({"name": sec["name"], "columns": sec["columns"],
                                 "n_rows": int(so[hi_cell]) - int(so[lo_cell])})
            layout = BlockLayout(n_ranks, n_rows, col_schema, active_papi, sections)
            base = len(level_body)
            level_body.extend(b"\0" * layout.total)

            if layout.mask_words:
                write_column(level_body, base, layout.mask_offset,
                             np.array(layout.mask_value(), dtype=np.int64), "u32")
            if n_rows:
                write_column(level_body, base, 0, rank_off, "u32")
            for sec in sections:
                name = sec["name"]
                so = other_sorted[name]["offsets"]
                s_lo, s_hi = int(so[lo_cell]), int(so[hi_cell])
                if s_hi > s_lo:
                    write_column(level_body, base, layout.offsets[name + ".rank_offsets"],
                                 (so[lo_cell:hi_cell + 1] - s_lo).astype(np.uint32), "u32")
                    data = other_sorted[name]
                    write_column(level_body, base, layout.offsets[name + ".time"],
                                 data["time"][s_lo:s_hi], sec["columns"][0]["dtype"])
                    for ci, col in enumerate(sec["columns"][1:]):
                        if ci < len(data["params"]):
                            write_column(level_body, base,
                                         layout.offsets["%s.%s" % (name, col["name"])],
                                         data["params"][ci][s_lo:s_hi], col["dtype"])
            if n_rows:
                write_column(level_body, base, layout.offsets["starts"], start[lo:hi], col_schema[0]["dtype"])
                write_column(level_body, base, layout.offsets["ends"], end[lo:hi], col_schema[1]["dtype"])
                write_column(level_body, base, layout.offsets["regions"], region[lo:hi], col_schema[2]["dtype"])
                write_column(level_body, base, layout.offsets["depth"], depth[lo:hi], col_schema[3]["dtype"])
                write_column(level_body, base, layout.offsets["flags"], flags[lo:hi], "u8")
                for spec in [c for c in col_schema if c.get("derived")]:
                    write_column(level_body, base, layout.offsets[spec["name"]],
                                 derived[spec["name"]][lo:hi], spec["dtype"])
                for col in papi_schema:
                    if col["constant"]:
                        continue
                    src_col = papi_c if col["scope"] == "call" else papi_g
                    write_column(level_body, base, layout.offsets[col["name"]],
                                 src_col[lo:hi, col["index"]], col["dtype"])

            block_dir.append((base, layout.total, layout.interval_bytes, n_rows,
                              layout.core_bytes))

        # Visible extent of this level, in absolute quanta: block-relative
        # times plus the block's own base, and the resident table is already
        # absolute. Computed from the rows actually written, so a level that
        # drops intervals reports the span of what it kept.
        long_start_abs = rows.concat("long_start")
        long_end_abs = rows.concat("long_end")
        abs_start = block * lvl_meta["block_len_q"] + start if start.size else start
        abs_end = block * lvl_meta["block_len_q"] + end if end.size else end
        t0_candidates = [a.min() for a in (abs_start, long_start_abs) if a.size]
        t1_candidates = [a.max() for a in (abs_end, long_end_abs) if a.size]
        t0_q = int(min(t0_candidates)) if t0_candidates else 0
        t1_q = int(max(t1_candidates)) if t1_candidates else 0

        # Long-interval (resident) table.
        long_rank = rows.concat("long_rank")
        long_start = rows.concat("long_start")
        long_end = rows.concat("long_end")
        long_region = region_index[rows.concat("long_region")] if rows.long_region else np.empty(0, np.int64)
        long_depth = rows.concat("long_depth")
        long_flags = rows.concat("long_flags")
        long_derived = {name: rows.concat_derived("long_derived", name)
                        for name in DERIVED_COLUMNS if rows.long_derived.get(name)}
        if "comp_prefix" in long_derived and lvl_meta["quant"] > 1:
            long_derived["comp_prefix"] = long_derived["comp_prefix"] // lvl_meta["quant"]
        if long_rank.size:
            o = np.lexsort((long_start, long_rank))
            long_rank, long_start, long_end = long_rank[o], long_start[o], long_end[o]
            long_region, long_depth, long_flags = long_region[o], long_depth[o], long_flags[o]
            long_derived = {k: v[o] for k, v in long_derived.items()}
            if long_papi_c is not None:
                long_papi_c, long_papi_g = long_papi_c[o], long_papi_g[o]

        # Lay the level out: directory, long table, then the blocks.
        level_start = align_up(len(body), 8)
        body.extend(b"\0" * (level_start - len(body)))

        dir_bytes = bytearray()
        for off, length, ibytes, nrows, cbytes in block_dir:
            dir_bytes.extend(np.uint64(off).tobytes())
            dir_bytes.extend(np.uint32(length).tobytes())
            dir_bytes.extend(np.uint32(ibytes).tobytes())
            dir_bytes.extend(np.uint32(nrows).tobytes())
            dir_bytes.extend(np.uint32(cbytes).tobytes())
        dir_offset = len(body)
        body.extend(dir_bytes)

        long_offset = align_up(len(body), 8)
        body.extend(b"\0" * (long_offset - len(body)))
        # The resident table mirrors the block columns, so a viewer resolving a
        # rank out of it gets the same derived values it would from a block.
        # Named exactly as the block columns, so a viewer that resolves a rank
        # out of the resident table reads it through the same code path — the
        # whole point of the format is that a row is a row wherever it lives.
        long_cols = [("rank", long_rank, "u32"), ("starts", long_start, "u64"),
                     ("ends", long_end, "u64"), ("regions", long_region, region_dtype),
                     ("depth", long_depth, "u8"), ("flags", long_flags, "u8")]
        for spec in [c for c in col_schema if c.get("derived")]:
            long_cols.append((spec["name"], long_derived.get(
                spec["name"], np.zeros(long_rank.size, np.int64)), spec["dtype"]))
        # Counters for the resident tier too — these are the longest calls in the
        # trace (one MPI_Init per rank), so leaving them without counters would
        # blank out exactly the rows a viewer dwells on longest.
        for spec in papi_schema:
            if spec["constant"]:
                continue
            src_col = long_papi_c if spec["scope"] == "call" else long_papi_g
            arr = (src_col[:, spec["index"]] if src_col is not None
                   else np.zeros(long_rank.size, np.int64))
            long_cols.append((spec["name"], arr, spec["dtype"]))
        for _name, arr, dt in long_cols:
            pos = align_up(len(body), _DTYPE_BYTES[dt])
            body.extend(b"\0" * (pos - len(body)))
            body.extend(np.ascontiguousarray(arr, dtype=_NUMPY_DTYPES[dt]).tobytes())

        blocks_offset = align_up(len(body), 8)
        body.extend(b"\0" * (blocks_offset - len(body)))
        body.extend(level_body)

        lvl_meta.update({
            "directory": {"byte_offset": dir_offset, "row_bytes": 24, "n_blocks": n_blocks},
            "long_intervals": {"byte_offset": long_offset, "count": int(long_rank.size),
                               "columns": ["%s:%s" % (n, dt) for n, _a, dt in long_cols]},
            "blocks_byte_offset": blocks_offset,
            "block_columns": col_schema,
            "papi_columns": papi_schema,
            "other_events": other_schema,
            "n_rows": int(start.size),
            # The visible time span of this level, in absolute quanta. This is
            # the viewer's timeline, and it is not clock.trace_length: that is
            # the ProgramBegin->ProgramEnd span, which starts before the first
            # interval and ends after the last.
            "t0_q": t0_q, "t1_q": t1_q,
            # How many source intervals this level dropped, so the viewer can
            # say so when it picks the level automatically.
            "n_source": int(rows.n_source), "dropped": int(rows.n_dropped),
            "duplicated": int(rows.n_duplicated),
        })
        stats_levels.append({
            "level": level,
            "rows": int(start.size),
            "long_rows": int(long_rank.size),
            "directory_bytes": len(dir_bytes),
            "blocks_bytes": len(level_body),
            "block_sizes": [d[1] for d in block_dir],
            "interval_bytes": [d[2] for d in block_dir],
        })

    header = {
        "format": "PV3D",
        "version": FORMAT_VERSION,
        "source": {"trace_dir": opts.trace_dir,
                   "size_bytes": _dir_size(opts.trace_dir)},
        "clock": {"timer_resolution": meta.timer_resolution,
                  "global_offset": meta.global_offset,
                  "trace_length": meta.trace_length},
        "options": opts.as_json(),
        "n_ranks": n_ranks,
        "locations": meta.location_info,
        "regions": region_names_out,
        "region_dtype": region_dtype,
        "papi": {"counters": meta.papi_names if n_counters else []},
        "strings": _string_dict.items,
        "levels": levels_out,
    }

    json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    json_padded = json_bytes + b" " * (align_up(len(json_bytes), 4) - len(json_bytes))
    prefix = bytearray()
    prefix.extend(MAGIC_BIN)
    prefix.extend(np.uint32(FORMAT_VERSION).tobytes())
    prefix.extend(np.uint32(len(json_padded)).tobytes())
    bin_offset = len(MAGIC_BIN) + 4 + 4 + 4 + len(json_padded)
    bin_offset = align_up(bin_offset, 8)
    prefix.extend(np.uint32(bin_offset).tobytes())
    prefix.extend(json_padded)
    prefix.extend(b"\0" * (bin_offset - len(prefix)))

    blob = bytes(prefix) + bytes(body)
    stats = {"bytes": len(blob), "json_bytes": len(json_bytes),
             "levels_detail": stats_levels}
    return blob, header, stats


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# §10  Reference decoder and text mirror
# ─────────────────────────────────────────────────────────────────────────────

class Pv3dReader:
    """Minimal reference reader; also the engine behind `--format txt`."""

    def __init__(self, blob: bytes):
        if blob[:8] != MAGIC_BIN:
            raise ValueError("not a PV3D binary file")
        self.blob = blob
        json_len = int(np.frombuffer(blob, np.uint32, 1, 12)[0])
        self.bin_offset = int(np.frombuffer(blob, np.uint32, 1, 16)[0])
        self.header = json.loads(blob[20:20 + json_len].decode("utf-8").rstrip())
        self.n_ranks = self.header["n_ranks"]
        self._block_cache: dict[tuple[int, int], dict] = {}
        self._dir_cache: dict[int, dict] = {}

    def level(self, level: int) -> dict:
        for lv in self.header["levels"]:
            if lv["level"] == level:
                return lv
        raise KeyError(level)

    def block_directory(self, level: int) -> dict:
        if level in self._dir_cache:
            return self._dir_cache[level]
        lv = self.level(level)
        d = lv["directory"]
        base = self.bin_offset + d["byte_offset"]
        raw = np.frombuffer(self.blob, np.uint8,
                            d["n_blocks"] * d["row_bytes"], base)
        raw = raw.reshape(d["n_blocks"], d["row_bytes"])
        out = {
            "offset": raw[:, 0:8].copy().view(np.uint64).ravel(),
            "length": raw[:, 8:12].copy().view(np.uint32).ravel(),
            "interval_bytes": raw[:, 12:16].copy().view(np.uint32).ravel(),
            "n_rows": raw[:, 16:20].copy().view(np.uint32).ravel(),
            # Padding in version 1; bytes through `flags` from version 2 on.
            "core_bytes": raw[:, 20:24].copy().view(np.uint32).ravel(),
        }
        self._dir_cache[level] = out
        return out

    def block(self, level: int, index: int) -> dict:
        """Decode one block; only its own bytes are touched.

        Nothing but the interval row count comes from the directory.  Each
        other-event section's row count is its own `rank_offsets[n_ranks]`, and
        because the sections are laid out in schema order every section's
        position is computable once the previous one has been read — so the
        block stays self-describing without storing extra counts.
        """
        key = (level, index)
        if key in self._block_cache:
            return self._block_cache[key]
        lv = self.level(level)
        d = self.block_directory(level)
        base = self.bin_offset + lv["blocks_byte_offset"] + int(d["offset"][index])
        n_rows = int(d["n_rows"][index])
        active_papi = [c for c in lv["papi_columns"] if not c["constant"]]
        layout = BlockLayout(self.n_ranks, n_rows, lv["block_columns"], active_papi, [])
        out = {"n_rows": n_rows, "other": {},
               "rank_offsets": (np.frombuffer(self.blob, np.uint32, self.n_ranks + 1, base)
                                if n_rows else np.zeros(self.n_ranks + 1, np.uint32))}
        for col in lv["block_columns"]:
            out[col["name"]] = (np.frombuffer(
                self.blob, _NUMPY_DTYPES[col["dtype"]], n_rows,
                base + layout.offsets[col["name"]]) if n_rows
                else np.empty(0, _NUMPY_DTYPES[col["dtype"]]))
        for col in active_papi:
            out[col["name"]] = (np.frombuffer(
                self.blob, _NUMPY_DTYPES[col["dtype"]], n_rows,
                base + layout.offsets[col["name"]])
                if n_rows else np.empty(0, _NUMPY_DTYPES[col["dtype"]]))

        # Sections: only the ones the block's bitmask marks present are stored,
        # so an empty type costs nothing rather than a full rank_offsets of zeros.
        schema = lv.get("other_events", [])
        off = layout.total_before_sections
        present = [True] * len(schema)
        if schema and self.header.get("version", 1) >= 2:
            mask_words = (len(schema) + 31) // 32
            off = align_up(off, 4)
            words = np.frombuffer(self.blob, np.uint32, mask_words, base + off)
            off += 4 * mask_words
            present = [bool(int(words[i // 32]) >> (i % 32) & 1) for i in range(len(schema))]
        for sec, is_present in zip(schema, present):
            if not is_present:
                out["other"][sec["name"]] = {
                    "rank_offsets": np.zeros(self.n_ranks + 1, np.uint32), "n_rows": 0,
                    **{c["name"]: np.empty(0, _NUMPY_DTYPES[c["dtype"]]) for c in sec["columns"]}}
                continue
            off = align_up(off, 4)
            ranks = np.frombuffer(self.blob, np.uint32, self.n_ranks + 1, base + off)
            off += 4 * (self.n_ranks + 1)
            sec_rows = int(ranks[self.n_ranks])
            cols = {"rank_offsets": ranks, "n_rows": sec_rows}
            for col in sec["columns"]:
                size = _DTYPE_BYTES[col["dtype"]]
                off = align_up(off, size)
                cols[col["name"]] = np.frombuffer(
                    self.blob, _NUMPY_DTYPES[col["dtype"]], sec_rows, base + off)
                off += size * sec_rows
            out["other"][sec["name"]] = cols
        self._block_cache[key] = out
        return out

    def long_intervals(self, level: int) -> dict:
        lv = self.level(level)
        info = lv["long_intervals"]
        n = info["count"]
        pos = self.bin_offset + info["byte_offset"]
        out = {}
        for spec in info["columns"]:
            name, dt = spec.split(":")
            size = _DTYPE_BYTES[dt]
            pos = align_up(pos, size)
            out[name] = np.frombuffer(self.blob, _NUMPY_DTYPES[dt], n, pos)
            pos += size * n
        return out

    def intervals_of(self, level: int, rank: int) -> dict:
        """All intervals for one rank across the level, in absolute quantised time."""
        lv = self.level(level)
        d = self.block_directory(level)
        starts, ends, regions, depths, flags = [], [], [], [], []
        for b in range(lv["n_blocks"]):
            if int(d["n_rows"][b]) == 0:
                continue
            blk = self.block(level, b)
            lo = int(blk["rank_offsets"][rank])
            hi = int(blk["rank_offsets"][rank + 1])
            if hi <= lo:
                continue
            base = b * lv["block_len_q"]
            starts.append(blk["starts"][lo:hi].astype(np.int64) + base)
            ends.append(blk["ends"][lo:hi].astype(np.int64) + base)
            regions.append(blk["regions"][lo:hi].astype(np.int64))
            depths.append(blk["depth"][lo:hi].astype(np.int64))
            flags.append(blk["flags"][lo:hi].astype(np.int64))
        empty = np.empty(0, np.int64)
        return {
            "start": np.concatenate(starts) if starts else empty,
            "end": np.concatenate(ends) if ends else empty,
            "region": np.concatenate(regions) if regions else empty,
            "depth": np.concatenate(depths) if depths else empty,
            "flags": np.concatenate(flags) if flags else empty,
        }


def render_text(blob: bytes) -> str:
    """Exact text mirror: the JSON header, then every stored value as decimals.

    Values are space-separated; a newline is emitted when switching column,
    block or loss level.  The offsets printed are the *binary* byte offsets, so
    the text and binary forms carry identical information.
    """
    reader = Pv3dReader(blob)
    lines = [MAGIC_TXT.decode("ascii"),
             json.dumps(reader.header, separators=(",", ":"))]
    for lv in reader.header["levels"]:
        level = lv["level"]
        lines.append("# level %d" % level)
        d = reader.block_directory(level)
        lines.append("# directory")
        for key in ("offset", "length", "interval_bytes", "n_rows", "core_bytes"):
            lines.append(" ".join(str(int(v)) for v in d[key]))
        longs = reader.long_intervals(level)
        lines.append("# long_intervals")
        for name, col in longs.items():
            lines.append(" ".join(str(int(v)) for v in col))
        for b in range(lv["n_blocks"]):
            blk = reader.block(level, b)
            lines.append("# block %d" % b)
            lines.append(" ".join(str(int(v)) for v in blk["rank_offsets"]))
            for col in lv["block_columns"]:
                lines.append(" ".join(str(int(v)) for v in blk[col["name"]]))
            for col in lv["papi_columns"]:
                if not col["constant"]:
                    lines.append(" ".join(str(int(v)) for v in blk[col["name"]]))
            for sec in lv.get("other_events", []):
                data = blk["other"][sec["name"]]
                lines.append("# event %s" % sec["name"])
                lines.append(" ".join(str(int(v)) for v in data["rank_offsets"]))
                for col in sec["columns"]:
                    lines.append(" ".join(str(int(v)) for v in data[col["name"]]))
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# §11  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def report(stats: dict, opts: Options) -> None:
    h = stats["header"]
    clock = h["clock"]
    res = clock["timer_resolution"]
    src = h["source"]["size_bytes"]

    print()
    print("PV3D conversion report")
    print("=" * 72)
    print("source        %s" % h["source"]["trace_dir"])
    print("              %s in %d locations, %s events read"
          % (_human(src), h["n_ranks"], f"{stats['events_read']:,}"))
    print("intervals     %s recovered" % f"{stats['intervals']:,}")
    if stats.get("begin_cut_cycles"):
        cut = stats["begin_cut_cycles"]
        print("begin time    cut at %d cycles (%.3f ms into the run); %s intervals dropped"
              % (cut, (cut - clock["global_offset"]) / (0.001 * res),
                 f"{stats['begin_dropped']:,}"))
    d = stats.get("derived") or {}
    if opts.derived and d:
        print("derived       %d columns per interval (+14 B/row); max iteration index %s, "
              "max nesting depth %d, %s overlapping pairs"
              % (len(DERIVED_COLUMNS), f"{d.get('max_iter_index', 0):,}",
                 d.get("max_depth", 0), f"{d.get('overlapping', 0):,}"))
    elif not opts.derived:
        print("derived       not stored (--no-derived); the viewer must compute percentiles "
              "itself and cannot render block-at-a-time")
    print("output        %s  (%s, %.1f%% of source)"
          % (opts.output, _human(stats["bytes"]),
             100.0 * stats["bytes"] / src if src else 0.0))

    print()
    print("Time blocks")
    print("  %-5s %16s %12s %12s %10s %10s %8s %12s"
          % ("level", "length (cycles)", "run-ms", "video-ms", "frames",
             "video-s", "blocks", "% run/block"))
    for lv in stats["levels"]:
        bc = lv["block_len_cycles"]
        run_ms = bc / (0.001 * res)
        video_ms = bc / (60 * 200000)
        frames = bc / 200000
        print("  %-5d %16d %12.4f %12.4f %10.2f %10.4f %8d %12.6f"
              % (lv["level"], bc, run_ms, video_ms, frames, frames / 60,
                 lv["n_blocks"], 100.0 * bc / clock["trace_length"]))
    print("  (video-ms is the requested block_cycles/(60*200000); frames and")
    print("   video-s are shown alongside because that ratio reads as seconds)")

    print()
    print("Per level")
    for det, lv in zip(stats["levels_detail"], stats["levels"]):
        level = det["level"]
        rows = stats["level_rows"][level]
        span_ms = (lv["t1_q"] - lv["t0_q"]) * lv["quant"] / (0.001 * res)
        sizes = np.array(det["block_sizes"]) if det["block_sizes"] else np.zeros(1)
        ivb = np.array(det["interval_bytes"]) if det["interval_bytes"] else np.zeros(1)
        total = det["directory_bytes"] + det["blocks_bytes"]
        print("  level %d  quant=%d  block=%gx base (%d cycles)"
              % (level, lv["quant"], lv["multiplier"], lv["block_len_cycles"]))
        kept = rows["source"] - rows["dropped"]
        print("    rows            %s stored (%s source, %s dropped, %s duplicated)"
              % (f"{det['rows']:,}", f"{rows['source']:,}",
                 f"{rows['dropped']:,}", f"{rows['duplicated']:,}"))
        print("    visible span    %.3f ms (%s of the source intervals survive: %.1f%%)"
              % (span_ms, f"{kept:,}",
                 100.0 * kept / rows["source"] if rows["source"] else 100.0))
        print("    resident table  %s long intervals" % f"{det['long_rows']:,}")
        print("    directory       %s (%.1f%% of the level)"
              % (_human(det["directory_bytes"]),
                 100.0 * det["directory_bytes"] / total if total else 0))
        print("    blocks          %s total; per block mean %s, p95 %s, max %s"
              % (_human(det["blocks_bytes"]), _human(sizes.mean()),
                 _human(np.percentile(sizes, 95)), _human(sizes.max())))
        print("    interval-only   mean %s per block (what a viewer fetches to draw)"
              % _human(ivb.mean()))
        if det["rows"]:
            print("    bytes/interval  %.2f" % (total / det["rows"]))
        cols = ", ".join("%s:%s" % (c["name"], c["dtype"]) for c in lv["block_columns"])
        print("    columns         %s" % cols)
        if lv["papi_columns"]:
            # Two columns per counter: the increment during a call and during the gap after
            # it. Naming the scope matters — a counter can be flat inside MPI calls and still
            # move during computation.
            n_counters_lv = len({c["counter"] for c in lv["papi_columns"]})
            dropped = ["%s/%s" % (c["counter"], c["scope"])
                       for c in lv["papi_columns"] if c["constant"]]
            sizes = ", ".join("%s/%s:%s" % (c["counter"].replace("PAPI_", ""), c["scope"],
                                            c["dtype"])
                              for c in lv["papi_columns"] if not c["constant"])
            print("    papi            %d counters in %d columns; %d never advance and are "
                  "dropped%s" % (n_counters_lv, len(lv["papi_columns"]), len(dropped),
                                 (": " + ", ".join(dropped)) if dropped else ""))
            if sizes:
                print("    papi columns    %s" % sizes)

    print()
    t = stats["timing"]
    print("Timing        definitions %.2fs  read %.2fs  transform %.2fs  assemble %.2fs  total %.2fs"
          % (t["definitions_s"], t["read_s"], t["transform_s"], t["assemble_s"], t["total_s"]))
    if t["read_s"] > 0:
        print("              effective %s events/s during the read"
              % f"{int(stats['events_read'] / t['read_s']):,}")
    an = dict(stats["anomalies"])
    mismatch = an.pop("region_mismatch", 0)
    flagged = {k: v for k, v in an.items() if v}
    print("Anomalies     %s" % ("none" if not flagged else flagged))
    if mismatch:
        print("              %s intervals had a Leave whose region ref differed from"
              % f"{mismatch:,}")
        print("              its Enter's (normal with compiler instrumentation; the")
        print("              region entered is the one stored)")
    if stats["rank_parse_warning"]:
        print("WARNING       some location groups had no trailing rank number; "
              "the dense row index was used instead")
    if 1 in stats["level_rows"] and stats["level_rows"][1]["dropped"] == 0:
        print("NOTE          level 1 dropped no intervals: on Score-P traces the 256-cycle")
        print("              threshold is below the instrumentation cost. Use")
        print("              --ll1-drop-threshold to make level 1 actually thin the data.")


def _human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024
    return "%.1f GiB" % n


# ─────────────────────────────────────────────────────────────────────────────
# §12  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    try:
        opts = resolve_options(argv)
        stats = convert(opts)
    except ValueError as exc:          # bad CLI values: report, don't traceback
        print("error: %s" % exc, file=sys.stderr)
        return 2
    if not opts.quiet:
        report(stats, opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

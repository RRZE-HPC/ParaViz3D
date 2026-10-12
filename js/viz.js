import * as THREE from 'three';

import { gsap } from 'gsap';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { Reflector } from 'three/addons/objects/Reflector.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

import { HTMLMesh } from 'three/addons/interactive/HTMLMesh.js';
import { InteractiveGroup } from 'three/addons/interactive/InteractiveGroup.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import Stats from 'three/addons/libs/stats.module.js';

import { initVampirTimeline, updateVampirTimeline, loadTimelineData, handleTimelinePointer,
         setTimelineView, setTimelineSelection, HUD_CANVAS_WIDTH } from './vampir-timeline.js';
import { isRecording, startRecording, stopRecording, downloadBlob } from './recorder.js';

import { config } from './interface.js';
import { COL, TextSource, Pv3dSource, makeSample, heatmapColumns, isGapHeatmap, rankPercentiles } from './tracesource.js';
import { httpTransport, fileTransport, sniff, Pv3dFile, RangeUnsupportedError } from './pv3d.js';

// Three.js variables
let scene, camera, renderer, controls, masterTimeline, instancedMesh, bordersObject;
let sphereMesh = null, sphereEdgesObject = null;
let pointLightUpper, pointLightLower, helperUpper, helperLower, gui;
const sphereLights = []; // extra lights added only when Sphere geometry is active
let dolly, controller1, controller2, controllerGrip1, controllerGrip2;
let vrControlsGroup, vrControlsMesh;
let vampirHudMesh, vampirTexture, vampirReady = false;
let vizGroup, vrHudParent;          // visualization container + VR HUD anchor
let xrLastTime = null;              // for manual timeline advancement in XR
let pinching = false;               // two-controller pinch-to-scale state
let pinchStartDist = 1;
let pinchStartScale = 1;
// The loaded trace, as a TraceSource (js/tracesource.js) — a columnar model plus a cursor,
// shared by all colour schemes. `sample` is the per-frame cursor result, allocated once per
// visualization and refilled in place so a frame allocates nothing.
let traceSource = null;
let sample = null;
// Scratch for the one metric that ranks ranks live every frame ("Windowed average comp work"),
// sized with the trace so its per-frame pass allocates nothing either.
let speedVals = null, speedPct = null;
//: Set while a block fetch is outstanding; playback is held rather than drawn from data that
//: has not arrived. `resumeAfterLoad` remembers whether the user had it playing.
let waitingForData = false, resumeAfterLoad = false;
const timeProxy = { time: 0 };

// ---------------------------------------------------------------------------
// The three time spans
// ---------------------------------------------------------------------------
// One span used to serve three jobs, which worked only because they were always equal. The
// timeline HUD can now be zoomed, and playback can be restricted to a selected region, so
// they are not:
//
//   trace  [0, currentMaxTicks]   all the data
//   view   [viewStart, viewEnd]   what the HUD draws
//   play   [playStart, playEnd]   what GSAP animates
//
// All three in ticks (1 µs for legacy .txt, one timer cycle for .pv3dbin). `progress` is only
// ever meaningful against the *play* span; the HUD's x axis is only ever the *view* span.
// Mixing the two is the easiest mistake to make here, so every conversion goes through one of
// the functions below rather than being open-coded at the call site. (The time->pixel half of
// the view mapping lives in vampir-timeline.js, which is the only thing that draws.)
let viewStart = 0, viewEnd = 0;
let playStart = 0, playEnd = 0;
//: The marked region in ticks, or null when playback spans the whole trace. `play` is derived
//: from it, so this is the one piece of state that says "a selection exists".
let timelineSelection = null;
//: Absolute time of timeline zero, for the wall clock. Kept at module scope so a repaint
//: forced outside GSAP's onUpdate can show the same thing GSAP's would.
let clockInitialOffset = 0;

/**
 * Repaints everything that follows the playhead.
 *
 * GSAP only fires onUpdate when its time actually changes, so rebuilding the tween and landing
 * on progress 0 — which is what committing a selection ahead of the playhead does — renders
 * nothing and leaves the clock and the HUD showing the previous range's position.
 */
function refreshTimelineUI() {
  updateInstanceColors();
  updateClockDisplay(clockInitialOffset);
  if (vampirReady) {
    updateVampirTimeline(timeProxy.time);
    if (vampirTexture) vampirTexture.needsUpdate = true;
  }
}

/** HUD x, as a 0-1 fraction of its width, -> trace time in ticks. */
function uvToTime(u) {
  return viewStart + u * (viewEnd - viewStart);
}

/** Trace time -> GSAP progress over the animated range. */
function timeToProgress(t) {
  const span = playEnd - playStart;
  return span > 0 ? (t - playStart) / span : 0;
}

/** Resets both spans to the whole trace — the state every load starts from. */
function resetSpans(traceTicks) {
  viewStart = 0; viewEnd = traceTicks;
  playStart = 0; playEnd = traceTicks;
}

// ---------------------------------------------------------------------------
// Temporal resolution: the zoom level and the frame step are the same quantity
// ---------------------------------------------------------------------------
// Zooming the timeline in by 10x means you now care about detail 10x finer, so the frame step
// follows it down; typing a finer frame step means you want to see that detail, so the view
// follows it in. The relationship is multiplicative about whatever the Frame Step field held
// when the trace was loaded, which keeps the user's own value as the fully-zoomed-out setting
// instead of overwriting it with a derived one.
let baseFrameStepUs = 100;      // the field's value at load; the anchor for both directions
let tweenStepTicks = 100;       // the step the current tween's duration was computed from
let syncingResolution = false;  // each direction writes the other; this breaks the loop

/** The Frame Step field's current value in ticks. Read live everywhere, because zoom writes
 *  it — the `config` snapshot is only the value the trace was loaded with. */
function liveFrameStepTicks() {
  const us = parseFloat(document.getElementById('timeGranularity')?.value) || 100;
  return microsecondsToTicks(us);
}

/** Zoom factor currently applied to the timeline, 1 when fully zoomed out. */
function zoomFactor() {
  const span = viewEnd - viewStart;
  return span > 0 && currentMaxTicks > 0 ? currentMaxTicks / span : 1;
}

/**
 * Pushes the current zoom into the Frame Step field and rescales playback to match.
 *
 * Playback speed is handled with `timeScale()` rather than by rebuilding the tween: duration
 * is proportional to 1/frameStep, so scaling the rate by newStep/baseStep is equivalent — and
 * unlike a rebuild it leaves `time()` and `progress()` untouched, so the playhead does not
 * jump while the wheel is turning.
 */
function syncFrameStepToZoom() {
  if (syncingResolution) return;
  syncingResolution = true;
  try {
    const us = baseFrameStepUs / zoomFactor();
    const field = document.getElementById('timeGranularity');
    if (field) {
      // Keep it readable: three significant figures is plenty for a step in microseconds.
      field.value = String(us >= 1 ? Math.max(1, Math.round(us)) : Number(us.toPrecision(3)));
      field.dispatchEvent(new Event('input', { bubbles: true }));
    }
    // Against the step the tween's duration was built from, not the load-time one: a rebuild
    // (a selection) recomputes the duration at the then-current step and resets timeScale to 1,
    // so anchoring on `baseFrameStepTicks` would reapply the whole zoom on top of it.
    if (masterTimeline && tweenStepTicks > 0) {
      masterTimeline.timeScale(microsecondsToTicks(us) / tweenStepTicks);
    }
  } finally {
    syncingResolution = false;
  }
}

/** The inverse: a hand-typed frame step zooms the view about its current centre. */
function syncZoomToFrameStep() {
  if (syncingResolution || !vampirReady || currentMaxTicks <= 0) return;
  const us = parseFloat(document.getElementById('timeGranularity')?.value);
  if (!(us > 0)) return;
  syncingResolution = true;
  try {
    const wanted = Math.max(minViewSpan(),
                            Math.min(currentMaxTicks, currentMaxTicks * (us / baseFrameStepUs)));
    const centre = (viewStart + viewEnd) / 2;
    let start = centre - wanted / 2;
    let end = start + wanted;
    if (start < 0) { start = 0; end = wanted; }
    if (end > currentMaxTicks) { end = currentMaxTicks; start = Math.max(0, end - wanted); }
    viewStart = start; viewEnd = end;
    applyTimelineView();
    // Against the step the tween's duration was built from, not the load-time one: a rebuild
    // (a selection) recomputes the duration at the then-current step and resets timeScale to 1,
    // so anchoring on `baseFrameStepTicks` would reapply the whole zoom on top of it.
    if (masterTimeline && tweenStepTicks > 0) {
      masterTimeline.timeScale(microsecondsToTicks(us) / tweenStepTicks);
    }
  } finally {
    syncingResolution = false;
  }
}
let compositeCanvas = null, compositeCtx = null; // off-screen canvas used to burn in overlays while recording
let hoverTooltip = { visible: false, text: '', x: 0, y: 0 };
// Trace span in ticks (see TraceSource: 1 µs for legacy .txt, one cycle for .pv3dbin).
// Set by setupAnimation(); read by getFrameStep().
let currentMaxTicks = 0;
const tempColor = new THREE.Color();

// A ColorScheme bundles all state for one independent coloring strategy (which metric type,
// its colors/cutoffs/modes, and the precomputed per-event color tables). Dual View runs two
// of these (scheme2/scheme3, one per split-viewport half) instead of just scheme1.
function createColorScheme() {
  return {
    compColor: new THREE.Color(0x000000), compColorAlpha: 0.0,
    mpiStartColor: new THREE.Color(0xff0000), mpiStartAlpha: 1.0,
    mpiEndColor: new THREE.Color(0x00ff00), mpiEndAlpha: 1.0,

    // Coloring mode: 'gradient' (default MPI start/end lerp), 'iterations' (round-robin per
    // Wait/Waitall), 'heatmap' (MPI Heatmap speeders/laggers family), 'speedHeatmap', or 'distinguish'
    colorMode: 'gradient',
    iterationColors: [],       // Array of THREE.Color for iteration mode
    iterationAlphas: [],       // Matching alpha values for iteration colors

    // Heatmap speeders/laggers state (shared by "MPI Heatmap speeders and laggers", "Comp
    // Heatmap speeders and laggers", and "MPI Heatmap instantaneous speed (loop)").
    // 'mpi': indexed by event index — ranks MPI calls by Enter/Leave time (heatmapAppliesTo).
    // 'comp': indexed by gap index — ranks the out-of-MPI (Comp) periods instead.
    // 'mpiDuration': indexed by event index like 'mpi', but ranks calls by duration — a single
    // color per call (Enter/Leave colors are set equal), reusing the exact same render path as 'mpi'.
    heatmapAppliesTo: 'mpi',   // 'mpi' | 'comp' | 'mpiDuration'
    // Names of the two percentile columns this heatmap reads, chosen from heatmapAppliesTo by
    // heatmapColumns(). The per-event THREE.Color tables these replaced were built eagerly over
    // the whole trace before the first frame, once per scheme; the columns are shared by all
    // three schemes and turned into colours by percentileToBucketColor() at render time.
    enterCol: COL.PCT_ENTER, leaveCol: COL.PCT_LEAVE,
    gapIndexed: false,         // true when enterCol/leaveCol are indexed by gap, not by interval
    heatmapCutoffs: [],        // percentiles, highest -> lowest, as given by the user
    heatmapBucketColors: [],   // [{color: THREE.Color, alpha}, ...], length = cutoffs.length + 1
    heatmapBetweenMode: 'custom', // 'custom' | 'keep' | 'gradient'
    heatmapFlatColor: new THREE.Color(0x000000), // flat color for whichever role plays "the gap"
    heatmapFlatAlpha: 0.0,

    // "Heatmap instantaneous speed" state (colorMode 'speedHeatmap') — live, per-frame
    // percentile of recent comp-time pace, distinct from the per-call heatmaps above. Its
    // cumulative-comp-time prefix lives on the TraceSource (COL.COMP_PREFIX), since it is a
    // property of the trace and not of any colour scheme.
    heatmapWindowUs: 1000,    // trailing window, as typed in the µs UI field
    heatmapWindowTicks: 1000, // the same window in ticks, resolved by prepareScheme()

    // "MPI Heatmap instantaneous speed (run)" state (colorMode 'speedHeatmapRun') — like the
    // 'mpiDuration' heatmap, but restricted to one named MPI region and ranked against every
    // occurrence of that region across all ranks and all loops (a single flat percentile pool),
    // rather than per-column against just the same loop position. Reuses heatmapBucketColors/
    // heatmapCutoffs/heatmapBetweenMode/heatmapFlatColor/heatmapFlatAlpha above for its colors
    // and in-between-calls behavior — only the target filtering/lookup state is new here.
    runTargetFn: '',        // exact MPI function name to rank, e.g. "MPI_Waitall"
    runTargetRegionId: -1,  // its region id in the source, or -1 when the trace never calls it
    runIndex: null,         // Array<Int32Array> of that region's rows per rank (from the source)
    runSample: null,        // its own per-frame cursor result — the target occurrences are a
                            // subsequence, so the main bracket cannot answer for them

    // "Distinguish Sends, Receives, Waitalls" state (colorMode 'distinguish') — classifies each
    // call by function name into a bucket, unlike the heatmaps above there's no cross-rank
    // statistical computation, just a per-event color lookup.
    distinguishUseGradient: false,
    distinguishBetweenMode: 'custom', // 'custom' | 'keep' | 'gradient'
    distinguishFlatColor: new THREE.Color(0x000000), distinguishFlatAlpha: 0,           // flat-mode Comp
    distinguishCompBeginColor: new THREE.Color(0x000000), distinguishCompBeginAlpha: 0, // gradient-mode Comp begin
    distinguishCompEndColor: new THREE.Color(0x000000), distinguishCompEndAlpha: 0,     // gradient-mode Comp end
    // Per-bucket {enter, leave} color/alpha lookup: flat mode uses the same color for both
    // (renders as one flat color per call); gradient mode uses the bucket's own begin/end pair.
    distinguishBucketColors: {}, // { [bucket]: { enter: {color,alpha}, leave: {color,alpha} } }
    // Render-time lookup built from the above: region id -> its bucket's {enter, leave}.
    // Classification stays here rather than in the trace file so the function-name lists below
    // are not frozen at conversion time or duplicated in a second language.
    distinguishBucketOfRegion: null, // Array<{enter, leave}> indexed by region id

    // "PAPI counter heatmap" state (colorMode 'papi') — a hardware-counter rate or ratio for
    // whichever segment each rank is currently in, ranked across ranks live every frame.
    // Unlike the per-call heatmaps there is nothing precomputed: the formula is a UI choice,
    // and the cross-rank comparison at an instant needs only the ranks' current rows.
    papiFormula: 'ipc',
    papiCallCols: null,   // {num: (string|null)[], den: (string|null)[]|null} for in-call
    papiGapCols: null,    // ... and for the Comp gap after an interval. null entry = the
                          // counter exists but never moves, so its increment is exactly 0.
    papiVals: null, papiPct: null, papiValid: null,   // per-scheme, not shared: Dual View
                                                      // runs three schemes in one frame
    papiOrder: null, papiCompact: null,               // scratch for the NaN-free ranking
    papiUnavailable: null,  // why this trace cannot answer, or null
    papiNoValueColor: new THREE.Color(0x202020), papiNoValueAlpha: 0.25,
    legendSuffix: '',       // names the formula, which the <option> text cannot
  };
}

// The hardware-counter metrics. Each names the counters it needs; the viewer resolves those
// to columns once a trace is open, since which counters a trace recorded is a property of the
// file. `perTime` divides by elapsed time instead of by a second counter, giving a rate.
//
// Only increments are ever read. A PAPI counter's absolute value is a per-core total since
// boot — it differs by billions between ranks and means nothing on its own — so the converter
// stores the increment over each call and over each Comp gap, and every formula here is a
// ratio or a rate built from those.
const PAPI_FORMULAS = {
  ipc: {
    label: 'IPC',
    unit: 'instructions / cycle',
    num: ['PAPI_TOT_INS'], den: ['PAPI_TOT_CYC'], perTime: false,
  },
  clock: {
    label: 'Effective clock',
    unit: 'GHz',
    // Cycles actually retired per unit of wall time. Reads near zero for a rank parked in
    // MPI and near the nominal clock for one that is executing, which no timing metric shows.
    num: ['PAPI_TOT_CYC'], den: null, perTime: true, scale: 1e-9,
  },
  flops: {
    label: 'FLOP rate',
    unit: 'GFLOP/s',
    num: ['PAPI_DP_OPS', 'PAPI_SP_OPS'], den: null, perTime: true, scale: 1e-9,
  },
  vec: {
    label: 'Vector lane use',
    unit: 'double-precision ops per vector instruction; 1 = scalar, 8 = AVX-512',
    num: ['PAPI_DP_OPS'], den: ['PAPI_VEC_DP'], perTime: false,
  },
};

// scheme1 always drives single-view mode (and VR fallback when Dual View is off — see
// renderFrame()). scheme2/scheme3 are only populated/rendered when dualViewActive is true.
let scheme1 = createColorScheme();
let scheme2 = createColorScheme();
let scheme3 = createColorScheme();
let dualViewActive = false;

// Legend content, rebuilt once per Visualize click by refreshLegendPanels() — { title,
// entries } per side, or null when that side has no legend to show. Read by both the DOM
// panels and the recording burn-in (drawLegendOverlays()) so they always show identical content.
let legendCache = { left: null, right: null };

// Per-scheme color/alpha buffers, (re)allocated to match the active geometry in
// initVisualization(). Buffer 1 is always allocated; 2/3 only when dualViewActive.
let colorBuffer1 = null, alphaBuffer1 = null;
let colorBuffer2 = null, alphaBuffer2 = null;
let colorBuffer3 = null, alphaBuffer3 = null;

// DOM elements
const elements = {
  rendererContainer: document.getElementById('renderer-container'),
  slider: document.getElementById('slider'),
  playPauseBtn: document.getElementById('play-pause'),
  prevFrameBtn: document.getElementById('prev-frame'),
  nextFrameBtn: document.getElementById('next-frame'),
  clockEl: document.getElementById('clock'),
  infoPanel: document.getElementById('info-panel'),
  controlsPanel: document.getElementById('controls-panel'),
  recordBtn: document.getElementById('record-btn'),
  recordingIndicator: document.getElementById('recording-indicator'),
  timelineCurrentTime: document.getElementById('timelineCurrentTime'),
  timelineMaxTime: document.getElementById('timelineMaxTime'),
  legendPanelLeft: document.getElementById('legend-panel-left'),
  legendPanelRight: document.getElementById('legend-panel-right'),
  detailStatus: document.getElementById('detail-status'),
  detailWarning: document.getElementById('detail-warning'),
  loadingStatus: document.getElementById('loading-status'),
  detailLevelSelect: document.getElementById('detailLevel'),
  clearSelectionBtn: document.getElementById('clear-selection'),
  detailLevelGroup: document.getElementById('detailLevelGroup'),
};

function setLoadingIndicator(on) {
  elements.loadingStatus?.classList.toggle('hidden', !on);
}

/**
 * `Ignore MPI_Init` and `Begin time` both drop intervals, and every derived column in a
 * .pv3dbin is a ranking over the intervals that exist — re-filtering in the viewer would leave
 * the stored percentiles describing a set that is no longer on screen. So for that format they
 * become read-only indicators of what the conversion already did; for legacy `.txt`, where the
 * viewer still computes everything itself, they keep working.
 */
function applyFilterIndicators(src) {
  const isPv3d = src.format === 'pv3dbin';
  const ignoreEl = document.getElementById('ignoreMpiInit');
  const beginEl = document.getElementById('beginTime');
  const opts = isPv3d ? (src.file.header.options || {}) : null;

  if (ignoreEl) {
    ignoreEl.disabled = isPv3d;
    if (isPv3d) {
      const dropped = (opts.region_list || []).includes('MPI_Init')
        && opts.region_list_mode === 'ignore';
      ignoreEl.checked = dropped || !src.regionNames.includes('MPI_Init');
      ignoreEl.title = 'Set when converting: otf2_to_pv3d.py --region-list MPI_Init '
                     + '--region-list-mode ignore';
    } else {
      ignoreEl.title = '';
    }
  }
  if (beginEl) {
    beginEl.disabled = isPv3d;
    if (isPv3d) {
      const cyc = opts.begin_time_arg || 0;
      beginEl.value = cyc ? String(cyc) : '0';
      beginEl.title = 'Set when converting: otf2_to_pv3d.py --begin-time';
    } else {
      beginEl.title = '';
    }
  }
}

/**
 * Says which detail level is being drawn, and — when that level dropped intervals — how much
 * of the trace is actually on screen. Dropping is not a blur: a rank inside a dropped 27 us
 * call renders as Comp, so the number has to be visible rather than implied.
 */
function updateDetailStatus(src) {
  const info = src && src.levelInfo;
  // The legacy text format has exactly one level, so the control would do nothing there.
  elements.detailLevelGroup?.classList.toggle('hidden', !info);
  if (!info) {
    elements.detailStatus?.classList.add('hidden');
    elements.detailWarning?.classList.add('hidden');
    return;
  }
  elements.detailStatus.textContent =
    `Detail level ${info.level} (quantum ${info.quant} cycles)`;
  elements.detailStatus.classList.remove('hidden');

  if (info.dropped > 0) {
    const shown = info.nSource - info.dropped;
    const pct = info.nSource ? (100 * shown / info.nSource) : 100;
    let text =
      `Showing ${shown.toLocaleString()} of ${info.nSource.toLocaleString()} intervals `
      + `(${pct.toFixed(1)}%) — level ${info.level} omits calls shorter than `
      + `${info.quant} cycles. Choose a finer detail level to see them.`;
    // A counter metric is hit harder than a timing one. The increments were measured over the
    // *full* interval sequence and then filtered, so at a lossy level a "Comp gap" increment
    // covers the dropped MPI calls as well as the computation — measured on this trace, IPC in
    // gaps reads 2.56 at level 0 and 0.45 at level 2 for that reason.
    const papiActive = (dualViewActive ? [scheme1, scheme2, scheme3] : [scheme1])
      .some(sc => sc.colorMode === 'papi');
    if (papiActive) {
      text += ' Counter values are affected more than timings: a Comp gap at this level spans '
            + 'the calls that were dropped, so its rate mixes them in. Use level 0 to read '
            + 'them faithfully.';
    }
    elements.detailWarning.textContent = text;
    elements.detailWarning.classList.remove('hidden');
  } else {
    elements.detailWarning.classList.add('hidden');
  }
}

/**
 * Sets the colour behind the visualization.
 *
 * Also paints the wrapper element, which is otherwise what shows through for the instant
 * between a window resize and the next frame — a black flash on a light background.
 * The recording path needs nothing: it composites `renderer.domElement`, so the new colour
 * is already in the pixels it copies.
 */
function setSceneBackground(hex) {
  viewControls.background = hex;
  if (scene) {
    if (!scene.background || !scene.background.isColor) scene.background = new THREE.Color();
    scene.background.set(hex);
  }
  const wrapper = document.querySelector('.viz-wrapper');
  if (wrapper) wrapper.style.backgroundColor = hex;
}

// Initialize Three.js scene
function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(viewControls.background);

  const width = elements.rendererContainer.clientWidth;
  const height = elements.rendererContainer.clientHeight;
  const fov = 33.4;

  camera = new THREE.PerspectiveCamera(fov, width / height, 0.1, 1000);
  camera.position.set(0, 0, 25); // Desktop camera position


  // Create dolly for VR locomotion (camera added to dolly only when entering VR)
  dolly = new THREE.Group();
  dolly.position.set(0, 0, 25);
  scene.add(dolly);
  scene.add(camera); // Camera starts in scene for desktop controls

  // Container for all visualization objects — translated/rotated by VR controls
  vizGroup = new THREE.Group();
  scene.add(vizGroup);

  // Anchor group for the vampir HUD in VR — follows dolly each frame
  vrHudParent = new THREE.Group();
  scene.add(vrHudParent);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.xr.enabled = true;

  document.body.appendChild( VRButton.createButton( renderer ) );
  elements.rendererContainer.appendChild(renderer.domElement);

  // Handle VR session start/end to move camera in/out of dolly
  renderer.xr.addEventListener('sessionstart', () => {
    scene.remove(camera);
    camera.position.set(0, 1.6, 0); // Eye height in VR
    dolly.add(camera);
    controls.enabled = false;
    gsap.ticker.sleep();
    document.getElementById('exit-vr-btn').style.display = 'inline-block';
    if (vampirHudMesh) positionHUDForVR();
    if (isRecording()) stopRecording(); // canvas capture doesn't reflect the XR framebuffer
    if (elements.recordBtn) elements.recordBtn.style.display = 'none';
  });

  renderer.xr.addEventListener('sessionend', () => {
    dolly.remove(camera);
    camera.position.copy(dolly.position);
    camera.position.y = 0;
    scene.add(camera);
    controls.enabled = true;
    controls.update();
    xrLastTime = null;
    gsap.ticker.wake();
    document.getElementById('exit-vr-btn').style.display = 'none';
    if (vampirHudMesh) positionHUDForDesktop();
    camera.remove(vrControlsGroup);
    scene.add(vrControlsGroup);
    if (elements.recordBtn) elements.recordBtn.style.display = '';
  });

  // Setup VR controllers
  const controllerModelFactory = new XRControllerModelFactory();

  controller1 = renderer.xr.getController(0);
  dolly.add(controller1);

  controller2 = renderer.xr.getController(1);
  dolly.add(controller2);

  controllerGrip1 = renderer.xr.getControllerGrip(0);
  controllerGrip1.add(controllerModelFactory.createControllerModel(controllerGrip1));
  dolly.add(controllerGrip1);

  controllerGrip2 = renderer.xr.getControllerGrip(1);
  controllerGrip2.add(controllerModelFactory.createControllerModel(controllerGrip2));
  dolly.add(controllerGrip2);

  function onSqueezeStart(event) {
    const ctrl = event.target;
    if (!vizGroup || ctrl.userData.dragging) return;
    const other = ctrl === controller1 ? controller2 : controller1;

    if (other.userData.dragging && !pinching) {
      // Both grips now held → switch to pinch-scale mode.
      // Return vizGroup to scene first (detach from the first controller that grabbed it).
      scene.attach(vizGroup);
      other.userData.dragging = false;
      ctrl.userData.dragging = false;
      pinching = true;
      // Record baseline: distance between controllers and current uniform scale.
      const p1 = new THREE.Vector3();
      const p2 = new THREE.Vector3();
      controller1.getWorldPosition(p1);
      controller2.getWorldPosition(p2);
      pinchStartDist = Math.max(0.001, p1.distanceTo(p2));
      pinchStartScale = vizGroup.scale.x;
    } else if (!pinching) {
      // Single grip drag
      ctrl.userData.dragging = true;
      ctrl.attach(vizGroup);
    }
  }

  function onSqueezeEnd(event) {
    const ctrl = event.target;
    if (pinching) {
      // One grip released → end pinch, freeze scale where it is.
      pinching = false;
      controller1.userData.dragging = false;
      controller2.userData.dragging = false;
      // vizGroup is already in scene during pinch
    } else if (ctrl.userData.dragging) {
      ctrl.userData.dragging = false;
      scene.attach(vizGroup);
    }
  }

  controller1.addEventListener('squeezestart', onSqueezeStart);
  controller1.addEventListener('squeezeend', onSqueezeEnd);
  controller2.addEventListener('squeezestart', onSqueezeStart);
  controller2.addEventListener('squeezeend', onSqueezeEnd);

  // Exit VR button (visible only while in session)
  document.getElementById('exit-vr-btn').addEventListener('click', () => {
    renderer.xr.getSession()?.end();
  });

  // Add laser pointers to controllers
  const laserGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -5)
  ]);
  const laserMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });

  const laser1 = new THREE.Line(laserGeometry, laserMaterial);
  controller1.add(laser1);

  const laser2 = new THREE.Line(laserGeometry, laserMaterial);
  controller2.add(laser2);

  // Create VR interactive controls group (only listens to XR controllers, not mouse)
  vrControlsGroup = new InteractiveGroup();
  vrControlsGroup.listenToXRControllerEvents(controller1);
  vrControlsGroup.listenToXRControllerEvents(controller2);
  scene.add(vrControlsGroup);

  // Create VR controls mesh from the controls panel HTML
  function createVRControls() {
    if (vrControlsMesh) {
      vrControlsGroup.remove(vrControlsMesh);
    }
    vrControlsMesh = new HTMLMesh(elements.controlsPanel);
    vrControlsMesh.scale.setScalar(2);
    vrControlsGroup.add(vrControlsMesh);
  }

  // Position VR controls at the bottom of the field of view when entering VR
  renderer.xr.addEventListener('sessionstart', () => {
    createVRControls();
    // Parent to camera so controls always face the user (true HUD)
    scene.remove(vrControlsGroup);
    camera.add(vrControlsGroup);
    // Camera-local position: below eye level, tilted like a music stand
    vrControlsMesh.position.set(0, -0.55, -1.0);
    vrControlsMesh.rotation.x = Math.PI / 6;
    if (vampirHudMesh) positionHUDForVR();
  });

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enabled = true;
  controls.update();

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
  scene.add(ambientLight);

  pointLightUpper = new THREE.PointLight(0xffffff, 5, 0, 0);
  pointLightUpper.position.set(5, 200, 25);
  scene.add(pointLightUpper);

  pointLightLower = new THREE.PointLight(0xffffff, 5, 0, 0);
  pointLightLower.position.set(-5, -200, -25);
  scene.add(pointLightLower);

  // Add light helpers to visualize light positions
  helperUpper = new THREE.PointLightHelper(pointLightUpper, 1);
  scene.add(helperUpper);

  helperLower = new THREE.PointLightHelper(pointLightLower, 1);
  scene.add(helperLower);

  // Setup GUI for light controls
  setupLightGUI();

  // // Load GLTF model
  // const loader = new GLTFLoader();
  // loader.load(
  //   'ABeautifulGame.glb',
  //   //'kitchen.glb',
  //   (gltf) => {
  //     const model = gltf.scene;
  //     model.scale.set(10, 10, 10);
  //     scene.add(model);
  //   },
  //   (progress) => {
  //     console.log('Loading model:', (progress.loaded / progress.total * 100).toFixed(1) + '%');
  //   },
  //   (error) => {
  //     console.error('Error loading GLTF model:', error);
  //   }
  // );
}

function createVampirHUD(canvas) {
  vampirTexture = new THREE.CanvasTexture(canvas);
  vampirTexture.minFilter = THREE.LinearFilter;
  vampirTexture.magFilter = THREE.LinearFilter;

  const hudGeometry = new THREE.PlaneGeometry(1, 1);
  const hudMaterial = new THREE.MeshBasicMaterial({
    map: vampirTexture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  vampirHudMesh = new THREE.Mesh(hudGeometry, hudMaterial);
  vampirHudMesh.renderOrder = 9999;

  // Always attach to camera so it acts as a true HUD in both desktop and VR
  camera.add(vampirHudMesh);
  if (renderer.xr.isPresenting) {
    positionHUDForVR();
  } else {
    positionHUDForDesktop();
  }
  vampirReady = true;
}

function destroyVampirHUD() {
  if (vampirHudMesh) {
    // Remove from whatever parent currently holds it
    vampirHudMesh.parent?.remove(vampirHudMesh);
    vampirHudMesh.geometry.dispose();
    vampirHudMesh.material.dispose();
    vampirHudMesh = null;
  }
  if (vampirTexture) {
    vampirTexture.dispose();
    vampirTexture = null;
  }
  vampirReady = false;
}

function positionHUDForDesktop() {
  if (!vampirHudMesh) return;
  const fovRad = THREE.MathUtils.degToRad(camera.fov);
  const d = 2.0;
  const visibleH = 2 * d * Math.tan(fovRad / 2);
  const visibleW = visibleH * camera.aspect;

  // Full viewport width, 25% of viewport height
  const hudWidth = visibleW;
  const hudHeight = visibleH * 0.25;

  // Margin from bottom edge for the GSAP controls overlay
  const margin = visibleH * 0.07;
  const bottomY = -visibleH / 2;
  const hudCenterY = bottomY + margin + hudHeight / 2;

  vampirHudMesh.scale.set(hudWidth, hudHeight, 1);
  vampirHudMesh.position.set(0, hudCenterY, -d);
  vampirHudMesh.rotation.set(0, 0, 0);
}

function positionHUDForVR() {
  if (!vampirHudMesh) return;
  // Camera-local space: floats straight ahead at roughly eye level.
  // Scale preserves the 2:1 canvas aspect ratio (1200×600).
  vampirHudMesh.scale.set(1.2, 0.6, 1);
  vampirHudMesh.position.set(0, -0.05, -1.0);
  vampirHudMesh.rotation.set(0, 0, 0);
}

function addLightFolder(name, light, helper = null) {
  const folder = gui.addFolder(name);
  folder.add(light.position, 'x', -300, 300).name('X').onChange(() => helper?.update());
  folder.add(light.position, 'y', -300, 300).name('Y').onChange(() => helper?.update());
  folder.add(light.position, 'z', -300, 300).name('Z').onChange(() => helper?.update());
  folder.add(light, 'intensity', 0, 20).name('Intensity');
  folder.addColor({ color: '#ffffff' }, 'color').name('Color').onChange((val) => {
    light.color.set(val);
    helper?.update();
  });
  return folder;
}

const sphereLightFolders = [];
const viewControls = { roll: 0, background: '#000000' };

// Camera placed on the given side of the target, looking back toward it
// (e.g. '+X' = camera on the +X side). Y-axis views need a perpendicular
// `up` since the default (0,1,0) is parallel to the view direction there,
// which is a degenerate case for orbit-style controls.
const AXIS_VIEWS = {
  '+X': { dir: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 1, 0) },
  '-X': { dir: new THREE.Vector3(-1, 0, 0), up: new THREE.Vector3(0, 1, 0) },
  '+Y': { dir: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, -1) },
  '-Y': { dir: new THREE.Vector3(0, -1, 0), up: new THREE.Vector3(0, 0, 1) },
  '+Z': { dir: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0) },
  '-Z': { dir: new THREE.Vector3(0, 0, -1), up: new THREE.Vector3(0, 1, 0) },
};

function setAxisView(axisKey) {
  const { dir, up } = AXIS_VIEWS[axisKey];
  const radius = camera.position.distanceTo(controls.target) || 25;
  camera.up.copy(up);
  camera.position.copy(controls.target).addScaledVector(dir, radius);
  controls.update();
}

function setupLightGUI() {
  gui = new GUI({ title: 'Lights & View' });
  gui.close();
  addLightFolder('Upper Light', pointLightUpper, helperUpper);
  addLightFolder('Lower Light', pointLightLower, helperLower);

  const viewFolder = gui.addFolder('View');
  // Live, because it is a thing you tune by eye — and because a figure for a paper usually
  // wants a white background, which is awkward to ask for through the Visualize round trip.
  viewFolder.addColor(viewControls, 'background')
    .name('Background')
    .onChange(setSceneBackground);
  viewFolder.add(viewControls, 'roll', -180, 180, 1)
    .name('Depth Rotation (°)')
    .onChange((deg) => {
      vizGroup.rotation.z = THREE.MathUtils.degToRad(deg);
    });

  const axisViewActions = {};
  for (const axisKey of Object.keys(AXIS_VIEWS)) {
    axisViewActions[axisKey] = () => setAxisView(axisKey);
    viewFolder.add(axisViewActions, axisKey).name(`View from ${axisKey}`);
  }
}

// Renders the scene either as a single full-canvas pass (the normal single-view path, and
// also the VR fallback below) or, when Dual View is active outside VR, as two scissor-clipped
// half-canvas passes (scheme2 on the left, scheme3 on the right) sharing one camera/
// OrbitControls, with the camera's aspect adjusted per half and restored afterward. Replaces
// every direct renderer.render(scene, camera) call in the module.
function renderFrame() {
  const inVR = renderer.xr.isPresenting;
  const dual = dualViewActive && !inVR;
  const isSphere = sphereMesh !== null;

  if (!dual) {
    // VR fallback while Dual View is checked renders scheme2 (the split's left/primary half)
    // full-screen instead of splitting; otherwise this is just today's single-view path,
    // already kept in sync with scheme1 by updateInstanceColors().
    if (dualViewActive) {
      syncSchemeBufferToLiveAttribute(colorBuffer2, alphaBuffer2, isSphere);
    }
    renderer.setScissorTest(false);
    renderer.render(scene, camera);
    return;
  }

  // Logical (CSS) pixel size — the same units passed to renderer.setSize(); setViewport/
  // setScissor scale by devicePixelRatio internally, so using renderer.domElement.width/height
  // (already DPR-scaled) here would render into only a corner of the canvas on Hi-DPI displays.
  const fullW = elements.rendererContainer.clientWidth;
  const fullH = elements.rendererContainer.clientHeight;
  const halfW = Math.floor(fullW / 2);
  const savedAspect = camera.aspect;

  renderer.setScissorTest(true);

  // Left half: scheme2 / type2
  syncSchemeBufferToLiveAttribute(colorBuffer2, alphaBuffer2, isSphere);
  camera.aspect = halfW / fullH;
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, halfW, fullH);
  renderer.setScissor(0, 0, halfW, fullH);
  renderer.render(scene, camera);

  // Right half: scheme3 / type3
  syncSchemeBufferToLiveAttribute(colorBuffer3, alphaBuffer3, isSphere);
  camera.aspect = (fullW - halfW) / fullH;
  camera.updateProjectionMatrix();
  renderer.setViewport(halfW, 0, fullW - halfW, fullH);
  renderer.setScissor(halfW, 0, fullW - halfW, fullH);
  renderer.render(scene, camera);

  // Restore full-canvas state so anything that runs between frames (hover raycasting via
  // hudRaycast(), the resize handler, etc.) sees the camera/renderer in the state it expects.
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, fullW, fullH);
  camera.aspect = savedAspect;
  camera.updateProjectionMatrix();
}

function animate() {
  controls.update();

  // VR thumbstick controls
  if (renderer.xr.isPresenting) {
    const session = renderer.xr.getSession();
    if (session) {
      for (const source of session.inputSources) {
        if (!source.gamepad) continue;
        const axes = source.gamepad.axes;
        if (axes.length < 4) continue;
        const stickX = axes[2];
        const stickY = axes[3];
        const speed = 0.1;

        if (source.handedness === 'right') {
          // Right stick: rotate the visualization (X=horizontal, Y=vertical)
          vizGroup.rotation.y -= stickX * 0.02;
          vizGroup.rotation.x = THREE.MathUtils.clamp(
            vizGroup.rotation.x - stickY * 0.02,
            -Math.PI / 2, Math.PI / 2
          );
        } else {
          // Left stick: translate the visualization (X=horizontal, Y=vertical)
          const camRight = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
          camRight.y = 0;
          camRight.normalize();
          vizGroup.position.addScaledVector(camRight, stickX * speed);
          vizGroup.position.y -= stickY * speed;
        }
      }
    }

    // Pinch-to-scale: both grip buttons held → scale vizGroup by inter-controller ratio
    if (pinching) {
      const p1 = new THREE.Vector3();
      const p2 = new THREE.Vector3();
      controller1.getWorldPosition(p1);
      controller2.getWorldPosition(p2);
      const currentDist = Math.max(0.001, p1.distanceTo(p2));
      const newScale = Math.max(0.05, pinchStartScale * (currentDist / pinchStartDist));
      vizGroup.scale.setScalar(newScale);
    }

    if (vrControlsMesh) vrControlsMesh.material.map?.update();
  }

  // In XR mode, GSAP's window.rAF ticker is asleep; advance the timeline
  // manually using real elapsed time (performance.now() is in ms, convert to s).
  if (renderer.xr.isPresenting) {
    if (masterTimeline && !masterTimeline.paused()) {
      const now = performance.now();
      if (xrLastTime !== null) {
        const dt = (now - xrLastTime) / 1000;
        const newTime = Math.min(masterTimeline.duration(), masterTimeline.time() + dt);
        masterTimeline.seek(newTime, false);
      }
      xrLastTime = now;
    } else {
      xrLastTime = null; // reset so the first frame after unpause doesn't jump
    }
  }

  if (masterTimeline && !masterTimeline.paused()) {
    elements.slider.value = masterTimeline.progress().toString();

    // Timejump: if the next event across all ranks is farther than the threshold,
    // skip ahead to 0.5 s before it.
    //
    // The threshold and the 0.5 s lead-in are trace time, so the comparison has to be made
    // against `timeProxy.time` (where the playhead is *in the trace*). It used to read
    // masterTimeline.time(), which is playback time — a 1.3 s trace at 100 µs/frame and 60 fps
    // plays for 218 s, so the two ran on completely different scales and the feature could
    // only ever fire by accident.
    if (traceSource && currentMaxTicks > 0 && document.getElementById('activateTimejumps')?.checked) {
      const thresholdTicks = (parseFloat(document.getElementById('timejumpThreshold')?.value) || 5)
        * traceSource.ticksPerSecond;
      const currentTicks = timeProxy.time;
      const nextEventTicks = traceSource.nextEventAfter(currentTicks);
      // Ignore a gap that leads out of the play span: inside a selection the point is to stay
      // in it, and jumping to an event beyond the end flag would escape the restriction.
      if (nextEventTicks !== Infinity && nextEventTicks <= playEnd
          && nextEventTicks - currentTicks > thresholdTicks) {
        const targetTicks = Math.max(playStart, nextEventTicks - 0.5 * traceSource.ticksPerSecond);
        masterTimeline.seek(masterTimeline.duration() * timeToProgress(targetTicks), false);
      }
    }
  }

  // Update vampir HUD texture each frame when timeline is playing
  if (vampirReady && vampirTexture && masterTimeline && !masterTimeline.paused()) {
    vampirTexture.needsUpdate = true;
  }

  // Auto-stop recording once a full playback pass completes.
  if (isRecording() && masterTimeline && masterTimeline.progress() >= 1) {
    stopRecording();
  }

  renderFrame();

  // Burn the clock/tooltip HTML overlays into the composite canvas so they
  // show up in the recording (captureStream() only sees renderer.domElement,
  // which never contains DOM elements positioned on top of it via CSS).
  if (isRecording() && compositeCtx) {
    compositeCtx.drawImage(renderer.domElement, 0, 0);
    drawClockOverlay(compositeCtx);
    drawTooltipOverlay(compositeCtx);
    drawLegendOverlays(compositeCtx);
  }
}


// Maps a percentile (0-100) to a color by walking the user-given cutoffs
// (highest -> lowest): 100 -> heatmapBucketColors[0] ("Color 1"), cutoffs[0]
// -> heatmapBucketColors[1] ("Color 2"), etc. Between two anchors, linearly
// interpolates weighted by proximity. Below the lowest cutoff, clamps to the
// last color (no lower anchor was specified).
//
// This used to run once per event at load time, filling per-event color tables; it now runs
// per visible rank per frame (~3,500 calls) straight off a percentile column. `out` is a
// caller-owned {color, alpha} scratch so the frame allocates nothing — the old version's
// clone() per interpolated color was affordable exactly once, not sixty times a second.
function percentileToBucketColor(scheme, percentile, out) {
  const buckets = scheme.heatmapBucketColors;
  const cutoffs = scheme.heatmapCutoffs;
  if (buckets.length === 0) { out.color.copy(scheme.compColor); out.alpha = scheme.compColorAlpha; return out; }
  if (cutoffs.length === 0 || percentile >= 100 || buckets.length === 1) return copyBucket(buckets[0], out);

  if (percentile >= cutoffs[0]) {
    const span = 100 - cutoffs[0];
    const t = span > 0 ? (100 - percentile) / span : 1;
    return lerpBucket(buckets[0], buckets[1] ?? buckets[0], t, out);
  }
  for (let k = 0; k < cutoffs.length - 1; k++) {
    if (percentile >= cutoffs[k + 1]) {
      const span = cutoffs[k] - cutoffs[k + 1];
      const t = span > 0 ? (cutoffs[k] - percentile) / span : 1;
      return lerpBucket(buckets[k + 1] ?? buckets[buckets.length - 1], buckets[k + 2] ?? buckets[buckets.length - 1], t, out);
    }
  }
  return copyBucket(buckets[buckets.length - 1], out);
}

function copyBucket(b, out) {
  out.color.copy(b.color);
  out.alpha = b.alpha;
  return out;
}

function lerpBucket(a, b, t, out) {
  out.color.copy(a.color).lerp(b.color, t);
  out.alpha = a.alpha + (b.alpha - a.alpha) * t;
  return out;
}

// Two scratch buckets are enough: every branch below needs at most an "enter"/"start" colour
// and a "leave"/"end" colour live at the same time.
const bucketA = { color: new THREE.Color(), alpha: 1 };
const bucketB = { color: new THREE.Color(), alpha: 1 };

// "Distinguish Sends, Receives, Waitalls" — classifies each call by function name into one
// of 4 buckets. Explicit lists rather than substring matching, so an unrecognized MPI
// function name falls into "Other MPI calls" rather than being silently mis-bucketed.
const DISTINGUISH_SENDS = new Set([
  'MPI_Send', 'MPI_Isend', 'MPI_Bsend', 'MPI_Ibsend', 'MPI_Ssend', 'MPI_Issend',
  'MPI_Rsend', 'MPI_Irsend', 'MPI_Send_init', 'MPI_Bsend_init', 'MPI_Ssend_init', 'MPI_Rsend_init',
]);
const DISTINGUISH_RECEIVES = new Set(['MPI_Recv', 'MPI_Irecv', 'MPI_Recv_init']);
const DISTINGUISH_WAITALLS = new Set(['MPI_Wait', 'MPI_Waitall', 'MPI_Waitany', 'MPI_Waitsome']);

function classifyMpiFunction(fn) {
  if (DISTINGUISH_SENDS.has(fn)) return 'Sends';
  if (DISTINGUISH_RECEIVES.has(fn)) return 'Receives';
  if (DISTINGUISH_WAITALLS.has(fn)) return 'Waitalls';
  return 'Other MPI calls';
}


// Scoped equivalent of the single global colorMode/iterationColors/... assignment this replaced.
function applyMetricConfigToScheme(metricType, colors, extra, target) {
  if (metricType === 'coloredIterations') {
    target.colorMode = 'iterations';
    target.iterationColors = [];
    target.iterationAlphas = [];
    for (let n = 1; ; n++) {
      const key = `Color #${n}`;
      if (!colors[key]) break;
      target.iterationColors.push(new THREE.Color(colors[key].r, colors[key].g, colors[key].b));
      target.iterationAlphas.push(colors[key].a ?? 1.0);
    }
    if (target.iterationColors.length === 0) {
      target.iterationColors.push(new THREE.Color(1, 0, 0), new THREE.Color(0, 1, 0), new THREE.Color(0, 0, 1));
      target.iterationAlphas.push(1.0, 1.0, 1.0);
    }
  } else if (metricType === 'heatmapSpeeders' || metricType === 'compHeatmapSpeeders' || metricType === 'mpiHeatmapSpeedLoop'
          || metricType === 'compHeatmapSpeedLoop' || metricType === 'compHeatmapSpeedRun') {
    target.colorMode = 'heatmap';
    target.heatmapAppliesTo = metricType === 'compHeatmapSpeeders' ? 'comp'
                      : metricType === 'mpiHeatmapSpeedLoop' ? 'mpiDuration'
                      : metricType === 'compHeatmapSpeedLoop' ? 'compDuration'
                      : metricType === 'compHeatmapSpeedRun' ? 'compRun'
                      : 'mpi';
    target.iterationColors = [];

    // "Color 1" (100th percentile / base) through "Color N+1" (lowest cutoff)
    target.heatmapBucketColors = [];
    for (let n = 1; ; n++) {
      const key = `Color ${n}`;
      if (!colors[key]) break;
      target.heatmapBucketColors.push({
        color: new THREE.Color(colors[key].r, colors[key].g, colors[key].b),
        alpha: colors[key].a ?? 1.0,
      });
    }
    if (target.heatmapBucketColors.length === 0) {
      target.heatmapBucketColors.push({ color: new THREE.Color(1, 0, 0), alpha: 1.0 });
    }

    target.heatmapCutoffs = (extra.heatmapCutoffs || [])
      .map(v => parseFloat(v))
      .filter(v => !isNaN(v));

    target.heatmapBetweenMode = extra.betweenMpi || 'custom';

    // Flat color for whichever role plays "the gap": Comp for MPI-mode, MPI for Comp-mode
    // (both compDuration and compRun are still "Comp-mode" — the gap is an MPI call).
    const flatKey = (target.heatmapAppliesTo === 'comp' || target.heatmapAppliesTo === 'compDuration' || target.heatmapAppliesTo === 'compRun') ? 'MPI' : 'Comp';
    if (colors[flatKey]) {
      target.heatmapFlatColor.setRGB(colors[flatKey].r, colors[flatKey].g, colors[flatKey].b);
      target.heatmapFlatAlpha = colors[flatKey].a ?? 1.0;
    }
  } else if (metricType === 'mpiHeatmapSpeedRun') {
    target.colorMode = 'speedHeatmapRun';
    target.iterationColors = [];
    target.runTargetFn = (extra.runTargetFn || '').trim();

    // "Color 1" (100th percentile / base) through "Color N+1" (lowest cutoff) — same
    // cutoff/color population pattern as the other heatmap-family metrics.
    target.heatmapBucketColors = [];
    for (let n = 1; ; n++) {
      const key = `Color ${n}`;
      if (!colors[key]) break;
      target.heatmapBucketColors.push({
        color: new THREE.Color(colors[key].r, colors[key].g, colors[key].b),
        alpha: colors[key].a ?? 1.0,
      });
    }
    if (target.heatmapBucketColors.length === 0) {
      target.heatmapBucketColors.push({ color: new THREE.Color(1, 0, 0), alpha: 1.0 });
    }

    target.heatmapCutoffs = (extra.heatmapCutoffs || [])
      .map(v => parseFloat(v))
      .filter(v => !isNaN(v));

    target.heatmapBetweenMode = extra.betweenMpi || 'custom';

    if (colors['Comp']) {
      target.heatmapFlatColor.setRGB(colors['Comp'].r, colors['Comp'].g, colors['Comp'].b);
      target.heatmapFlatAlpha = colors['Comp'].a ?? 1.0;
    }
  } else if (metricType === 'heatmapSpeed') {
    target.colorMode = 'speedHeatmap';
    target.iterationColors = [];

    // "Color 1" (100th percentile / base) through "Color N+1" (lowest cutoff) — same
    // cutoff/color population pattern as the other two heatmap metrics.
    target.heatmapBucketColors = [];
    for (let n = 1; ; n++) {
      const key = `Color ${n}`;
      if (!colors[key]) break;
      target.heatmapBucketColors.push({
        color: new THREE.Color(colors[key].r, colors[key].g, colors[key].b),
        alpha: colors[key].a ?? 1.0,
      });
    }
    if (target.heatmapBucketColors.length === 0) {
      target.heatmapBucketColors.push({ color: new THREE.Color(1, 0, 0), alpha: 1.0 });
    }

    target.heatmapCutoffs = (extra.heatmapCutoffs || [])
      .map(v => parseFloat(v))
      .filter(v => !isNaN(v));

    target.heatmapWindowUs = parseFloat(extra.windowSize) || 1000;
  } else if (metricType === 'papiHeatmap') {
    target.colorMode = 'papi';
    target.iterationColors = [];
    target.papiFormula = PAPI_FORMULAS[extra.papiFormula] ? extra.papiFormula : 'ipc';

    const spec = PAPI_FORMULAS[target.papiFormula];
    // The <option> text is all getMetricDisplayName() can see, and it cannot name the
    // formula, so the legend would otherwise claim "PAPI counter heatmap" while showing
    // vector-lane utilisation.
    target.legendSuffix = ` — ${spec.label}`;

    target.heatmapBucketColors = [];
    for (let n = 1; ; n++) {
      const key = `Color ${n}`;
      if (!colors[key]) break;
      target.heatmapBucketColors.push({
        color: new THREE.Color(colors[key].r, colors[key].g, colors[key].b),
        alpha: colors[key].a ?? 1.0,
      });
    }
    if (target.heatmapBucketColors.length === 0) {
      target.heatmapBucketColors.push({ color: new THREE.Color(1, 0, 0), alpha: 1.0 });
    }

    target.heatmapCutoffs = (extra.heatmapCutoffs || [])
      .map(v => parseFloat(v))
      .filter(v => !isNaN(v));

    const nv = colors['No value'];
    target.papiNoValueColor = nv ? new THREE.Color(nv.r, nv.g, nv.b) : new THREE.Color(0x202020);
    target.papiNoValueAlpha = nv ? (nv.a ?? 1.0) : 0.25;
  } else if (metricType === 'distinguish') {
    target.colorMode = 'distinguish';
    target.iterationColors = [];

    target.distinguishUseGradient = !!extra.useGradient;
    target.distinguishBetweenMode = extra.betweenMpi || 'custom';

    const readColor = (key) => colors[key]
      ? { color: new THREE.Color(colors[key].r, colors[key].g, colors[key].b), alpha: colors[key].a ?? 1.0 }
      : { color: new THREE.Color(0, 0, 0), alpha: 0 };

    for (const bucket of ['Sends', 'Receives', 'Waitalls', 'Other MPI calls']) {
      if (target.distinguishUseGradient) {
        target.distinguishBucketColors[bucket] = {
          enter: readColor(`${bucket} begin`),
          leave: readColor(`${bucket} end`),
        };
      } else {
        const c = readColor(bucket);
        target.distinguishBucketColors[bucket] = { enter: c, leave: c };
      }
    }

    if (target.distinguishUseGradient) {
      const beginC = readColor('Comp begin');
      const endC = readColor('Comp end');
      target.distinguishCompBeginColor.copy(beginC.color);
      target.distinguishCompBeginAlpha = beginC.alpha;
      target.distinguishCompEndColor.copy(endC.color);
      target.distinguishCompEndAlpha = endC.alpha;
    } else {
      const c = readColor('Comp');
      target.distinguishFlatColor.copy(c.color);
      target.distinguishFlatAlpha = c.alpha;
    }
  } else {
    target.colorMode = 'gradient';
    target.iterationColors = [];
    if (colors['MPI begin']) {
      target.mpiStartColor.setRGB(colors['MPI begin'].r, colors['MPI begin'].g, colors['MPI begin'].b);
      target.mpiStartAlpha = colors['MPI begin'].a ?? 1.0;
    }
    if (colors['MPI end']) {
      target.mpiEndColor.setRGB(colors['MPI end'].r, colors['MPI end'].g, colors['MPI end'].b);
      target.mpiEndAlpha = colors['MPI end'].a ?? 1.0;
    }
    if (colors['Comp']) {
      target.compColor.setRGB(colors['Comp'].r, colors['Comp'].g, colors['Comp'].b);
      target.compColorAlpha = colors['Comp'].a;
    }
    if (colors['MPI']) {
      target.mpiStartColor.setRGB(colors['MPI'].r, colors['MPI'].g, colors['MPI'].b);
      target.mpiStartAlpha = colors['MPI'].a ?? 1.0;
      target.mpiEndColor.copy(target.mpiStartColor);
      target.mpiEndAlpha = target.mpiStartAlpha;
    }
  }
}

// Builds the discrete percentile-range rows shared by every heatmap-family colorMode, aligned
// index-for-index with scheme.heatmapBucketColors. Mirrors percentileToBucketColor()'s anchor
// semantics: buckets[0] = 100th-percentile anchor, buckets[k+1] = cutoffs[k]-anchor, the last
// bucket is the flat clamp below the lowest cutoff. This is a legend-only simplification of a
// continuous per-pixel lerp — it labels each anchor as the range down to the next anchor, not
// exact interpolation math.
function heatmapFamilyEntries(scheme) {
  const buckets = scheme.heatmapBucketColors;
  const cutoffs = scheme.heatmapCutoffs;
  const out = [];
  if (buckets.length === 0) return out;

  if (cutoffs.length === 0) {
    out.push({ label: 'All values', color: buckets[0].color, alpha: buckets[0].alpha });
    return out;
  }

  buckets.forEach((b, i) => {
    let label;
    if (i === 0) {
      label = '100th percentile';
    } else if (i - 1 < cutoffs.length - 1) {
      label = `${cutoffs[i]}th–${cutoffs[i - 1]}th percentile`;
    } else if (i - 1 === cutoffs.length - 1) {
      label = `below ${cutoffs[cutoffs.length - 1]}th percentile`;
    } else {
      // More colors than cutoffs (mismatched config) — defensive fallback.
      label = `Color ${i + 1}`;
    }
    out.push({ label, color: b.color, alpha: b.alpha });
  });
  return out;
}

// A gap/flat color row, with a suffix noting when the gap isn't actually flat (its color
// depends on the adjacent call instead, per heatmapBetweenMode/distinguishBetweenMode).
function flatColorEntry(label, color, alpha, betweenMode) {
  const suffix = betweenMode === 'keep' ? ' (matches adjacent color)'
               : betweenMode === 'gradient' ? ' (blends between calls)'
               : '';
  return { label: `${label}${suffix}`, color, alpha };
}

// Mirror of applyMetricConfigToScheme(): given a populated scheme, returns an ordered list of
// legend rows {label, color: THREE.Color, alpha}. Branches on scheme.colorMode (not metricType
// directly) since several metricType values collapse onto the same colorMode.
function buildLegendEntries(scheme) {
  const entries = [];

  if (scheme.colorMode === 'iterations') {
    scheme.iterationColors.forEach((color, i) => {
      entries.push({ label: `Iteration ${i + 1}`, color, alpha: scheme.iterationAlphas[i] ?? 1.0 });
    });

  } else if (scheme.colorMode === 'heatmap' || scheme.colorMode === 'speedHeatmapRun') {
    entries.push(...heatmapFamilyEntries(scheme));
    // Flat/gap color label depends on which role plays "the gap" — mirrors the flatKey logic
    // in applyMetricConfigToScheme (Comp for MPI-mode/run-mode, MPI for Comp-mode/compDuration/compRun).
    const isCompMode = scheme.colorMode === 'heatmap'
      && (scheme.heatmapAppliesTo === 'comp' || scheme.heatmapAppliesTo === 'compDuration' || scheme.heatmapAppliesTo === 'compRun');
    const flatLabel = isCompMode ? 'MPI' : 'Comp';
    entries.push(flatColorEntry(flatLabel, scheme.heatmapFlatColor, scheme.heatmapFlatAlpha, scheme.heatmapBetweenMode));

  } else if (scheme.colorMode === 'speedHeatmap') {
    // Windowed pace: no gap concept — every instant is colored by live pace.
    entries.push(...heatmapFamilyEntries(scheme));

  } else if (scheme.colorMode === 'papi') {
    entries.push(...heatmapFamilyEntries(scheme));
    // Named explicitly rather than left as an unexplained grey: a rank gets no value at the
    // edges of the trace, inside a call whose counters were not recorded, or when the
    // formula's denominator did not advance.
    entries.push({ label: 'No value', color: scheme.papiNoValueColor, alpha: scheme.papiNoValueAlpha });

  } else if (scheme.colorMode === 'distinguish') {
    for (const bucket of ['Sends', 'Receives', 'Waitalls', 'Other MPI calls']) {
      const bc = scheme.distinguishBucketColors[bucket];
      if (!bc) continue;
      if (scheme.distinguishUseGradient) {
        entries.push({ label: `${bucket} (begin)`, color: bc.enter.color, alpha: bc.enter.alpha });
        entries.push({ label: `${bucket} (end)`, color: bc.leave.color, alpha: bc.leave.alpha });
      } else {
        entries.push({ label: bucket, color: bc.enter.color, alpha: bc.enter.alpha });
      }
    }
    if (scheme.distinguishUseGradient) {
      entries.push(flatColorEntry('Comp (begin)', scheme.distinguishCompBeginColor, scheme.distinguishCompBeginAlpha, scheme.distinguishBetweenMode));
      entries.push(flatColorEntry('Comp (end)', scheme.distinguishCompEndColor, scheme.distinguishCompEndAlpha, scheme.distinguishBetweenMode));
    } else {
      entries.push(flatColorEntry('Comp', scheme.distinguishFlatColor, scheme.distinguishFlatAlpha, scheme.distinguishBetweenMode));
    }

  } else {
    // 'gradient' — coloredMPI / coloredOutMPI / unrecognized fallback.
    if (scheme.mpiStartColor.equals(scheme.mpiEndColor) && scheme.mpiStartAlpha === scheme.mpiEndAlpha) {
      entries.push({ label: 'MPI', color: scheme.mpiStartColor, alpha: scheme.mpiStartAlpha });
    } else {
      entries.push({ label: 'MPI (begin)', color: scheme.mpiStartColor, alpha: scheme.mpiStartAlpha });
      entries.push({ label: 'MPI (end)', color: scheme.mpiEndColor, alpha: scheme.mpiEndAlpha });
    }
    entries.push({ label: 'Comp', color: scheme.compColor, alpha: scheme.compColorAlpha });
  }

  return entries;
}

// Reads the human-readable metric name straight out of #metricType1's <option> list (all
// three metricType selects duplicate identical option text) — avoids maintaining a second
// id->label map that could drift from mix.html's option text.
function getMetricDisplayName(metricType) {
  return document.querySelector(`#metricType1 option[value="${metricType}"]`)?.textContent || metricType;
}

// Fills a legend panel's title and swatch rows from a built entries list.
function renderLegendPanel(panelEl, title, entries) {
  panelEl.querySelector('.legend-title').textContent = title;
  const rowsEl = panelEl.querySelector('.legend-rows');
  rowsEl.innerHTML = '';
  for (const entry of entries) {
    const row = document.createElement('div');
    row.className = 'legend-row';

    const swatch = document.createElement('div');
    swatch.className = 'legend-swatch';
    const c = entry.color;
    swatch.style.backgroundColor = `rgba(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(c.b * 255)}, ${entry.alpha})`;
    row.appendChild(swatch);

    const label = document.createElement('span');
    label.textContent = entry.label;
    row.appendChild(label);

    rowsEl.appendChild(row);
  }
}

// Rebuilds legendCache and the on-screen panel(s) from the currently active scheme(s) — the
// left panel from scheme1 in single-view or scheme2 in Dual View, the right panel from scheme3
// when Dual View is active. Called once per Visualize click, not per frame.
function refreshLegendPanels() {
  const showLegend = !!config.metric?.showLegend;
  if (!showLegend) {
    elements.legendPanelLeft.classList.add('hidden');
    elements.legendPanelRight.classList.add('hidden');
    legendCache.left = null;
    legendCache.right = null;
    return;
  }

  const leftScheme = dualViewActive ? scheme2 : scheme1;
  const leftType = dualViewActive ? config.metric.type2 : config.metric.type1;
  legendCache.left = { title: getMetricDisplayName(leftType) + (leftScheme.legendSuffix || ''),
                       entries: buildLegendEntries(leftScheme) };
  renderLegendPanel(elements.legendPanelLeft, legendCache.left.title, legendCache.left.entries);
  elements.legendPanelLeft.classList.remove('hidden');

  if (dualViewActive) {
    legendCache.right = { title: getMetricDisplayName(config.metric.type3) + (scheme3.legendSuffix || ''),
                          entries: buildLegendEntries(scheme3) };
    renderLegendPanel(elements.legendPanelRight, legendCache.right.title, legendCache.right.entries);
    elements.legendPanelRight.classList.remove('hidden');
  } else {
    elements.legendPanelRight.classList.add('hidden');
    legendCache.right = null;
  }
}

// Binds a scheme to the loaded trace: resolves which named columns its colorMode reads, and
// builds the small per-scheme lookups that depend on the trace's region table. Replaces the
// nine build*Colors() passes, which walked the whole trace once per scheme before the first
// frame could be drawn — with derived values living on the TraceSource, all three Dual View
// schemes now share one set of columns and this returns in microseconds.
//
// Returns the column names the scheme needs; the caller unions them across active schemes and
// asks the source for exactly those.
function prepareScheme(scheme, src) {
  scheme.runIndex = null;
  scheme.runSample = null;
  scheme.distinguishBucketOfRegion = null;
  scheme.papiUnavailable = null;
  if (scheme.colorMode !== 'papi') scheme.legendSuffix = '';
  scheme.heatmapWindowTicks = scheme.heatmapWindowUs * (src.ticksPerSecond / 1e6);

  if (scheme.colorMode === 'iterations') {
    // REGION as well as ITER: past a rank's last interval the iteration index has to be bumped
    // once more if that last call was itself a Wait/Waitall.
    return [COL.ITER, COL.REGION];
  }
  if (scheme.colorMode === 'heatmap') {
    const [enterCol, leaveCol] = heatmapColumns(scheme.heatmapAppliesTo);
    scheme.enterCol = enterCol;
    scheme.leaveCol = leaveCol;
    scheme.gapIndexed = isGapHeatmap(scheme.heatmapAppliesTo);
    return enterCol === leaveCol ? [enterCol] : [enterCol, leaveCol];
  }
  if (scheme.colorMode === 'speedHeatmap') {
    return [COL.COMP_PREFIX];
  }
  if (scheme.colorMode === 'speedHeatmapRun') {
    scheme.runTargetRegionId = src.regionId(scheme.runTargetFn);
    // -1 when the trace never calls that function: runIndex() then yields empty per-rank lists
    // and every rank falls through to the flat colour, as it did before.
    scheme.runIndex = src.runIndex(scheme.runTargetRegionId);
    scheme.runSample = makeSample(src.nRanks);
    return [COL.PCT_DUR_REGION];
  }
  if (scheme.colorMode === 'distinguish') {
    scheme.distinguishBucketOfRegion = src.regionNames.map(
      name => scheme.distinguishBucketColors[classifyMpiFunction(name)]);
    return [COL.REGION];
  }
  if (scheme.colorMode === 'papi') {
    return preparePapiScheme(scheme, src);
  }
  return [];
}

/**
 * Resolves a PAPI formula's counters to the columns of this trace, for both scopes.
 *
 * A metric reads whichever segment a rank is in — its MPI call, or the Comp gap it sits in —
 * so both the `call` and `gap` columns of every counter are needed. They are adjacent in the
 * block layout, so they coalesce into one byte range.
 *
 * Returns the column names to fetch, and sets `scheme.papiUnavailable` when the trace cannot
 * answer at all. Returning an empty list without that set would be the silent-failure trap:
 * no columns fetched, every rank identical, nothing said.
 */
function preparePapiScheme(scheme, src) {
  const spec = PAPI_FORMULAS[scheme.papiFormula] || PAPI_FORMULAS.ipc;
  scheme.papiVals = new Float64Array(src.nRanks);
  scheme.papiPct = new Float64Array(src.nRanks);
  scheme.papiValid = new Uint8Array(src.nRanks);
  scheme.papiOrder = new Int32Array(src.nRanks);
  scheme.papiCompact = new Float64Array(src.nRanks);
  scheme.papiCallCols = scheme.papiGapCols = null;
  scheme.papiUnavailable = null;

  if (!src.papiCounters || src.papiCounters.length === 0) {
    scheme.papiUnavailable = src.format === 'pv3dbin'
      ? 'This .pv3dbin stores no hardware counters — it was converted with '
        + '--ignore-papi-presets. Re-convert it without that flag to use the PAPI metric.'
      : 'This is a legacy .txt trace (function;starttime;endtime); it records no hardware '
        + 'counters at all. The PAPI metric needs a .pv3dbin converted from an OTF2 trace.';
    return [];
  }

  const missing = [...spec.num, ...(spec.den || [])]
    .filter(c => !src.papiCounters.includes(c));
  if (missing.length) {
    scheme.papiUnavailable = `${spec.label} needs ${missing.join(' and ')}, which this trace `
      + `does not record. It has: ${src.papiCounters.join(', ')}.`;
    return [];
  }

  // A counter present but never moving is stored as no column at all; its increment is known
  // to be exactly zero, which is an answer — unlike an absent counter, which is not.
  const resolve = (names, scope) => names.map(
    c => (src.papiIsConstant(c, scope) ? null : src.papiColumnFor(c, scope)));
  const pack = (scope) => ({
    num: resolve(spec.num, scope),
    den: spec.den ? resolve(spec.den, scope) : null,
  });
  scheme.papiCallCols = pack('call');
  scheme.papiGapCols = pack('gap');

  // A denominator that is identically zero makes every rank's ratio undefined — that is a
  // refusal, where a zero numerator is merely a legitimate zero.
  if (spec.den && scheme.papiCallCols.den.every(c => c === null)
               && scheme.papiGapCols.den.every(c => c === null)) {
    scheme.papiUnavailable = `${spec.label} divides by ${spec.den.join(' + ')}, which never `
      + 'advances anywhere in this trace, so the ratio is undefined for every rank.';
    return [];
  }

  const names = new Set();
  for (const pk of [scheme.papiCallCols, scheme.papiGapCols]) {
    for (const c of pk.num) if (c) names.add(c);
    if (pk.den) for (const c of pk.den) if (c) names.add(c);
  }
  return [...names];
}

/** Sum of one side of a formula at (chunk, row); NaN when any part has no answer. */
function papiSum(src, cols, chunk, row) {
  let total = 0;
  for (let k = 0; k < cols.length; k++) {
    if (cols[k] === null) continue;          // constant counter: contributes exactly 0
    const d = src.papiDelta(cols[k], chunk, row);
    if (!Number.isFinite(d)) return NaN;
    total += d;
  }
  return total;
}

/** Resolves a scheme's `runIndex` when the source builds it asynchronously (.pv3dbin reads
 *  one column of the whole level to find a region's occurrences). */
async function resolveSchemeAsync(scheme) {
  if (scheme.runIndex && typeof scheme.runIndex.then === 'function') {
    scheme.runIndex = await scheme.runIndex;
  }
}

// Cumulative extra offset for a rank at `index` along one physical axis:
// adds a gap every time `index` crosses a CC-Numa/socket/node boundary, so a
// rank that's simultaneously at a socket AND node boundary picks up both gaps.
function boundaryOffset(index, periods, gaps) {
  let offset = 0;
  for (let level = 0; level < periods.length; level++) {
    if (periods[level] > 0) offset += Math.floor(index / periods[level]) * gaps[level];
  }
  return offset;
}

function generateSimpleGrid(totalRanks, dimensions, axisMapping, spacing = 0.5) {
  const coordinates = [];
  let gridX, gridY, gridZ;

  // "Width" is always physical/screen X, "Height" is Y, "Depth" is Z
  // (see the fixed physMap in the visualize click handler) — so the
  // Node/Socket/CC-Numa dimension fields can be applied directly to
  // physicalIndices.x/y/z below without re-deriving the axis mapping.
  const mt = config.geometry?.machineTopology;
  const hybridActive = config.geometry?.type === 'square'
    ? config.geometry.square?.hybridView
    : config.geometry?.type === 'cuboid'
      ? config.geometry.cuboid?.hybridView
      : false;
  const spacingActive = hybridActive && mt?.displayMethod === 'spacing';

  const periodsAndGaps = (axisLabel) => {
    if (!spacingActive) return { periods: [], gaps: [] };
    return {
      periods: [
        mt.ccNumaDomainDimensions?.[axisLabel],
        mt.socketDimensions?.[axisLabel],
        mt.nodeDimensions?.[axisLabel],
      ],
      gaps: [
        mt.extraSpacing?.ccNuma || 0,
        mt.extraSpacing?.socket || 0,
        mt.extraSpacing?.node || 0,
      ],
    };
  };
  const widthPG = periodsAndGaps('width');
  const heightPG = periodsAndGaps('height');
  const depthPG = periodsAndGaps('depth');

  const autoLayout = () => {
    // For 3D, create a cube-like structure
    const cubeRoot = Math.ceil(Math.pow(totalRanks, 1 / 3));
    gridX = cubeRoot;
    gridY = cubeRoot;
    gridZ = Math.ceil(totalRanks / (gridX * gridY));
  };

  if (dimensions && (dimensions[0] || dimensions[1] || dimensions[2])) {
    const userX = dimensions[0];
    const userY = dimensions[1];
    const userZ = dimensions[2];

    if (userX && userY && userZ) {
      if (userX <= 0 || userY <= 0 || userZ <= 0) {
        alert('Grid dimensions must be positive numbers. Falling back to automatic layout.');
        autoLayout();
      } else if (userX * userY * userZ < totalRanks) {
        alert(`Warning: Grid dimensions [${userX}, ${userY}, ${userZ}] are too small for ${totalRanks} ranks. Falling back to automatic layout.`);
        autoLayout();
      } else {
        gridX = userX;
        gridY = userY;
        gridZ = userZ;
        if (gridX * gridY * gridZ > totalRanks) {
          console.warn(`Notice: Grid dimensions [${gridX}, ${gridY}, ${gridZ}] are larger than the required ${totalRanks} ranks. The grid will have empty spaces.`);
        }
      }
    } else if (userX && userY) {
      if (userX <= 0 || userY <= 0) {
        alert('Grid dimensions must be positive numbers. Falling back to automatic layout.');
        autoLayout();
      } else {
        gridX = userX;
        gridY = userY;
        gridZ = Math.ceil(totalRanks / (gridX * gridY));
      }
    } else if (userX) {
      if (userX <= 0) {
        alert('Grid dimension must be a positive number. Falling back to automatic layout.');
        autoLayout();
      } else {
        gridX = userX;
        const remaining = Math.ceil(totalRanks / gridX);
        gridY = Math.ceil(Math.sqrt(remaining));
        gridZ = Math.ceil(remaining / gridY);
      }
    } else {
      autoLayout();
    }
  } else {
    autoLayout();
  }

  // Calculate grid extents for each logical dimension
  const gridSizes = { x: gridX, y: gridY, z: gridZ };

  // Apply axis mapping to determine physical dimensions
  const physicalSizes = {
    x: gridSizes[axisMapping.x],
    y: gridSizes[axisMapping.y],
    z: gridSizes[axisMapping.z]
  };

  // Extra spacing accumulated across the whole axis extent, so the grid
  // (including the added gaps) stays centered at the origin.
  const totalWidth = (physicalSizes.x - 1) * spacing + boundaryOffset(physicalSizes.x - 1, widthPG.periods, widthPG.gaps);
  const totalHeight = (physicalSizes.y - 1) * spacing + boundaryOffset(physicalSizes.y - 1, heightPG.periods, heightPG.gaps);
  const totalDepth = (physicalSizes.z - 1) * spacing + boundaryOffset(physicalSizes.z - 1, depthPG.periods, depthPG.gaps);
  const startX = -totalWidth / 2;
  const startY = -totalHeight / 2;
  const startZ = -totalDepth / 2;

  for (let i = 0; i < totalRanks; i++) {
    // Calculate logical grid position
    const ix = i % gridX;
    const iy = Math.floor(i / gridX) % gridY;
    const iz = Math.floor(i / (gridX * gridY));

    // Logical positions in grid space
    const logicalPos = { x: ix, y: iy, z: iz };

    // Map logical positions to physical positions based on axis mapping
    const physicalIndices = {
      x: logicalPos[axisMapping.x],
      y: logicalPos[axisMapping.y],
      z: logicalPos[axisMapping.z]
    };

    // Depth is reversed relative to the other axes: the default camera looks
    // from +Z toward the origin, so without this, higher rank indices along
    // depth would end up closer to the camera (in front) instead of farther
    // away (behind) — flipping the index here makes rank order increase
    // going *into* the screen, so later ranks stack behind earlier ones.
    const depthIndex = physicalSizes.z - 1 - physicalIndices.z;

    // Calculate actual 3D coordinates, plus any CC-Numa/socket/node boundary spacing
    const mappedPos = new THREE.Vector3(
      startX + physicalIndices.x * spacing + boundaryOffset(physicalIndices.x, widthPG.periods, widthPG.gaps),
      startY + physicalIndices.y * spacing + boundaryOffset(physicalIndices.y, heightPG.periods, heightPG.gaps),
      startZ + depthIndex * spacing + boundaryOffset(depthIndex, depthPG.periods, depthPG.gaps)
    );

    coordinates.push(mappedPos);
  }

  return coordinates;
}

// Pure per-scheme color computation: writes finalColor/finalAlpha for every rank/face into
// the given output typed arrays (colorOut: 3 floats/instance for InstancedMesh, 9 floats/face
// for sphere geometry; alphaOut: 1 float/instance, or 3/face). Doesn't touch any live GPU-facing
// attribute directly — callers swap the desired scheme's buffer in via
// syncSchemeBufferToLiveAttribute() before rendering.
//
// Reads the shared cursor result `sample`, which updateInstanceColors() refills once per frame
// before calling this for each active scheme: which interval brackets `currentTime` is a
// property of the trace, not of a colour scheme, so it is resolved once rather than per scheme.
// Colours come from percentile columns through percentileToBucketColor() here, where the old
// version read per-event colour tables built over the whole trace at load time.
function computeSchemeColors(scheme, currentTime, isSphere, faceLimit, colorOut, alphaOut) {
  const src = traceSource;
  const smp = sample;
  const n = src.nRanks;

  // "Heatmap instantaneous speed" ranks ranks against each other live, every frame — unlike
  // the other heatmap modes (percentile precomputed once per call), this cross-rank pass
  // has to run here since it depends on the current playback position.
  if (scheme.colorMode === 'speedHeatmap') {
    const windowStart = Math.max(0, currentTime - scheme.heatmapWindowTicks);
    for (let r = 0; r < n; r++) {
      speedVals[r] = src.compTimeAt(r, currentTime) - src.compTimeAt(r, windowStart);
    }
    rankPercentiles(speedVals, n, speedPct);
  }

  // "…speed (run)" brackets a subsequence — only one region's occurrences — so it needs its
  // own cursor result; the shared one brackets every interval regardless of region.
  if (scheme.colorMode === 'speedHeatmapRun' && scheme.runSample) {
    src.sampleRun(scheme.runIndex, currentTime, scheme.runSample);
  }

  // The PAPI metric evaluates its formula over whichever segment each rank is currently in —
  // the MPI call it is inside, or the Comp gap between two calls — and ranks the results
  // against each other. Hardware rates are directly comparable across ranks running different
  // code, which is why an instantaneous ranking is the right question here, unlike durations.
  if (scheme.colorMode === 'papi' && scheme.papiCallCols) {
    const vals = scheme.papiVals, valid = scheme.papiValid;
    const spec = PAPI_FORMULAS[scheme.papiFormula] || PAPI_FORMULAS.ipc;
    const scale = spec.scale || 1;
    for (let i = 0; i < n; i++) {
      valid[i] = 0;
      let chunk, row, cols, dt;
      if (smp.inside[i] === 1) {
        chunk = smp.curChunk[i]; row = smp.curRow[i]; cols = scheme.papiCallCols;
        dt = smp.curEnd[i] - smp.curStart[i];
      } else if (smp.prevRow[i] >= 0 && smp.curRow[i] >= 0) {
        // In the gap after `prev`: the increment is stored on the interval it follows.
        chunk = smp.prevChunk[i]; row = smp.prevRow[i]; cols = scheme.papiGapCols;
        dt = smp.curStart[i] - smp.prevEnd[i];
      } else {
        continue;              // before a rank's first call or after its last
      }
      const num = papiSum(src, cols.num, chunk, row);
      if (!Number.isFinite(num)) continue;
      let v;
      if (cols.den) {
        const den = papiSum(src, cols.den, chunk, row);
        if (!Number.isFinite(den) || den <= 0) continue;   // undefined, not zero
        v = num / den;
      } else {
        if (!(dt > 0)) continue;
        v = num * src.ticksPerSecond / dt;
      }
      if (!Number.isFinite(v)) continue;
      vals[i] = v * scale;
      valid[i] = 1;
    }

    // Rank only the ranks that have a value. rankPercentiles sorts with a comparator that
    // returns NaN for an invalid entry, which Array.prototype.sort treats as "equal" — one
    // NaN would scramble every other rank's percentile, not just its own.
    let m = 0;
    for (let i = 0; i < n; i++) {
      if (valid[i]) { scheme.papiOrder[m] = i; scheme.papiCompact[m] = vals[i]; m++; }
    }
    rankPercentiles(scheme.papiCompact, m, scheme.papiCompact);
    for (let k = 0; k < m; k++) scheme.papiPct[scheme.papiOrder[k]] = scheme.papiCompact[k];
  }

  for (let i = 0; i < n && i < faceLimit; i++) {
    const curChunk = smp.curChunk[i], curRow = smp.curRow[i];
    const prevChunk = smp.prevChunk[i], prevRow = smp.prevRow[i];
    const hasCur = curRow >= 0, hasPrev = prevRow >= 0;
    const inside = smp.inside[i] === 1;
    let finalColor;
    let finalAlpha;

    if (scheme.colorMode === 'iterations' && scheme.iterationColors.length > 0) {
      let iterIdx;
      if (hasCur) {
        iterIdx = src.cell(COL.ITER, curChunk, curRow);
      } else if (hasPrev) {
        // Past this rank's last interval: hold its iteration index, bumped once more when that
        // last call was itself the Wait/Waitall that would have ended the iteration.
        iterIdx = src.cell(COL.ITER, prevChunk, prevRow);
        const lastFn = src.regionNames[src.cell(COL.REGION, prevChunk, prevRow)];
        if (lastFn === 'MPI_Wait' || lastFn === 'MPI_Waitall') iterIdx++;
      } else {
        iterIdx = 0;   // this rank has no intervals at all
      }
      const iterSlot = iterIdx % scheme.iterationColors.length;
      finalColor = scheme.iterationColors[iterSlot];
      finalAlpha = scheme.iterationAlphas[iterSlot] ?? 1.0;
    } else if (scheme.colorMode === 'heatmap' && !scheme.gapIndexed) {
      if (inside) {
        // Inside a call: lerp from this call's own Enter-percentile color to its Leave-percentile
        // color ('mpiDuration' has enterCol === leaveCol, so this renders as one flat color)
        const eventDuration = smp.curEnd[i] - smp.curStart[i];
        const progress = eventDuration > 0 ? (currentTime - smp.curStart[i]) / eventDuration : 1.0;
        const e = percentileToBucketColor(scheme, src.cell(scheme.enterCol, curChunk, curRow), bucketA);
        const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, curChunk, curRow), bucketB);
        finalColor = tempColor.copy(e.color).lerp(l.color, progress);
        finalAlpha = e.alpha + (l.alpha - e.alpha) * progress;
      } else {
        // Gap between calls: behavior depends on the "in-between MPI calls" mode
        if (scheme.heatmapBetweenMode === 'keep' && hasPrev) {
          const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          finalColor = l.color;
          finalAlpha = l.alpha;
        } else if (scheme.heatmapBetweenMode === 'gradient' && hasPrev && hasCur) {
          const gapDuration = smp.curStart[i] - smp.prevEnd[i];
          const gapProgress = gapDuration > 0 ? (currentTime - smp.prevEnd[i]) / gapDuration : 1.0;
          const s = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          const e = percentileToBucketColor(scheme, src.cell(scheme.enterCol, curChunk, curRow), bucketB);
          finalColor = tempColor.copy(s.color).lerp(e.color, gapProgress);
          finalAlpha = s.alpha + (e.alpha - s.alpha) * gapProgress;
        } else if (scheme.heatmapBetweenMode === 'gradient' && hasPrev && !hasCur) {
          // Tail of the trace: no next call to gradient toward — hold the last leave color
          const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          finalColor = l.color;
          finalAlpha = l.alpha;
        } else {
          // 'custom' mode, or no prev/next event to reference (edge of the trace)
          finalColor = scheme.heatmapFlatColor;
          finalAlpha = scheme.heatmapFlatAlpha;
        }
      }
    } else if (scheme.colorMode === 'heatmap' && scheme.gapIndexed) {
      // Comp-family heatmaps: the *gaps* are ranked, and the MPI calls are "the in-between".
      // A gap column is stored on the interval it follows, so the gap the playhead sits in is
      // indexed by prevRow, and the gap after the current call by curRow.
      if (!inside) {
        const hasGap = hasPrev && hasCur;
        if (hasGap) {
          const gapStart = smp.prevEnd[i];
          const gapEnd = smp.curStart[i];
          const gapDuration = gapEnd - gapStart;
          const progress = gapDuration > 0 ? (currentTime - gapStart) / gapDuration : 1.0;
          const e = percentileToBucketColor(scheme, src.cell(scheme.enterCol, prevChunk, prevRow), bucketA);
          const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketB);
          finalColor = tempColor.copy(e.color).lerp(l.color, progress);
          finalAlpha = e.alpha + (l.alpha - e.alpha) * progress;
        } else {
          // Leading/trailing edge of the trace (before the first event or after the
          // last one) isn't a call-bounded Comp period — flat fallback.
          finalColor = scheme.heatmapFlatColor;
          finalAlpha = scheme.heatmapFlatAlpha;
        }
      } else {
        // Inside an MPI call: reference the Comp periods immediately before and after it.
        const hasNextGap = hasCur && smp.curIsLast[i] === 0;
        if (scheme.heatmapBetweenMode === 'keep' && hasPrev) {
          const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          finalColor = l.color;
          finalAlpha = l.alpha;
        } else if (scheme.heatmapBetweenMode === 'gradient' && hasPrev && hasNextGap) {
          const eventDuration = smp.curEnd[i] - smp.curStart[i];
          const progress = eventDuration > 0 ? (currentTime - smp.curStart[i]) / eventDuration : 1.0;
          const s = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          const e = percentileToBucketColor(scheme, src.cell(scheme.enterCol, curChunk, curRow), bucketB);
          finalColor = tempColor.copy(s.color).lerp(e.color, progress);
          finalAlpha = s.alpha + (e.alpha - s.alpha) * progress;
        } else if (scheme.heatmapBetweenMode === 'gradient' && hasPrev && !hasNextGap) {
          const l = percentileToBucketColor(scheme, src.cell(scheme.leaveCol, prevChunk, prevRow), bucketA);
          finalColor = l.color;
          finalAlpha = l.alpha;
        } else {
          // 'custom' mode, or no prev/next Comp period to reference (edge of the trace)
          finalColor = scheme.heatmapFlatColor;
          finalAlpha = scheme.heatmapFlatAlpha;
        }
      }
    } else if (scheme.colorMode === 'speedHeatmap') {
      const bucket = percentileToBucketColor(scheme, speedPct[i], bucketA);
      finalColor = bucket.color;
      finalAlpha = bucket.alpha;
    } else if (scheme.colorMode === 'papi') {
      // One flat colour for the whole segment: a segment has a single counter increment,
      // so there is nothing to interpolate between the way an Enter/Leave pair allows.
      // papiValid is null only if a scheme reached render without prepareScheme succeeding,
      // which the pre-flight refusal should prevent; fall back to "no value" rather than throw.
      if (scheme.papiValid && scheme.papiValid[i]) {
        const bucket = percentileToBucketColor(scheme, scheme.papiPct[i], bucketA);
        finalColor = bucket.color;
        finalAlpha = bucket.alpha;
      } else {
        finalColor = scheme.papiNoValueColor;
        finalAlpha = scheme.papiNoValueAlpha;
      }
    } else if (scheme.colorMode === 'speedHeatmapRun') {
      const rs = scheme.runSample;
      const rCur = rs.curRow[i], rPrev = rs.prevRow[i];
      const rHasCur = rCur >= 0, rHasPrev = rPrev >= 0;

      if (rs.inside[i] === 1) {
        // Constant color for the whole call — no Enter/Leave lerp, just this call's own bucket.
        const b = percentileToBucketColor(scheme, src.cell(COL.PCT_DUR_REGION, rs.curChunk[i], rCur), bucketA);
        finalColor = b.color;
        finalAlpha = b.alpha;
      } else {
        // Gap between two occurrences of runTargetFn — may span other (non-target) MPI calls
        // and Comp periods alike; behavior depends on the "in-between MPI calls" mode.
        if (scheme.heatmapBetweenMode === 'keep' && rHasPrev) {
          const b = percentileToBucketColor(scheme, src.cell(COL.PCT_DUR_REGION, rs.prevChunk[i], rPrev), bucketA);
          finalColor = b.color;
          finalAlpha = b.alpha;
        } else if (scheme.heatmapBetweenMode === 'gradient' && rHasPrev && rHasCur) {
          const gapDuration = rs.curStart[i] - rs.prevEnd[i];
          const gapProgress = gapDuration > 0 ? (currentTime - rs.prevEnd[i]) / gapDuration : 1.0;
          const s = percentileToBucketColor(scheme, src.cell(COL.PCT_DUR_REGION, rs.prevChunk[i], rPrev), bucketA);
          const e = percentileToBucketColor(scheme, src.cell(COL.PCT_DUR_REGION, rs.curChunk[i], rCur), bucketB);
          finalColor = tempColor.copy(s.color).lerp(e.color, gapProgress);
          finalAlpha = s.alpha + (e.alpha - s.alpha) * gapProgress;
        } else if (scheme.heatmapBetweenMode === 'gradient' && rHasPrev && !rHasCur) {
          // Tail of the trace: no next occurrence to gradient toward — hold the last color.
          const b = percentileToBucketColor(scheme, src.cell(COL.PCT_DUR_REGION, rs.prevChunk[i], rPrev), bucketA);
          finalColor = b.color;
          finalAlpha = b.alpha;
        } else {
          // 'custom' mode, or no prev/next occurrence to reference (edge of the trace, or the
          // rank never calls runTargetFn at all)
          finalColor = scheme.heatmapFlatColor;
          finalAlpha = scheme.heatmapFlatAlpha;
        }
      }
    } else if (scheme.colorMode === 'distinguish') {
      const byRegion = scheme.distinguishBucketOfRegion;
      if (inside) {
        // Inside a call: lerp from this call's bucket "enter" color to its "leave" color
        // (flat mode sets enter===leave per call, so this renders as one flat color)
        const b = byRegion[src.cell(COL.REGION, curChunk, curRow)];
        const eventDuration = smp.curEnd[i] - smp.curStart[i];
        const progress = eventDuration > 0 ? (currentTime - smp.curStart[i]) / eventDuration : 1.0;
        finalColor = tempColor.copy(b.enter.color).lerp(b.leave.color, progress);
        finalAlpha = b.enter.alpha + (b.leave.alpha - b.enter.alpha) * progress;
      } else {
        // Gap between calls: behavior depends on the "in-between MPI calls" mode
        const bp = hasPrev ? byRegion[src.cell(COL.REGION, prevChunk, prevRow)] : null;
        const bc = hasCur ? byRegion[src.cell(COL.REGION, curChunk, curRow)] : null;
        if (scheme.distinguishBetweenMode === 'keep' && hasPrev) {
          finalColor = bp.leave.color;
          finalAlpha = bp.leave.alpha;
        } else if (scheme.distinguishBetweenMode === 'gradient' && hasPrev && hasCur) {
          const gapDuration = smp.curStart[i] - smp.prevEnd[i];
          const gapProgress = gapDuration > 0 ? (currentTime - smp.prevEnd[i]) / gapDuration : 1.0;
          finalColor = tempColor.copy(bp.leave.color).lerp(bc.enter.color, gapProgress);
          finalAlpha = bp.leave.alpha + (bc.enter.alpha - bp.leave.alpha) * gapProgress;
        } else if (scheme.distinguishBetweenMode === 'gradient' && hasPrev && !hasCur) {
          finalColor = bp.leave.color;
          finalAlpha = bp.leave.alpha;
        } else if (scheme.distinguishBetweenMode === 'custom' && scheme.distinguishUseGradient) {
          // Gradient mode's 'custom': the gap itself lerps Comp begin -> Comp end over its
          // own duration, independent of the surrounding calls' colors.
          const gapStart = hasPrev ? smp.prevEnd[i] : 0;
          const gapEnd = hasCur ? smp.curStart[i] : gapStart;
          const gapDuration = gapEnd - gapStart;
          const gapProgress = gapDuration > 0 ? (currentTime - gapStart) / gapDuration : 1.0;
          finalColor = tempColor.copy(scheme.distinguishCompBeginColor).lerp(scheme.distinguishCompEndColor, gapProgress);
          finalAlpha = scheme.distinguishCompBeginAlpha + (scheme.distinguishCompEndAlpha - scheme.distinguishCompBeginAlpha) * gapProgress;
        } else {
          // Flat mode's 'custom', or no prev/next event to reference (edge of the trace)
          finalColor = scheme.distinguishFlatColor;
          finalAlpha = scheme.distinguishFlatAlpha;
        }
      }
    } else if (inside) {
      const eventDuration = smp.curEnd[i] - smp.curStart[i];
      const progress = eventDuration > 0 ? (currentTime - smp.curStart[i]) / eventDuration : 1.0;
      finalColor = tempColor.copy(scheme.mpiStartColor).lerp(scheme.mpiEndColor, progress);
      finalAlpha = scheme.mpiStartAlpha + (scheme.mpiEndAlpha - scheme.mpiStartAlpha) * progress;
    } else {
      finalColor = scheme.compColor;
      finalAlpha = scheme.compColorAlpha;
    }
    if (isSphere) {
      for (let v = 0; v < 3; v++) {
        colorOut[i*9+v*3+0] = finalColor.r;
        colorOut[i*9+v*3+1] = finalColor.g;
        colorOut[i*9+v*3+2] = finalColor.b;
        alphaOut[i*3+v] = finalAlpha;
      }
    } else {
      colorOut[i*3+0] = finalColor.r;
      colorOut[i*3+1] = finalColor.g;
      colorOut[i*3+2] = finalColor.b;
      alphaOut[i] = finalAlpha;
    }
  }
}

// Copies a scheme's computed color/alpha buffers into the live GPU-facing attributes (the
// InstancedMesh's instanceColor/instanceAlpha, or the sphere geometry's color/faceAlpha) and
// flags them for re-upload. Called once per half when Dual View splits the render.
function syncSchemeBufferToLiveAttribute(colorBuf, alphaBuf, isSphere) {
  if (isSphere) {
    const colorAttr = sphereMesh.geometry.attributes.color;
    const alphaAttrSphere = sphereMesh.geometry.attributes.faceAlpha;
    colorAttr.array.set(colorBuf);
    colorAttr.needsUpdate = true;
    alphaAttrSphere.array.set(alphaBuf);
    alphaAttrSphere.needsUpdate = true;
  } else {
    instancedMesh.instanceColor.array.set(colorBuf);
    instancedMesh.instanceColor.needsUpdate = true;
    const alphaAttr = instancedMesh.geometry.attributes.instanceAlpha;
    alphaAttr.array.set(alphaBuf);
    alphaAttr.needsUpdate = true;
  }
}

// Recomputes color buffers for scheme1 always, and scheme2/scheme3 when Dual View is active,
// then syncs scheme1's result to the live attribute (the default single-view target — the
// dual-view split itself is handled by renderFrame(), which re-syncs per half at render time).
/**
 * Makes sure the data for the current playhead is in memory, and repaints once it arrives.
 *
 * With .pv3dbin a block is fetched as the playhead reaches it, so both playback and a seek can
 * land somewhere not yet loaded. Drawing anyway would show every rank as idle — not a blurry
 * picture but a false one — so playback is held and the frame repainted when the block lands.
 * Hooked into updateInstanceColors() because every path that moves the playhead goes through
 * it: playback, the slider, the frame buttons and a timejump alike.
 */
function ensureDataForCurrentTime() {
  if (!traceSource || !traceSource.ready || waitingForData) return true;
  const t = timeProxy.time;
  if (traceSource.ready(t)) return true;

  waitingForData = true;
  if (masterTimeline && !masterTimeline.paused()) {
    resumeAfterLoad = true;
    masterTimeline.pause();
  }
  setLoadingIndicator(true);
  Promise.resolve(traceSource.ensure(t)).then(() => {
    waitingForData = false;
    setLoadingIndicator(false);
    updateInstanceColors();
    if (resumeAfterLoad && masterTimeline) masterTimeline.play();
  }).catch(err => {
    waitingForData = false;
    setLoadingIndicator(false);
    console.error('pv3d: block fetch failed', err);
  });
  return false;
}

function updateInstanceColors() {
  if (!traceSource) return;
  // Repaint anyway with what is cached — a half-right frame for the few milliseconds a fetch
  // takes beats a blank one — but the repaint scheduled above is the one that counts.
  ensureDataForCurrentTime();
  const currentTime = timeProxy.time;
  const isSphere = sphereMesh !== null;
  const faceLimit = isSphere
    ? Math.floor(sphereMesh.geometry.attributes.color.array.length / 9)
    : traceSource.nRanks;

  // Resolve the cursor once for the frame — which interval each rank is in is a property of
  // the trace, so the schemes below all read the same answer instead of each maintaining a
  // bracket of its own.
  traceSource.sampleAll(currentTime, sample);

  computeSchemeColors(scheme1, currentTime, isSphere, faceLimit, colorBuffer1, alphaBuffer1);
  if (dualViewActive) {
    computeSchemeColors(scheme2, currentTime, isSphere, faceLimit, colorBuffer2, alphaBuffer2);
    computeSchemeColors(scheme3, currentTime, isSphere, faceLimit, colorBuffer3, alphaBuffer3);
  }
  syncSchemeBufferToLiveAttribute(colorBuffer1, alphaBuffer1, isSphere);
}

// The renderer works in integer ticks; seconds exist only at the display boundary.
// A tick is 1 µs on the legacy text path and one timer cycle for .pv3dbin, so the two UI
// fields that are typed in microseconds convert through here rather than assuming either.
function microsecondsToTicks(us) {
  return traceSource ? us * (traceSource.ticksPerSecond / 1e6) : us;
}

function ticksToSeconds(ticks) {
  return traceSource ? ticks / traceSource.ticksPerSecond : ticks;
}

function formatTicks(ticks) {
  return formatTime(ticksToSeconds(ticks));
}

function formatTime(absoluteTimeInSeconds) {
  const seconds = Math.floor(absoluteTimeInSeconds);
  const milliseconds = Math.floor((absoluteTimeInSeconds * 1000) % 1000);
  const microseconds = Math.floor((absoluteTimeInSeconds * 1000000) % 1000);
  return `${seconds}s ${String(milliseconds).padStart(3, '0')}ms ${String(microseconds).padStart(3, '0')}µs`;
}

function updateClockDisplay(initialOffset = 0) {
  if (!masterTimeline) return;

  const elapsedTicks = timeProxy.time;

  elements.clockEl.textContent = `Time: ${formatTicks(initialOffset + elapsedTicks)}`;
  // Timeline display next to the slider is relative to trace start (0 to maxTime - minTime),
  // unlike the absolute clock above which includes initialOffset (the trace's min time).
  //
  // Measured against the play span, so the pair stays "elapsed / total" for whatever the
  // slider covers. Against the trace it would read 523ms / 392ms at the start of a selection
  // — an absolute position over a duration, with the numerator larger than the denominator.
  if (elements.timelineCurrentTime) {
    elements.timelineCurrentTime.textContent = formatTicks(elapsedTicks - playStart);
  }
}

function drawPill(ctx, x, y, text, font) {
  ctx.font = font;
  const paddingX = 8, paddingY = 4;
  const metrics = ctx.measureText(text);
  const textHeight = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent || 13;
  const w = metrics.width + paddingX * 2;
  const h = textHeight + paddingY * 2;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();

  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + paddingX, y + h / 2);
}

// Mirrors #info-panel's on-screen position (top:10px; left:10px) so the
// recorded video matches what's visible while recording.
function drawClockOverlay(ctx) {
  drawPill(ctx, 10, 10, elements.clockEl.textContent, '13px monospace');
}

// Mirrors #rank-hover-tooltip's inline styling/position.
function drawTooltipOverlay(ctx) {
  if (!hoverTooltip.visible) return;
  drawPill(ctx, hoverTooltip.x, hoverTooltip.y, hoverTooltip.text, '13px monospace');
}

// Mirrors #legend-panel-left/#legend-panel-right's on-screen position/content. Reads from
// legendCache (built once per Visualize click by refreshLegendPanels()) rather than
// recomputing, so the burn-in always matches whatever is currently on screen.
function drawLegendOverlays(ctx) {
  if (!config.metric?.showLegend) return;
  if (legendCache.left) drawLegendPanelOverlay(ctx, legendCache.left, 10, 60, 'left');
  if (legendCache.right) drawLegendPanelOverlay(ctx, legendCache.right, 10, 60, 'right');
}

function drawLegendPanelOverlay(ctx, { title, entries }, margin, yStart, align) {
  let y = drawLegendTextRow(ctx, title, margin, yStart, align, 'bold 13px monospace');
  for (const entry of entries) {
    y = drawLegendSwatchRow(ctx, entry, margin, y, align);
  }
}

// A title pill, reusing drawPill's exact styling but computed for right-alignment (mirrors
// CSS `right: 10px`) when align === 'right'.
function drawLegendTextRow(ctx, text, margin, y, align, font) {
  ctx.font = font;
  const w = ctx.measureText(text).width + 16;
  const x = align === 'right' ? ctx.canvas.width - margin - w : margin;
  drawPill(ctx, x, y, text, font);
  return y + 24;
}

// A colored swatch + label row, matching drawPill's pill background with an inset color patch.
function drawLegendSwatchRow(ctx, entry, margin, y, align) {
  const font = '13px monospace';
  ctx.font = font;
  const swatchSize = 14, gap = 6, paddingX = 8, paddingY = 4;
  const textWidth = ctx.measureText(entry.label).width;
  const w = paddingX * 2 + swatchSize + gap + textWidth;
  const h = swatchSize + paddingY * 2;
  const x = align === 'right' ? ctx.canvas.width - margin - w : margin;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();

  const c = entry.color;
  ctx.fillStyle = `rgba(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(c.b * 255)}, ${entry.alpha})`;
  ctx.beginPath();
  ctx.roundRect(x + paddingX, y + paddingY, swatchSize, swatchSize, 3);
  ctx.fill();

  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(entry.label, x + paddingX + swatchSize + gap, y + h / 2);

  return y + h + 4;
}

function setupAnimation(maxTicks, initialOffset) {
  currentMaxTicks = maxTicks;
  resetSpans(maxTicks);
  timelineSelection = null;
  hudDrag = null;
  elements.clearSelectionBtn?.classList.add('hidden');
  // The Frame Step the user asked for is the fully-zoomed-out setting; zoom scales it from here.
  clockInitialOffset = initialOffset;
  baseFrameStepUs = parseFloat(config.controls?.timeGranularity) || 100;
  tweenStepTicks = microsecondsToTicks(baseFrameStepUs);
  // Relative to trace start (maxTicks is already the span, i.e. rawMaxTime - rawMinTime).
  if (elements.timelineMaxTime) elements.timelineMaxTime.textContent = formatTicks(maxTicks);

  masterTimeline = gsap.timeline({
    paused: true,
    // Looping a selection, expressed as "start again" rather than `repeat: -1`. The repeat
    // flag would change what progress() means at the boundary, and progress() is read in eight
    // places here — including the recording auto-stop, which wants exactly one pass.
    onComplete: () => {
      if (timelineSelection && !isRecording()) masterTimeline.play(0);
    },
    onUpdate: () => {
      updateInstanceColors();
      updateClockDisplay(initialOffset);
      // The HUD is driven by trace time, not progress: where the playhead sits on its x axis
      // depends on the zoom window, which progress knows nothing about.
      if (vampirReady) {
        updateVampirTimeline(timeProxy.time);
        vampirTexture.needsUpdate = true;
      }
    }
  });

  rebuildPlayTween(playStart, playEnd, playStart);
}

/**
 * Builds (or rebuilds) the single tween that animates `timeProxy.time` across the play span.
 *
 * Rebuilding is what makes a selection cheap: `progress` already means "position within the
 * animated range", so narrowing that range silently redefines it for the slider, the
 * frame-step buttons and the arrow keys without touching any of them.
 *
 * `preserveTicks` keeps the playhead where it was when the new range still contains it, so
 * selecting a region around the current position does not jump the view.
 */
function rebuildPlayTween(startTicks, endTicks, preserveTicks) {
  if (!masterTimeline) return;
  playStart = startTicks;
  playEnd = Math.max(startTicks + 1, endTicks);

  // Total playback duration = (number of discrete frames implied by the
  // granularity) / (frames per second) — e.g. a 6s trace at 100µs granularity
  // is 60,000 frames; at 60fps that's 1000s of real playback, at 30fps it's
  // 2000s (half the fps doubles the duration, matching "fps" literally).
  //
  // The step comes from the live field, not the Visualize-time snapshot in `config`: zoom
  // rewrites that field, so a snapshot would rebuild the tween at the resolution the trace was
  // loaded with and silently undo the zoom's effect on playback speed.
  const speed = parseFloat(config.controls?.playbackSpeed) || 60;
  tweenStepTicks = liveFrameStepTicks();
  const span = playEnd - playStart;
  const numFrames = tweenStepTicks > 0 ? span / tweenStepTicks : span;
  const durationSeconds = speed > 0 ? numFrames / speed : numFrames;

  const wasPaused = masterTimeline.paused();
  masterTimeline.clear();
  // fromTo, not to: a `to()` captures its start value when it first renders, which would make
  // this correct only because `progress()` below happens to be the next thing to touch the
  // proxy. Stating both ends makes `time = playStart + progress * span` an invariant of the
  // tween rather than an ordering accident — and that invariant is what lets timeScale move
  // playback speed without moving the playhead.
  //
  // immediateRender is left at its default (true) on purpose. With it off, creating the tween
  // does not write the "from" value, and `progress(0)` below is a no-op on a freshly-cleared
  // timeline already at time 0 — so nothing renders and `timeProxy.time` keeps whatever the
  // previous range left in it. Measured: selecting 523-915 ms after a seek to 392 ms left the
  // playhead reading 392 ms at progress 0.
  masterTimeline.fromTo(timeProxy,
    { time: playStart },
    { time: playEnd, duration: durationSeconds, ease: 'none' });

  // The duration just computed already reflects the current frame step, so any timeScale left
  // over from a previous zoom would apply it a second time.
  masterTimeline.timeScale(1);

  const keep = preserveTicks >= playStart && preserveTicks <= playEnd
    ? timeToProgress(preserveTicks) : 0;
  masterTimeline.progress(keep);
  if (wasPaused) masterTimeline.pause();
  elements.slider.value = String(keep);
  refreshTimelineUI();
}

function bestFactorPair(n) {
  for (let a = Math.floor(Math.sqrt(n)); a >= 1; a--) {
    if (n % a === 0) return [n / a, a];
  }
  return [n, 1];
}

// ─── Icosphere (exact JS translation of vedranaa/icosphere) ──────────────────

function jsIcosahedron() {
  const phi = (1 + Math.sqrt(5)) / 2;
  const s = Math.sqrt(1 + phi * phi);
  const raw = [[0,1,phi],[0,-1,phi],[1,phi,0],[-1,phi,0],[phi,0,1],[-phi,0,1]];
  const vertices = [
    ...raw.map(([x,y,z]) => [x/s, y/s, z/s]),
    ...raw.map(([x,y,z]) => [-x/s,-y/s,-z/s])
  ];
  const faces = [
    [0,5,1],[0,3,5],[0,2,3],[0,4,2],[0,1,4],
    [1,5,8],[5,3,10],[3,2,7],[2,4,11],[4,1,9],
    [7,11,6],[11,9,6],[9,8,6],[8,10,6],[10,7,6],
    [2,11,7],[4,9,11],[1,8,9],[5,10,8],[3,7,10]
  ];
  return { vertices, faces };
}

function jsFacesTemplate(nu) {
  const faces = [];
  for (let i = 0; i < nu; i++) {
    const v0 = (i * (i + 1)) >> 1, skip = i + 1;
    for (let j = 0; j < i; j++) {
      faces.push([j+v0, j+v0+skip, j+v0+skip+1]);
      faces.push([j+v0, j+v0+skip+1, j+v0+1]);
    }
    faces.push([i+v0, i+v0+skip, i+v0+skip+1]);
  }
  return faces;
}

function jsVertexOrdering(nu) {
  const left   = Array.from({length: nu-1}, (_,j) => 3+j);
  const right  = Array.from({length: nu-1}, (_,j) => nu+2+j);
  const bottom = Array.from({length: nu-1}, (_,j) => 2*nu+1+j);
  const insideLen = ((nu+1)*(nu+2)>>1) - 3*nu;
  const inside = Array.from({length: insideLen}, (_,j) => 3*nu+j);
  const o = [0];
  for (let i = 0; i < nu-1; i++) {
    o.push(left[i]);
    o.push(...inside.slice((i*(i-1))>>1, (i*(i+1))>>1));
    o.push(right[i]);
  }
  o.push(1, ...bottom, 2);
  return o;
}

function jsInsidePoints(vAB, vAC) {
  const pts = [];
  for (let i = 1; i < vAB.length; i++) {
    for (let k = 0; k < i; k++) {
      const wk = (k+1)/(i+1), wk2 = (i-k)/(i+1);
      pts.push([
        wk2*vAB[i][0] + wk*vAC[i][0],
        wk2*vAB[i][1] + wk*vAC[i][1],
        wk2*vAB[i][2] + wk*vAC[i][2]
      ]);
    }
  }
  return pts;
}

function jsSubdivideMesh(vertices, faces, nu) {
  const F = faces.length, V = vertices.length;

  // Collect sorted unique edges [min,max]
  const edgeMap = new Map();
  const edges = [];
  for (const [a,b,c] of faces) {
    for (const [u,v] of [[a,b],[b,c],[a,c]]) {
      const mn = Math.min(u,v), mx = Math.max(u,v), key = `${mn},${mx}`;
      if (!edgeMap.has(key)) { edgeMap.set(key, edges.length); edges.push([mn,mx]); }
    }
  }
  edges.sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  edgeMap.clear();
  for (let i = 0; i < edges.length; i++) edgeMap.set(`${edges[i][0]},${edges[i][1]}`, i);
  const E = edges.length;

  // edge_indices[u][v] = +i (u<v) or -i (u>v)
  const edge_indices = Array.from({length: V}, () => new Map());
  for (let i = 0; i < E; i++) {
    const [u,v] = edges[i];
    edge_indices[u].set(v, i);
    edge_indices[v].set(u, -i);
  }

  const insidePerFace = ((nu-1)*(nu-2))>>1;
  const subvertices = new Array(V + E*(nu-1) + F*insidePerFace);
  const subfaces    = new Array(F * nu * nu);
  for (let i = 0; i < V; i++) subvertices[i] = [...vertices[i]];

  const template = jsFacesTemplate(nu);
  const ordering = jsVertexOrdering(nu);
  const reordTpl = template.map(f => f.map(idx => ordering[idx]));
  const r = Array.from({length: nu-1}, (_,i) => i);

  // On-edge vertices
  for (let e = 0; e < E; e++) {
    const [u,v] = edges[e];
    for (let k = 0; k < nu-1; k++) {
      const w1 = (nu-1-k)/nu, w2 = (k+1)/nu;
      subvertices[V+e*(nu-1)+k] = [
        w1*vertices[u][0]+w2*vertices[v][0],
        w1*vertices[u][1]+w2*vertices[v][1],
        w1*vertices[u][2]+w2*vertices[v][2]
      ];
    }
  }

  // On-face vertices + subface connectivity
  for (let f = 0; f < F; f++) {
    const [fA,fB,fC] = faces[f];
    const tStart = f*insidePerFace + E*(nu-1) + V;
    const T = Array.from({length: insidePerFace}, (_,i) => tStart+i);

    const eAB = edge_indices[fA].get(fB);
    const eAC = edge_indices[fA].get(fC);
    const eBC = edge_indices[fB].get(fC);

    const mkEdge = (e) => r.map(k => Math.abs(e)*(nu-1)+V+k);
    const rev = (arr, flag) => flag ? [...arr].reverse() : arr;

    const AB = rev(mkEdge(eAB), eAB < 0);
    const AC = rev(mkEdge(eAC), eAC < 0);
    const BC = rev(mkEdge(eBC), eBC < 0);
    const VEF = [fA, fB, fC, ...AB, ...AC, ...BC, ...T];

    for (let s = 0; s < nu*nu; s++)
      subfaces[f*nu*nu+s] = reordTpl[s].map(idx => VEF[idx]);

    if (insidePerFace > 0) {
      const pts = jsInsidePoints(AB.map(i => subvertices[i]), AC.map(i => subvertices[i]));
      for (let t = 0; t < T.length; t++) subvertices[T[t]] = pts[t];
    }
  }
  return { vertices: subvertices, faces: subfaces };
}

function jsIcosphere(numFaces) {
  const nu = Math.max(1, Math.ceil(Math.sqrt(numFaces / 20)));
  let { vertices, faces } = jsIcosahedron();
  if (nu > 1) {
    ({ vertices, faces } = jsSubdivideMesh(vertices, faces, nu));
    const len = v => Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    vertices = vertices.map(v => { const l = len(v); return [v[0]/l,v[1]/l,v[2]/l]; });
  }
  return { vertices, faces };
}

// ─────────────────────────────────────────────────────────────────────────────

// (Re)allocates the module-level color/alpha buffers to match the active geometry's size.
// Buffer 1 is always allocated; 2/3 only when Dual View is active, to avoid wasted memory
// for the common single-view case. colorSize/alphaSize are geometry-dependent: instanceCount*3
// / instanceCount for InstancedMesh, faceCount*9 / faceCount*3 for sphere geometry.
function allocateColorBuffers(colorSize, alphaSize) {
  colorBuffer1 = new Float32Array(colorSize);
  alphaBuffer1 = new Float32Array(alphaSize);
  if (dualViewActive) {
    colorBuffer2 = new Float32Array(colorSize);
    alphaBuffer2 = new Float32Array(alphaSize);
    colorBuffer3 = new Float32Array(colorSize);
    alphaBuffer3 = new Float32Array(alphaSize);
  } else {
    colorBuffer2 = null; alphaBuffer2 = null;
    colorBuffer3 = null; alphaBuffer3 = null;
  }
}

// Takes an already-loaded TraceSource rather than raw file text: which format a file is in,
// and how to turn it into the columnar model, is the loader's business (js/tracesource.js),
// not the renderer's.
async function initVisualization(src, dimensions, axisMapping, numSphereFaces = 0) {
  if (masterTimeline) masterTimeline.kill();
  if (instancedMesh) { vizGroup.remove(instancedMesh); instancedMesh = null; }
  if (bordersObject) { vizGroup.remove(bordersObject); bordersObject = null; }
  if (sphereMesh) { vizGroup.remove(sphereMesh); sphereMesh = null; }
  if (sphereEdgesObject) { vizGroup.remove(sphereEdgesObject); sphereEdgesObject = null; }
  while (sphereLights.length) scene.remove(sphereLights.pop());
  while (sphereLightFolders.length) sphereLightFolders.pop().destroy();

  traceSource = src;
  const maxTicks = src.t1;
  const initialOffset = src.t0;   // trace start, in ticks — the wall-clock zero
  const instanceCount = src.nRanks;

  if (instanceCount === 0) {
    alert('Could not parse rank data from the file.');
    return;
  }

  sample = makeSample(instanceCount);
  speedVals = new Float64Array(instanceCount);
  speedPct = new Float64Array(instanceCount);

  // Bind each active scheme to this trace, then fetch the union of the columns they read —
  // only those. The nine eager whole-trace passes this replaced ran before the first frame
  // regardless of which metric was selected, and once per scheme on top of that.
  const wanted = new Set(prepareScheme(scheme1, src));
  if (dualViewActive) {
    for (const c of prepareScheme(scheme2, src)) wanted.add(c);
    for (const c of prepareScheme(scheme3, src)) wanted.add(c);
  }
  // A scheme that cannot answer must say why. Without this it would fetch no columns and
  // paint every rank the same, which looks like data.
  for (const sc of dualViewActive ? [scheme1, scheme2, scheme3] : [scheme1]) {
    if (sc.papiUnavailable) { alert(sc.papiUnavailable); return; }
  }
  await src.ensureColumns([...wanted]);
  await resolveSchemeAsync(scheme1);
  if (dualViewActive) { await resolveSchemeAsync(scheme2); await resolveSchemeAsync(scheme3); }
  // For .pv3dbin this fetches the first block; for the text path it is a no-op.
  if (src.ensure) await src.ensure(0);

  if (numSphereFaces > 0) {
    const nu = Math.max(1, Math.ceil(Math.sqrt(numSphereFaces / 20)));
    const actualFaces = 20 * nu * nu;
    if (actualFaces !== instanceCount) {
      alert(`Sphere face count mismatch: the selected option produces ${actualFaces} faces (ν=${nu}), but the trace file contains ${instanceCount} ranks. Please select a matching face count.`);
      return;
    }

    // ── Sphere geometry ───────────────────────────────────────────────────
    const { vertices: sv, faces: sf } = jsIcosphere(numSphereFaces);
    const faceCount = sf.length;
    const R = 8;
    allocateColorBuffers(faceCount * 9, faceCount * 3);

    const posArr   = new Float32Array(faceCount * 9);
    const colorArr = new Float32Array(faceCount * 9);
    const alphaArr = new Float32Array(faceCount * 3).fill(0);

    for (let f = 0; f < faceCount; f++) {
      for (let v = 0; v < 3; v++) {
        const vi = sf[f][v];
        posArr  [f*9+v*3+0] = sv[vi][0] * R;
        posArr  [f*9+v*3+1] = sv[vi][1] * R;
        posArr  [f*9+v*3+2] = sv[vi][2] * R;
        colorArr[f*9+v*3+0] = scheme1.compColor.r;
        colorArr[f*9+v*3+1] = scheme1.compColor.g;
        colorArr[f*9+v*3+2] = scheme1.compColor.b;
      }
    }

    const sphereGeo = new THREE.BufferGeometry();
    sphereGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    sphereGeo.setAttribute('color',    new THREE.BufferAttribute(colorArr, 3));
    sphereGeo.setAttribute('faceAlpha',new THREE.BufferAttribute(alphaArr, 1));
    sphereGeo.computeVertexNormals();

    const sphereSide = config.geometry.sphere?.doubleSide ? THREE.DoubleSide : THREE.FrontSide;
    const sphereMat = new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, flatShading: true, side: sphereSide });
    sphereMat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'attribute float faceAlpha;\nvarying float vFaceAlpha;\nvoid main() {\n  vFaceAlpha = faceAlpha;'
      );
      shader.fragmentShader = 'varying float vFaceAlpha;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <color_fragment>',
        '#include <color_fragment>\ndiffuseColor.a *= vFaceAlpha;'
      );
    };

    sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
    vizGroup.add(sphereMesh);

    // Extra lights for sphere mode
    const sphereLightDefs = [
      ['Sphere Light 1', 200, 5, 0],
      ['Sphere Light 2', -200, -5, 0],
      ['Sphere Light 3', 0, -25, 200],
      ['Sphere Light 4', 0, 25, -200],
    ];
    for (const [name, x, y, z] of sphereLightDefs) {
      const l = new THREE.PointLight(0xffffff, 2, 0, 0);
      l.position.set(x, y, z);
      scene.add(l);
      sphereLights.push(l);
      sphereLightFolders.push(addLightFolder(name, l));
    }

    // Indexed geometry for edge overlay
    const idxPos = new Float32Array(sv.length * 3);
    for (let i = 0; i < sv.length; i++) {
      idxPos[i*3+0] = sv[i][0]*R; idxPos[i*3+1] = sv[i][1]*R; idxPos[i*3+2] = sv[i][2]*R;
    }
    const idxArr = new Uint32Array(faceCount * 3);
    for (let f = 0; f < faceCount; f++) {
      idxArr[f*3+0] = sf[f][0]; idxArr[f*3+1] = sf[f][1]; idxArr[f*3+2] = sf[f][2];
    }
    const idxGeo = new THREE.BufferGeometry();
    idxGeo.setAttribute('position', new THREE.BufferAttribute(idxPos, 3));
    idxGeo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    sphereEdgesObject = new THREE.LineSegments(
      new THREE.EdgesGeometry(idxGeo),
      new THREE.LineBasicMaterial({ color: 0x444444 })
    );
    sphereEdgesObject.visible = !config.geometry.hideViewCores;
    vizGroup.add(sphereEdgesObject);

  } else {
    // ── Cuboid / Square geometry ───────────────────────────────────────────
    allocateColorBuffers(instanceCount * 3, instanceCount);
    const standardSpacing = parseFloat(config.geometry.advanced?.standardSpacing) || 0.5;
    const coordinates = generateSimpleGrid(instanceCount, dimensions, axisMapping, standardSpacing);
    const geometry = new THREE.BoxGeometry(0.4, 0.4, 0.4);

    const alphaArray = new Float32Array(instanceCount).fill(0.0);
    geometry.setAttribute('instanceAlpha', new THREE.InstancedBufferAttribute(alphaArray, 1));

    const material = new THREE.MeshStandardMaterial({ transparent: true, alphaTest: 0.01 });
    material.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'attribute float instanceAlpha;\nvarying float vInstanceAlpha;\nvoid main() {\n  vInstanceAlpha = instanceAlpha;'
      );
      shader.fragmentShader = 'varying float vInstanceAlpha;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <color_fragment>',
        '#include <color_fragment>\ndiffuseColor.a *= vInstanceAlpha;'
      );
    };

    instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
    vizGroup.add(instancedMesh);

    const dummy = new THREE.Object3D();
    for (let i = 0; i < instanceCount; i++) {
      if (coordinates[i]) dummy.position.copy(coordinates[i]);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);
      instancedMesh.setColorAt(i, scheme1.compColor);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate = true;

    const borderEdgeGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.4, 0.4, 0.4));
    const borderGeometries = [];
    for (let i = 0; i < instanceCount; i++) {
      if (coordinates[i]) {
        const instancedGeom = borderEdgeGeom.clone();
        instancedGeom.translate(coordinates[i].x, coordinates[i].y, coordinates[i].z);
        borderGeometries.push(instancedGeom);
      }
    }
    if (borderGeometries.length > 0) {
      const mergedBordersGeom = mergeGeometries(borderGeometries);
      bordersObject = new THREE.LineSegments(mergedBordersGeom, new THREE.LineBasicMaterial({ color: 0x444444 }));
      bordersObject.visible = !config.geometry.hideViewCores;
      vizGroup.add(bordersObject);
    }
  }

  setupAnimation(maxTicks, initialOffset);

  // Feed parsed trace data into the PixiJS vampir timeline HUD
  if (vampirReady) {
    loadTimelineData(await src.overview());
    updateVampirTimeline(playStart);   // ticks, not progress — see updateVampirTimeline
    vampirTexture.needsUpdate = true;
  }

  elements.playPauseBtn.textContent = 'Play';
  elements.slider.value = '0';
  resumeAfterLoad = false;
  updateDetailStatus(src);
  masterTimeline.pause();
  updateInstanceColors();
  updateClockDisplay(initialOffset);

  // Show controls
  elements.infoPanel.classList.remove('hidden');
  elements.controlsPanel.classList.remove('hidden');
  refreshLegendPanels();
}

elements.playPauseBtn.addEventListener('click', () => {
  if (!masterTimeline) return;
  masterTimeline.paused(!masterTimeline.paused());
  // A deliberate pause also cancels an automatic resume, so a block arriving does not
  // restart playback the user has just stopped.
  resumeAfterLoad = !masterTimeline.paused();
  elements.playPauseBtn.textContent = masterTimeline.paused() ? 'Play' : 'Pause';
});

elements.recordBtn?.addEventListener('click', () => {
  if (!masterTimeline) {
    alert('Load and visualize a trace first.');
    return;
  }
  if (renderer.xr.isPresenting) {
    alert('Recording is not available while in VR mode.');
    return;
  }

  if (isRecording()) {
    stopRecording();
    return;
  }

  // Reset to t=0 and force-paint that frame before capture starts, so the
  // very first recorded frame is the true start of the animation.
  masterTimeline.pause(0); // seeks to 0, fires onUpdate (colors/clock/HUD)
  elements.slider.value = '0';
  renderFrame();

  // Record a composite canvas (WebGL frame + clock/tooltip burned in) rather
  // than renderer.domElement directly, since the clock/tooltip are separate
  // DOM overlays that captureStream() on the raw canvas can't see.
  if (!compositeCanvas) compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = renderer.domElement.width;
  compositeCanvas.height = renderer.domElement.height;
  compositeCtx = compositeCanvas.getContext('2d');
  compositeCtx.drawImage(renderer.domElement, 0, 0);
  drawClockOverlay(compositeCtx);
  drawTooltipOverlay(compositeCtx);
  drawLegendOverlays(compositeCtx);

  const started = startRecording(compositeCanvas, {
    onStop: (blob) => {
      downloadBlob(blob, 'paraviz3d-recording');
      elements.recordBtn.textContent = '⏺ Record';
      elements.recordBtn.classList.remove('recording-active');
      elements.recordingIndicator?.classList.add('hidden');
    },
  });
  if (!started) return;

  masterTimeline.play(); // begin the pass that the recorder will capture
  elements.playPauseBtn.textContent = 'Pause';

  elements.recordBtn.textContent = '■ Stop Recording';
  elements.recordBtn.classList.add('recording-active');
  elements.recordingIndicator?.classList.remove('hidden');
});


// Converts the "Frame Step (µs)" UI field into a fraction of the timeline's
// 0-1 progress range. Read live (not cached) so changing the field takes
// effect on the very next click without needing to re-visualize.
function getFrameStep() {
  // Read the input directly rather than via `config` — config is a snapshot
  // only rebuilt on the Visualize click, so going through it here would make
  // this field require a re-visualize to take effect.
  const granularityMicroseconds = parseFloat(document.getElementById('timeGranularity')?.value) || 100;
  // Against the play span, not the trace: a frame step has to be the same fraction of
  // whatever the slider currently covers, or stepping would crawl inside a selection.
  const span = playEnd - playStart;
  return span > 0 ? microsecondsToTicks(granularityMicroseconds) / span : 0.0001;
}

elements.prevFrameBtn.addEventListener('click', () => {
  if (!masterTimeline) return;
  const newProgress = Math.max(0, masterTimeline.progress() - getFrameStep());
  masterTimeline.progress(newProgress).pause();
  elements.slider.value = newProgress.toString();
  updateInstanceColors();
  elements.playPauseBtn.textContent = 'Play';
});

elements.nextFrameBtn.addEventListener('click', () => {
  if (!masterTimeline) return;
  const newProgress = Math.min(1, masterTimeline.progress() + getFrameStep());
  masterTimeline.progress(newProgress).pause();
  elements.slider.value = newProgress.toString();
  updateInstanceColors();
  elements.playPauseBtn.textContent = 'Play';
});


elements.slider.addEventListener('input', () => {
  if (!masterTimeline) return;
  masterTimeline.progress(parseFloat(elements.slider.value)).pause();
  updateInstanceColors();
  elements.playPauseBtn.textContent = 'Play';
  renderFrame(); // force immediate repaint during drag
});

elements.clearSelectionBtn?.addEventListener('click', () => {
  if (masterTimeline) applySelection(null);
});

// Keyboard arrow keys for frame navigation
window.addEventListener('keydown', (e) => {
  if (!masterTimeline) return;
  if (e.key === 'Escape' && timelineSelection) {
    e.preventDefault();
    applySelection(null);
    return;
  }
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    const newProgress = Math.max(0, masterTimeline.progress() - getFrameStep());
    masterTimeline.progress(newProgress).pause();
    elements.slider.value = newProgress.toString();
    updateInstanceColors();
    elements.playPauseBtn.textContent = 'Play';
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    const newProgress = Math.min(1, masterTimeline.progress() + getFrameStep());
    masterTimeline.progress(newProgress).pause();
    elements.slider.value = newProgress.toString();
    updateInstanceColors();
    elements.playPauseBtn.textContent = 'Play';
  }
});

window.addEventListener('resize', () => {
  if (!camera || !renderer || !elements.rendererContainer) return;
  const width = elements.rendererContainer.clientWidth;
  const height = elements.rendererContainer.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  positionHUDForDesktop();
});

// Initialize
init();

// Timeline HUD interaction (must be after init() creates renderer)
const hoverRaycaster = new THREE.Raycaster();
const hoverMouse = new THREE.Vector2();
let isDraggingTimeline = false;
//: The in-progress HUD gesture, or null. `moved` is what separates a click (seek) from a
//: drag (select); it only flips once the pointer has travelled far enough to mean it.
let hudDrag = null;
const HUD_DRAG_THRESHOLD_PX = 3;

/**
 * Horizontal position on the HUD, 0-1, straight from the pointer's x.
 *
 * positionHUDForDesktop sizes the plane to exactly the viewport width at its own depth, so its
 * u axis and the canvas's x axis coincide. That makes this both cheaper than a raycast and
 * usable when the pointer has left the strip vertically — which a drag does constantly.
 */
function hudPointerU(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  return Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
}

/** Moves the playhead to a trace time, clamped into the play span. */
function seekToTicks(ticks) {
  const progress = Math.max(0, Math.min(1, timeToProgress(ticks)));
  masterTimeline.progress(progress).pause();
  elements.slider.value = progress.toString();
  elements.playPauseBtn.textContent = 'Play';
  refreshTimelineUI();
}

/**
 * Restricts playback to a region, or restores the whole trace when passed null.
 *
 * The tween is rebuilt over the region, which is what makes every existing `progress()` call
 * site — the slider, the frame buttons, the arrow keys — span the selection without knowing
 * it exists. Looping is an `onComplete` rather than `repeat: -1` so `progress()` keeps
 * meaning one unambiguous thing at the boundary.
 */
function applySelection(sel) {
  timelineSelection = sel;
  const start = sel ? sel.t0 : 0;
  const end = sel ? sel.t1 : currentMaxTicks;
  rebuildPlayTween(start, end, timeProxy.time);

  // The "x / y" beside the slider must describe what the slider now spans.
  if (elements.timelineMaxTime) elements.timelineMaxTime.textContent = formatTicks(end - start);
  if (elements.clearSelectionBtn) {
    elements.clearSelectionBtn.classList.toggle('hidden', !sel);
    if (sel) {
      elements.clearSelectionBtn.textContent =
        `✕ ${formatTicks(sel.t0)} → ${formatTicks(sel.t1)}`;
    }
  }
  setTimelineSelection(sel ? sel.t0 : null, sel ? sel.t1 : null);
  refreshTimelineUI();
}

function hudRaycast(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  hoverMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  hoverMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  hoverRaycaster.setFromCamera(hoverMouse, camera);
  return hoverRaycaster.intersectObject(vampirHudMesh);
}

/** Narrowest window the HUD may show: one of the trace's own quanta per pixel column.
 *  Below this the strip magnifies resolution the data does not have. */
function minViewSpan() {
  const quantum = (traceSource && traceSource.quant) || 1;   // .pv3dbin level quantum, else 1 µs
  return Math.max(quantum * HUD_CANVAS_WIDTH, 1);
}

/** Pushes the current view window to the HUD and repaints it. */
function applyTimelineView() {
  if (!vampirReady) return;
  setTimelineView(viewStart, viewEnd, timeProxy.time);
  if (vampirTexture) vampirTexture.needsUpdate = true;
}

/**
 * Zooms the timeline about a fixed trace time, Audacity-style: whatever was under the cursor
 * stays under it.
 */
function zoomTimelineAt(anchorTicks, factor) {
  const span = viewEnd - viewStart;
  const newSpan = Math.max(minViewSpan(), Math.min(currentMaxTicks, span * factor));
  if (newSpan === span) return false;

  // Keep `anchorTicks` at the same fraction across the window.
  const frac = span > 0 ? (anchorTicks - viewStart) / span : 0.5;
  let start = anchorTicks - frac * newSpan;
  let end = start + newSpan;
  if (start < 0) { start = 0; end = newSpan; }
  if (end > currentMaxTicks) { end = currentMaxTicks; start = Math.max(0, end - newSpan); }

  viewStart = start;
  viewEnd = end;
  applyTimelineView();
  return true;
}

// Wheel over the HUD zooms the timeline; wheel anywhere else keeps dollying the camera.
//
// The listener has to sit on an *ancestor* of the canvas, in the capture phase. OrbitControls
// registers its own `wheel` on `renderer.domElement`, and listeners on the same element run in
// registration order whatever their capture flag — so a capturing listener on the canvas would
// not reliably win. Capturing on the container does, and stopPropagation() then keeps the
// event away from OrbitControls entirely.
elements.rendererContainer.addEventListener('wheel', (event) => {
  if (!vampirHudMesh || !vampirReady || !masterTimeline) return;
  const hits = hudRaycast(event);
  if (!hits.length) return;                 // not over the strip: let the camera have it

  event.preventDefault();
  event.stopPropagation();

  // One notch per event, direction from the sign. 1.2 per notch is ~4 notches per octave.
  const factor = event.deltaY > 0 ? 1.2 : 1 / 1.2;
  if (zoomTimelineAt(uvToTime(hits[0].uv.x), factor)) syncFrameStepToZoom();
}, { capture: true, passive: false });

// Typing in Frame Step zooms the strip to match, so the field and the wheel are two ways of
// setting one thing rather than two settings that can disagree.
document.getElementById('timeGranularity')?.addEventListener('change', syncZoomToFrameStep);

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (!vampirHudMesh || !vampirReady || !masterTimeline) return;
  const intersects = hudRaycast(event);
  if (intersects.length === 0) return;

  // Press starts a gesture that is not yet a seek *or* a selection — which it becomes is
  // decided on release by whether the pointer moved. Seeking here, as this used to, would
  // make every drag scrub as well as select.
  hudDrag = { startTicks: uvToTime(intersects[0].uv.x), startX: event.clientX, moved: false };
  isDraggingTimeline = true;
  controls.enabled = false;
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (!vampirHudMesh || !vampirReady) return;

  if (hudDrag && masterTimeline) {
    if (Math.abs(event.clientX - hudDrag.startX) > HUD_DRAG_THRESHOLD_PX) hudDrag.moved = true;
    if (hudDrag.moved) {
      // Deliberately not raycast: during a fast drag the pointer leaves the strip, and a
      // raycast miss would freeze the edge where it left. The HUD plane spans the full
      // viewport width at its depth, so the x fraction of the canvas *is* its u.
      const t = uvToTime(hudPointerU(event));
      setTimelineSelection(hudDrag.startTicks, t);
      if (vampirTexture) vampirTexture.needsUpdate = true;
    }
    return;
  }

  // Cheap early-out: skip raycast if pointer is in the upper 60% of the canvas
  const rect = renderer.domElement.getBoundingClientRect();
  const normY = (event.clientY - rect.top) / rect.height;
  if (normY < 0.6) return;

  // Tooltip hover (only when not dragging)
  const intersects = hudRaycast(event);
  if (intersects.length > 0) {
    const uv = intersects[0].uv;
    handleTimelinePointer(uv.x, 1 - uv.y); // flip Y: Three.js UV 0=bottom, PixiJS 0=top
  } else {
    handleTimelinePointer(null, null);
  }
  if (vampirTexture) vampirTexture.needsUpdate = true;
});

window.addEventListener('pointerup', (event) => {
  if (!isDraggingTimeline) return;
  isDraggingTimeline = false;
  if (!renderer.xr.isPresenting) controls.enabled = true;

  const drag = hudDrag;
  hudDrag = null;
  if (!drag || !masterTimeline) return;

  if (!drag.moved) {
    // A press without movement is a click: seek, and leave any selection alone.
    seekToTicks(drag.startTicks);
    return;
  }

  const t0 = Math.min(drag.startTicks, uvToTime(hudPointerU(event)));
  const t1 = Math.max(drag.startTicks, uvToTime(hudPointerU(event)));
  // Too narrow to be meant: a few pixels of twitch should not trap playback in a sliver.
  if (t1 - t0 < Math.max(liveFrameStepTicks(), (viewEnd - viewStart) / HUD_CANVAS_WIDTH * 2)) {
    setTimelineSelection(null, null);
    applySelection(null);
    seekToTicks(drag.startTicks);
    return;
  }
  applySelection({ t0, t1 });
});

// Rank hover tooltip
const rankTooltip = document.getElementById('rank-hover-tooltip');
const meshRaycaster = new THREE.Raycaster();
const meshMouse = new THREE.Vector2();

renderer.domElement.addEventListener('pointermove', (event) => {
  if (!rankTooltip || !config.hoverInfoOptions?.rankName) {
    if (rankTooltip) rankTooltip.style.display = 'none';
    hoverTooltip.visible = false;
    return;
  }
  const rect = renderer.domElement.getBoundingClientRect();
  meshMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  meshMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  meshRaycaster.setFromCamera(meshMouse, camera);

  let rankIndex = -1;
  if (instancedMesh) {
    const hits = meshRaycaster.intersectObject(instancedMesh);
    if (hits.length > 0) rankIndex = hits[0].instanceId;
  } else if (sphereMesh) {
    const hits = meshRaycaster.intersectObject(sphereMesh);
    if (hits.length > 0) rankIndex = hits[0].faceIndex;
  }

  if (rankIndex >= 0) {
    const text = `Rank ${rankIndex}`;
    rankTooltip.textContent = text;
    rankTooltip.style.display = 'block';
    rankTooltip.style.left = `${event.clientX + 12}px`;
    rankTooltip.style.top  = `${event.clientY + 12}px`;
    hoverTooltip = { visible: true, text, x: event.clientX - rect.left + 12, y: event.clientY - rect.top + 12 };
  } else {
    rankTooltip.style.display = 'none';
    hoverTooltip.visible = false;
  }
});

renderer.domElement.addEventListener('pointerleave', () => {
  if (rankTooltip) rankTooltip.style.display = 'none';
  hoverTooltip.visible = false;
});

// Test display topology button — colors consecutive groups of 18 faces/ranks with maximally
// distinct hues (golden-ratio stepping), so the linear index->position mapping can be sanity
// checked visually. Works for the sphere's raw per-face-vertex color buffer, and for the
// square/cuboid InstancedMesh's per-instance color.
function groupTopologyColor(group) {
  const hue = (group * 0.618033988749895) % 1;
  return new THREE.Color().setHSL(hue, 0.85, 0.55);
}

document.getElementById('testTopologyBtn')?.addEventListener('click', () => {
  if (sphereMesh) {
    const colorAttr = sphereMesh.geometry.attributes.color;
    const faceCount = colorAttr.array.length / 9;
    const groupColors = [];
    for (let f = 0; f < faceCount; f++) {
      const group = Math.floor(f / 18);
      if (!groupColors[group]) groupColors[group] = groupTopologyColor(group);
      const c = groupColors[group];
      for (let v = 0; v < 3; v++) {
        colorAttr.array[f * 9 + v * 3 + 0] = c.r;
        colorAttr.array[f * 9 + v * 3 + 1] = c.g;
        colorAttr.array[f * 9 + v * 3 + 2] = c.b;
      }
    }
    colorAttr.needsUpdate = true;
  } else if (instancedMesh) {
    const groupColors = [];
    for (let i = 0; i < instancedMesh.count; i++) {
      const group = Math.floor(i / 18);
      if (!groupColors[group]) groupColors[group] = groupTopologyColor(group);
      instancedMesh.setColorAt(i, groupColors[group]);
    }
    instancedMesh.instanceColor.needsUpdate = true;
  } else {
    alert('Visualize with Sphere, Square, or Cuboid geometry first.');
  }
});

renderer.setAnimationLoop(animate);

// Wire interface Visualize button to drive the Three.js scene
/**
 * Picks the detail level to draw: the coarsest whose quantum still resolves the jump between
 * two frames, since a finer one would fetch more data to produce the same picture. The user
 * can override it; "auto" re-reads the frame-step field live, exactly as getFrameStep() does.
 */
function chooseDetailLevel(file) {
  const choice = elements.detailLevelSelect?.value ?? 'auto';
  const available = file.levels.map(l => l.level);
  if (choice !== 'auto') {
    const want = Number(choice);
    if (available.includes(want)) return want;
    console.warn(`pv3d: level ${want} is not in this file; levels present are ${available.join(', ')}`);
  }
  const granularityUs = parseFloat(document.getElementById('timeGranularity')?.value) || 100;
  const frameStepCycles = granularityUs * 1e-6 * file.timerResolution;
  return file.levelForStep(frameStepCycles);
}

/** Opens whichever format the chosen file is in, sniffed from its 8-byte magic. */
async function openTraceSource() {
  const timeskip = parseFloat(config.controls.beginTime) || 0;

  if (config.dataSource !== 'server' && config.localFile
      && /\.(pv3dbin|pv3dtxt)$/i.test(config.localFile.name)) {
    // A local .pv3dbin is read by byte range straight off disk — no server involved.
    return openPv3d(fileTransport(config.localFile));
  }

  if (config.dataSource === 'server') {
    const transport = httpTransport(config.serverFile);
    let kind;
    try {
      kind = await sniff(transport);
    } catch (err) {
      alert(`Failed to load server file: ${err.message}`);
      return null;
    }
    if (kind === 'pv3dbin') return openPv3d(transport);
    if (kind === 'pv3dtxt') {
      alert('This is a .pv3dtxt file — the text mirror of the binary format, meant for '
            + 'inspection and testing. Convert with --format bin to view it.');
      return null;
    }
    // Legacy text: still read whole, because it has no index to seek with.
    const response = await fetch(config.serverFile);
    if (!response.ok) { alert(`Failed to load server file: HTTP ${response.status}`); return null; }
    return parseLegacyText(await response.text(), timeskip);
  }

  if (!config.localFileData) {
    alert('No trace data available. Please select or upload a file.');
    return null;
  }
  return parseLegacyText(config.localFileData, timeskip);
}

function parseLegacyText(raw, timeskip) {
  const src = TextSource.parse(raw, timeskip, config.controls.ignoreMpiInit);
  if (!src) {
    alert('Parsing Error: The file does not appear to contain any valid "Rank" data blocks. '
          + 'Please check the file format.');
  }
  return src;
}

async function openPv3d(transport) {
  try {
    const file = await Pv3dFile.open(transport);
    const level = chooseDetailLevel(file);
    const lvl = file.level(level);
    await lvl.load();
    const src = new Pv3dSource(file, lvl);
    const reason = src.unsupportedReason();
    if (reason) { alert(reason); return null; }
    return src;
  } catch (err) {
    if (err instanceof RangeUnsupportedError) alert(err.message);
    else alert(`Failed to open the PV3D trace: ${err.message}`);
    return null;
  }
}

document.getElementById('visualizeButton').addEventListener('click', async () => {
  // config is already populated by interface.js handler (fires first)
  const source = await openTraceSource();
  if (!source) return;

  // Convert interface axis names (width/height/depth) to viz.js axis mapping (x/y/z)
  const physMap = { width: 'x', height: 'y', depth: 'z' };
  const axisMapping = {};
  let gridDim;

  if (config.geometry.type === 'square') {
    const axis1 = config.geometry.square.axis1;
    const axis2 = config.geometry.square.axis2;
    const axis3 = ['width', 'height', 'depth'].find(a => a !== axis1 && a !== axis2);
    axisMapping[physMap[axis1]] = 'x';
    axisMapping[physMap[axis2]] = 'y';
    axisMapping[physMap[axis3]] = 'z';
    const d1 = config.geometry.square.dim1 ? parseInt(config.geometry.square.dim1) : undefined;
    const d2 = config.geometry.square.dim2 ? parseInt(config.geometry.square.dim2) : undefined;
    gridDim = [d1, d2, 1];
  } else {
    const d1 = config.geometry.cuboid.dim1 ? parseInt(config.geometry.cuboid.dim1) : undefined;
    const d2 = config.geometry.cuboid.dim2 ? parseInt(config.geometry.cuboid.dim2) : undefined;
    const d3 = config.geometry.cuboid.dim3 ? parseInt(config.geometry.cuboid.dim3) : undefined;
    gridDim = (d1 || d2 || d3) ? [d1, d2, d3] : null;
    axisMapping[physMap[config.geometry.cuboid.axis1]] = 'x';
    axisMapping[physMap[config.geometry.cuboid.axis2]] = 'y';
    axisMapping[physMap[config.geometry.cuboid.axis3]] = 'z';
  }

  // Create or destroy vampir timeline HUD based on interface checkbox
  destroyVampirHUD();
  if (config.metric.includeTraditional) {
    const { canvas } = await initVampirTimeline();
    createVampirHUD(canvas);
  }

  // Apply metric colors from interface color pickers — scheme1 always from slot 1; when Dual
  // View is checked, scheme2/scheme3 additionally populated from slots 2/3.
  dualViewActive = !!config.metric.dualView;
  applyMetricConfigToScheme(config.metric.type1, config.metric.colors1, {
    heatmapCutoffs: config.metric.heatmapCutoffs1,
    betweenMpi: config.metric.betweenMpi1,
    windowSize: config.metric.windowSize1,
    useGradient: config.metric.useGradient1,
    runTargetFn: config.metric.runTargetFn1,
    papiFormula: config.metric.papiFormula1,
  }, scheme1);
  if (dualViewActive) {
    applyMetricConfigToScheme(config.metric.type2, config.metric.colors2, {
      heatmapCutoffs: config.metric.heatmapCutoffs2,
      betweenMpi: config.metric.betweenMpi2,
      windowSize: config.metric.windowSize2,
      useGradient: config.metric.useGradient2,
      runTargetFn: config.metric.runTargetFn2,
      papiFormula: config.metric.papiFormula2,
    }, scheme2);
    applyMetricConfigToScheme(config.metric.type3, config.metric.colors3, {
      heatmapCutoffs: config.metric.heatmapCutoffs3,
      betweenMpi: config.metric.betweenMpi3,
      windowSize: config.metric.windowSize3,
      useGradient: config.metric.useGradient3,
      runTargetFn: config.metric.runTargetFn3,
      papiFormula: config.metric.papiFormula3,
    }, scheme3);
  }

  applyFilterIndicators(source);

  const numSphereFaces = config.geometry.type === 'sphere' ? parseInt(config.geometry.sphere.faces) || 0 : 0;
  await initVisualization(source, gridDim, axisMapping, numSphereFaces);
});

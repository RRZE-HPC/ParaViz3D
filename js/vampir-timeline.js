import * as PIXI from 'pixi';

// Configuration
const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 600;

//: Exported so viz.js can express zoom limits in pixel columns without duplicating the number.
export const HUD_CANVAS_WIDTH = CANVAS_WIDTH;

let app;
let lanesContainer;
let barsContainer;
let playhead;
let tooltip;
let timelineDuration = 0;
//: The rasterised swim-lane image and the columnar data behind it. The bars used to be one
//: PIXI.Graphics plus one JS object each — 220,831 of both on the test trace, linear-scanned
//: on every pointer move. Painting them into one texture makes the cost proportional to the
//: pixels drawn instead of to the interval count, and hit-testing a binary search.
let barsSprite = null;
let overviewData = null;

//: The visible time window, in ticks — the HUD's own span, independent of what GSAP animates.
//: Defaults to the whole trace, so with no zoom every mapping below is the identity it was.
//: viz.js owns this window and pushes it here through setTimelineView(); this is a mirror, so
//: there is one source of truth for a value both modules convert against.
let viewStart = 0;
let viewEnd = 0;

//: The marked region in ticks, or null. Absolute trace time, so zooming moves it across the
//: strip without changing it.
let selection = null;
let selectionGraphics = null;

/** Visible-window position of a trace time, in canvas pixels. The single time->x mapping:
 *  the raster and the hit-test both go through it, where they used to each have their own. */
function timeToX(t) {
    const span = viewEnd - viewStart;
    return span > 0 ? ((t - viewStart) / span) * CANVAS_WIDTH : 0;
}

/** Canvas pixel -> trace time. */
function xToTime(x) {
    return viewStart + (x / CANVAS_WIDTH) * (viewEnd - viewStart);
}

/**
 * Paints every interval inside the visible window into one RGBA buffer and uploads it as a
 * single sprite.
 *
 * At this scale a bar is one or two pixels wide, so there is nothing to gain from a display
 * object per interval and a great deal to lose: 220,831 of them took seconds to build and made
 * every pointer move a linear scan.
 *
 * Only the visible slice is touched. Each rank's intervals are sorted by start, so a binary
 * search finds where the window begins and the loop stops once it ends — zooming in therefore
 * gets *cheaper*, instead of re-scanning tens of millions of rows per wheel notch.
 */
function rasteriseBars() {
    if (!overviewData || !app) return;
    const numRanks = overviewData.nRanks;
    if (numRanks === 0) return;

    if (barsSprite) { barsSprite.destroy({ texture: true, textureSource: true }); barsSprite = null; }

    const laneHeight = CANVAS_HEIGHT / numRanks;
    const pixels = new Uint8ClampedArray(CANVAS_WIDTH * CANVAS_HEIGHT * 4);
    const barTop = Math.max(0, Math.round(laneHeight * 0.1));
    const barBottom = Math.max(barTop + 1, Math.round(laneHeight - laneHeight * 0.1));

    for (let r = 0; r < numRanks; r++) {
        const starts = overviewData.starts[r], ends = overviewData.ends[r];
        const count = overviewData.counts[r];
        const laneY = r * laneHeight;
        const y0 = Math.min(CANVAS_HEIGHT - 1, Math.floor(laneY + barTop));
        const y1 = Math.min(CANVAS_HEIGHT, Math.ceil(laneY + barBottom));

        // First interval that could still be on screen: the earliest whose end reaches the
        // window. Ends are ascending because the intervals do not overlap.
        let lo = 0, hi = count;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (ends[mid] < viewStart) lo = mid + 1; else hi = mid; }

        for (let p = lo; p < count; p++) {
            if (starts[p] > viewEnd) break;                 // sorted: nothing later can show
            let x0 = Math.floor(timeToX(starts[p]));
            let x1 = Math.ceil(timeToX(ends[p]));
            if (x1 <= x0) x1 = x0 + 1;                      // a sub-pixel call still shows
            if (x1 <= 0 || x0 >= CANVAS_WIDTH) continue;
            if (x0 < 0) x0 = 0;
            if (x1 > CANVAS_WIDTH) x1 = CANVAS_WIDTH;
            for (let y = y0; y < y1; y++) {
                let o = (y * CANVAS_WIDTH + x0) * 4;
                for (let x = x0; x < x1; x++, o += 4) {
                    pixels[o] = 255; pixels[o + 1] = 0; pixels[o + 2] = 0; pixels[o + 3] = 217;
                }
            }
        }
    }

    const texture = PIXI.Texture.from({
        resource: pixels,
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        alphaMode: 'premultiply-alpha-on-upload',
    });
    barsSprite = new PIXI.Sprite(texture);
    barsContainer.addChild(barsSprite);
}

/**
 * Initialize the PixiJS timeline offscreen.
 * Returns the canvas element and dimensions for use as a Three.js texture source.
 */
export async function initVampirTimeline() {
    app = new PIXI.Application();
    await app.init({
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        backgroundAlpha: 0,
        antialias: true,
    });

    // Stop auto-rendering — we render manually
    app.ticker.stop();

    // Containers for lane backgrounds and event bars
    lanesContainer = new PIXI.Container();
    app.stage.addChild(lanesContainer);

    barsContainer = new PIXI.Container();
    app.stage.addChild(barsContainer);

    // Selection overlay: above the bars so the flags read, below the playhead so it still
    // shows inside a selection.
    selectionGraphics = new PIXI.Graphics();
    selectionGraphics.visible = false;
    app.stage.addChild(selectionGraphics);

    // Playhead (white vertical line)
    playhead = new PIXI.Graphics();
    playhead.rect(0, 0, 2, CANVAS_HEIGHT);
    playhead.fill({ color: 0xffffff });
    playhead.x = 0;
    app.stage.addChild(playhead);

    // Playhead glow
    const playheadGlow = new PIXI.Graphics();
    playheadGlow.rect(-4, 0, 10, CANVAS_HEIGHT);
    playheadGlow.fill({ color: 0xffffff, alpha: 0.15 });
    playhead.addChild(playheadGlow);

    // Tooltip text (hidden until hover)
    tooltip = new PIXI.Text({
        text: '',
        style: {
            fontFamily: 'monospace',
            fontSize: 13,
            fill: 0xffffff,
            stroke: { color: 0x000000, width: 3 },
        }
    });
    tooltip.visible = false;
    app.stage.addChild(tooltip);

    // Render initial empty frame
    app.render();

    return {
        canvas: app.canvas,
        canvasWidth: CANVAS_WIDTH,
        canvasHeight: CANVAS_HEIGHT,
    };
}

/**
 * Load trace data into the timeline.
 * @param {{nRanks: number, counts: Int32Array, starts: Float64Array[], ends: Float64Array[],
 *          regions: Uint16Array[], regionNames: string[], t1: number}} overview
 *        Columnar per-rank view from TraceSource.overview() — the renderer no longer keeps a
 *        per-interval object array for this HUD to borrow.
 */
export function loadTimelineData(overview) {
    if (!app) return;

    const maxTime = overview.t1;
    timelineDuration = maxTime;
    overviewData = overview;
    viewStart = 0;
    viewEnd = maxTime;          // a fresh trace always starts fully zoomed out
    selection = null;           // ... and with nothing selected
    lanesContainer.removeChildren();
    barsContainer.removeChildren();
    if (barsSprite) { barsSprite.destroy({ texture: true, textureSource: true }); barsSprite = null; }

    const numRanks = overview.nRanks;
    if (numRanks === 0) return;

    const laneHeight = CANVAS_HEIGHT / numRanks;

    // Draw lane backgrounds and separators
    const lanesGraphics = new PIXI.Graphics();
    for (let r = 0; r < numRanks; r++) {
        const y = r * laneHeight;

        // Alternating subtle background
        if (r % 2 === 0) {
            lanesGraphics.rect(0, y, CANVAS_WIDTH, laneHeight);
            lanesGraphics.fill({ color: 0xffffff, alpha: 0.05 });
        }

        // Lane separator line
        if (r > 0) {
            lanesGraphics.moveTo(0, y);
            lanesGraphics.lineTo(CANVAS_WIDTH, y);
            lanesGraphics.stroke({ color: 0xffffff, width: 0.5, alpha: 0.15 });
        }
    }
    lanesContainer.addChild(lanesGraphics);

    rasteriseBars();

    // Reset playhead height in case canvas didn't change
    playhead.clear();
    playhead.rect(0, 0, 2, CANVAS_HEIGHT);
    playhead.fill({ color: 0xffffff });

    app.render();
}

/**
 * Draws the marked region: the unselected parts dimmed, the region itself tinted, a flag at
 * each edge.
 *
 * Dimming the *outside* rather than tinting the inside is the Audacity convention and the
 * reason it reads well — the data you care about keeps its full contrast.
 */
function drawSelection() {
    if (!selectionGraphics) return;
    selectionGraphics.clear();
    if (!selection) { selectionGraphics.visible = false; return; }
    selectionGraphics.visible = true;

    const x0 = timeToX(selection.t0);
    const x1 = timeToX(selection.t1);
    const a = Math.max(0, Math.min(CANVAS_WIDTH, x0));
    const b = Math.max(0, Math.min(CANVAS_WIDTH, x1));

    // Shade everything outside the region.
    if (a > 0) { selectionGraphics.rect(0, 0, a, CANVAS_HEIGHT); }
    if (b < CANVAS_WIDTH) { selectionGraphics.rect(b, 0, CANVAS_WIDTH - b, CANVAS_HEIGHT); }
    selectionGraphics.fill({ color: 0x000000, alpha: 0.5 });

    // A faint tint inside, so a selection is still visible when it covers the whole view.
    selectionGraphics.rect(a, 0, Math.max(2, b - a), CANVAS_HEIGHT);
    selectionGraphics.fill({ color: 0x4aa3ff, alpha: 0.10 });

    // Flags, drawn only for an edge that is actually on screen.
    for (const x of [x0, x1]) {
        if (x < 0 || x > CANVAS_WIDTH) continue;
        selectionGraphics.rect(Math.min(CANVAS_WIDTH - 2, Math.max(0, x - 1)), 0, 2, CANVAS_HEIGHT);
        selectionGraphics.fill({ color: 0x4aa3ff, alpha: 0.95 });
    }
}

/** Marks a region, in ticks. Pass null to clear it. */
export function setTimelineSelection(t0, t1) {
    if (!app) return;
    selection = (t0 === null || t1 === null) ? null : { t0: Math.min(t0, t1), t1: Math.max(t0, t1) };
    drawSelection();
    app.render();
}

/**
 * Sets the visible time window and repaints the bars for it.
 *
 * Cheap enough to call on every wheel notch: the raster only walks the intervals inside the
 * window, so zooming in costs less work, not more.
 */
export function setTimelineView(startTicks, endTicks, playheadTicks) {
    if (!app) return;
    viewStart = startTicks;
    viewEnd = endTicks;
    rasteriseBars();
    drawSelection();          // the region is in absolute ticks; its pixels move with the view
    if (playheadTicks !== undefined) updateVampirTimeline(playheadTicks);
    else app.render();
}

/**
 * Move the playhead to a trace time, in ticks.
 *
 * Takes time rather than the old 0–1 progress because the playhead's x position depends on the
 * zoom window, which progress cannot express: the same progress sits at a different pixel once
 * the view is narrowed, and at no pixel at all when the playhead is outside it.
 */
export function updateVampirTimeline(ticks) {
    if (!app) return;

    const x = timeToX(ticks);
    playhead.visible = x >= 0 && x <= CANVAS_WIDTH;
    playhead.x = Math.max(0, Math.min(CANVAS_WIDTH - 2, x));

    // Render PixiJS frame to offscreen canvas
    app.render();
}

/**
 * Handle pointer hover for tooltip display.
 * @param {number|null} normX - normalized X (0–1), null to hide
 * @param {number|null} normY - normalized Y (0–1), null to hide
 */
export function handleTimelinePointer(normX, normY) {
    if (!app || !tooltip) return;

    if (normX === null || normY === null) {
        tooltip.visible = false;
        app.render();
        return;
    }

    const px = normX * CANVAS_WIDTH;
    const py = normY * CANVAS_HEIGHT;

    // Hit-test: the lane picks the rank, then one binary search inside that rank's slice —
    // a scan of every bar on the canvas used to run on every pointer move.
    let found = null;
    if (overviewData && overviewData.nRanks > 0 && timelineDuration > 0) {
        const r = Math.min(overviewData.nRanks - 1, Math.max(0, Math.floor(normY * overviewData.nRanks)));
        const starts = overviewData.starts[r], ends = overviewData.ends[r];
        const count = overviewData.counts[r];
        const t = xToTime(px);
        // Widen by a pixel of time so a sub-pixel call is still reachable by the cursor. In
        // view units, so it tightens as you zoom in rather than staying at the whole-trace
        // scale and swallowing neighbouring calls.
        const slack = (viewEnd - viewStart) / CANVAS_WIDTH;
        let lo = 0, hi = count;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (ends[mid] + slack < t) lo = mid + 1; else hi = mid; }
        if (lo < count && starts[lo] - slack <= t) {
            found = { fn: overviewData.regionNames[overviewData.regions[r][lo]] || 'MPI',
                      rank: r, start: starts[lo], end: ends[lo] };
        }
    }

    if (found) {
        tooltip.text = `rank ${found.rank}  ${found.fn}`;
        // Position tooltip near cursor, clamped to canvas bounds
        tooltip.x = Math.min(px + 10, CANVAS_WIDTH - tooltip.width - 10);
        tooltip.y = Math.max(py - 25, 5);
        tooltip.visible = true;
    } else {
        tooltip.visible = false;
    }

    app.render();
}

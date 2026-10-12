import Pickr from 'pickr';


// Pickr instance management
const pickrInstances = new Map();
const pickrLabelMap = new Map(); // pickr element id -> label text
let pickrIdCounter = 0;

function destroyPickrIn(container) {
    if (!container) return;
    container.querySelectorAll('.pickr-container').forEach(el => {
        const instance = pickrInstances.get(el.id);
        if (instance) {
            try { instance.destroyAndRemove(); } catch(e) {}
            pickrInstances.delete(el.id);
        }
    });
}

function initPickrIn(container) {
    if (!container) return;
    container.querySelectorAll('.pickr-container').forEach(el => {
        if (pickrInstances.has(el.id)) return;
        try {
            const instance = Pickr.create({
                el: el,
                theme: 'nano',
                default: el.dataset.default || 'rgba(255, 255, 255, 1)',
                components: {
                    preview: true,
                    opacity: true,
                    hue: true,
                    interaction: {
                        input: true,
                        save: true
                    }
                }
            });
            instance.on('save', () => instance.hide());
            instance.on('hide', () => instance.applyColor(true));
            pickrInstances.set(el.id, instance);
        } catch(e) {
            console.warn('Pickr init failed for', el.id, e);
        }
    });
}

// Data source handling
const dataSourceRadios = document.querySelectorAll('input[name="dataSource"]');
const localFileInput = document.getElementById('localFileInput');
const serverFileList = document.getElementById('serverFileList');
const serverFileSelect = document.getElementById('serverFileSelect');

function syncDataSourceUI() {
    const selected = document.querySelector('input[name="dataSource"]:checked')?.value;
    if (selected === 'local') {
        localFileInput.classList.remove('hidden');
        serverFileList.classList.add('hidden');
    } else {
        localFileInput.classList.add('hidden');
        serverFileList.classList.remove('hidden');
    }
}

dataSourceRadios.forEach(radio => {
    radio.addEventListener('change', syncDataSourceUI);
});

// Sync UI on load in case browser restored radio state
syncDataSourceUI();

// Local file upload handling
const dataFileInput = document.getElementById('dataFile');
const localFileName = document.getElementById('localFileName');
export let uploadedFileData = null;
//: The File object itself, kept alongside the text. A .pv3dbin is read by byte range through
//: file.slice(), so reading it into a string would both corrupt it and defeat the point;
//: viz.js decides which of the two to use once it has sniffed the magic.
export let uploadedFile = null;

dataFileInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) {
        uploadedFileData = null;
        uploadedFile = null;
        localFileName.textContent = 'No file chosen';
        return;
    }
    uploadedFile = file;
    // Only the legacy text format needs the whole file as a string; reading a multi-gigabyte
    // .pv3dbin this way would exhaust the heap before anything could be drawn.
    uploadedFileData = /\.(pv3dbin|pv3dtxt)$/i.test(file.name) ? null : await file.text();
    localFileName.textContent = file.name;
});

const customGeometryFileInput = document.getElementById('customGeometryFile');
const customGeometryFileName  = document.getElementById('customGeometryFileName');
export let uploadedCustomGeometryFile = null; // raw File object (binary for GLB, text for TXT)

const customDoubleSideGroup = document.getElementById('customDoubleSideGroup');

customGeometryFileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) {
        uploadedCustomGeometryFile = null;
        customGeometryFileName.textContent = 'No file chosen';
        customDoubleSideGroup.style.display = 'none';
        return;
    }
    uploadedCustomGeometryFile = file;
    customGeometryFileName.textContent = file.name;
    const isGlb = /\.(glb|gltf)$/i.test(file.name);
    customDoubleSideGroup.style.display = isGlb ? 'block' : 'none';
});

// Fetch server file list
// serve.py generates /api/list by walking the trace folders, which is always current;
// api/files/list.txtt is hand-maintained and drifts (it lists files that no longer exist and
// omits most of the ones that do), so it is only the fallback.
const TRACE_FILE_RE = /\.(txt|pv3dbin|pv3dtxt)$/i;

async function loadServerFileList() {
    let files = null;
    try {
        const r = await fetch('./api/list');
        if (r.ok) files = (await r.json()).map(e => (typeof e === 'string' ? e : e.path));
    } catch { /* fall through to the static listing */ }
    if (!files) {
        try {
            const r = await fetch('./api/files/list.txtt');
            if (r.ok) files = await r.json();
        } catch { /* nothing available */ }
    }
    serverFileSelect.innerHTML = '';
    const traces = (files || []).filter(f => TRACE_FILE_RE.test(f));
    if (traces.length === 0) {
        serverFileSelect.innerHTML = '<option disabled>No trace files found</option>';
        return;
    }
    for (const file of traces) {
        const option = document.createElement('option');
        option.value = file;
        option.textContent = file.replace(/^api\/(files|pv3d)\//, '');
        serverFileSelect.appendChild(option);
    }
}
loadServerFileList();

// Geometry type handling
export const geometryType = document.getElementById('geometryType');
const squareOptions = document.getElementById('squareOptions');
const cuboidOptions = document.getElementById('cuboidOptions');
const sphereOptions = document.getElementById('sphereOptions');
const hypercuboidOptions = document.getElementById('hypercuboidOptions');
const customOptions = document.getElementById('customOptions');
const machineTopologyFields = document.getElementById('machineTopologyFields');
const advancedOptionsContainer = document.getElementById('advancedOptionsContainer');
const testTopologyBtnWrapper = document.getElementById('testTopologyBtnWrapper');

function updateGeometryOptions() {
    squareOptions.classList.add('hidden');
    cuboidOptions.classList.add('hidden');
    sphereOptions.classList.add('hidden');
    hypercuboidOptions.classList.add('hidden');
    customOptions.classList.add('hidden');
    machineTopologyFields.classList.add('hidden');
    advancedOptionsContainer.classList.add('hidden');

    const type = geometryType.value;

    // "Test display topology" only works for the InstancedMesh (square/cuboid) and sphere
    // geometries — hidden for hypercuboid/machine/custom, which don't produce those meshes.
    testTopologyBtnWrapper.classList.toggle('hidden', !['square', 'cuboid', 'sphere'].includes(type));

    switch(type) {
        case 'square':
            squareOptions.classList.remove('hidden');
            break;
        case 'cuboid':
            cuboidOptions.classList.remove('hidden');
            break;
        case 'sphere':
            sphereOptions.classList.remove('hidden');
            break;
        case 'hypercuboid':
            hypercuboidOptions.classList.remove('hidden');
            updateProjectionLevels();
            break;
        case 'machine':
            machineTopologyFields.classList.remove('hidden');
            break;
        case 'custom':
            customOptions.classList.remove('hidden');
            break;
    }
    
    checkMachineTopology();
}

geometryType.addEventListener('change', updateGeometryOptions);

// Default Node/Socket/CC-Numa dimensions (and Node Gap, where specified) per geometry type,
// applied to the shared machine-topology fields whenever that geometry's hybrid view becomes
// the active case — the fields themselves are reused across Square/Cuboid, so the appropriate
// preset is (re)applied on entry rather than baked into the static HTML.
const MACHINE_TOPOLOGY_DEFAULTS = {
    square: {
        node: { width: 12, height: 6 },
        socket: { width: 6, height: 6 },
        ccNuma: { width: 6, height: 3 },
    },
    cuboid: {
        node: { width: 12, height: 6, depth: 1 },
        socket: { width: 6, height: 6, depth: 1 },
        ccNuma: { width: 3, height: 6, depth: 1 },
        nodeGap: -0.2,
    },
};

function applyMachineTopologyDefaults(type) {
    const preset = MACHINE_TOPOLOGY_DEFAULTS[type];
    if (!preset) return;
    document.getElementById('nodeDimWidth').value = preset.node.width;
    document.getElementById('nodeDimHeight').value = preset.node.height;
    document.getElementById('socketDimWidth').value = preset.socket.width;
    document.getElementById('socketDimHeight').value = preset.socket.height;
    document.getElementById('ccNumaDimWidth').value = preset.ccNuma.width;
    document.getElementById('ccNumaDimHeight').value = preset.ccNuma.height;
    if (preset.node.depth !== undefined) {
        document.getElementById('nodeDimDepth').value = preset.node.depth;
        document.getElementById('socketDimDepth').value = preset.socket.depth;
        document.getElementById('ccNumaDimDepth').value = preset.ccNuma.depth;
    }
    if (preset.nodeGap !== undefined) {
        document.getElementById('nodeGapSlider').value = preset.nodeGap;
        document.getElementById('nodeGapNumber').value = preset.nodeGap;
    }
}

// Check if machine topology should be shown
function checkMachineTopology() {
    const squareHybrid = document.getElementById('squareHybridView').checked;
    const cuboidHybrid = document.getElementById('cuboidHybridView').checked;
    const hypercuboidHybrid = document.getElementById('hypercuboidHybridView').checked;
    const isMachine = geometryType.value === 'machine';

    if (squareHybrid || cuboidHybrid || hypercuboidHybrid || isMachine) {
        machineTopologyFields.classList.remove('hidden');
    } else {
        machineTopologyFields.classList.add('hidden');
    }

    if (squareHybrid && geometryType.value === 'square') {
        applyMachineTopologyDefaults('square');
    } else if (cuboidHybrid && geometryType.value === 'cuboid') {
        applyMachineTopologyDefaults('cuboid');
    }

    if ((geometryType.value === 'square' || geometryType.value === 'cuboid') &&
        (squareHybrid || cuboidHybrid)) {
        advancedOptionsContainer.classList.remove('hidden');
    } else {
        advancedOptionsContainer.classList.add('hidden');
    }

    // Depth only makes sense for Cuboid (3 axes) — Square only has Width/Height.
    const showDepth = geometryType.value === 'cuboid';
    ['nodeDimDepthGroup', 'socketDimDepthGroup', 'ccNumaDimDepthGroup'].forEach(id => {
        document.getElementById(id)?.classList.toggle('hidden', !showDepth);
    });
    document.querySelectorAll('.dim-axis-group').forEach(el => {
        el.classList.toggle('two-col', !showDepth);
    });
}

document.getElementById('squareHybridView').addEventListener('change', checkMachineTopology);
document.getElementById('cuboidHybridView').addEventListener('change', checkMachineTopology);
document.getElementById('hypercuboidHybridView').addEventListener('change', checkMachineTopology);

// Show the gap sliders only when "Extra spacing" is the selected Display Method.
const displayMethodSelect = document.getElementById('displayMethod');
const extraSpacingFields = document.getElementById('extraSpacingFields');

function updateExtraSpacingVisibility() {
    extraSpacingFields.classList.toggle('hidden', displayMethodSelect.value !== 'spacing');
}

displayMethodSelect.addEventListener('change', updateExtraSpacingVisibility);
updateExtraSpacingVisibility();

// Keep a range slider and its paired number input in sync, in both directions.
function linkSliderAndNumber(sliderEl, numberEl) {
    sliderEl.addEventListener('input', () => { numberEl.value = sliderEl.value; });
    numberEl.addEventListener('input', () => { sliderEl.value = numberEl.value; });
}

linkSliderAndNumber(document.getElementById('ccNumaGapSlider'), document.getElementById('ccNumaGapNumber'));
linkSliderAndNumber(document.getElementById('socketGapSlider'), document.getElementById('socketGapNumber'));
linkSliderAndNumber(document.getElementById('nodeGapSlider'), document.getElementById('nodeGapNumber'));
linkSliderAndNumber(document.getElementById('standardSpacingSlider'), document.getElementById('standardSpacing'));

// Advanced options toggle
document.getElementById('advancedToggle').addEventListener('click', () => {
    const advancedOptions = document.getElementById('advancedOptions');
    advancedOptions.classList.toggle('hidden');
});

// Square axis handling
const squareAxis1 = document.getElementById('squareAxis1');
const squareAxis2 = document.getElementById('squareAxis2');

function updateSquareAxes() {
    const allAxes = ['width', 'height'];
    const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
    const axis1Val = squareAxis1.value;
    squareAxis1.innerHTML = allAxes.map(a =>
        `<option value="${a}"${a === axis1Val ? ' selected' : ''}>${cap(a)}</option>`
    ).join('');
    const axis2Val = allAxes.find(a => a !== axis1Val);
    squareAxis2.innerHTML = `<option value="${axis2Val}">${cap(axis2Val)}</option>`;
}

squareAxis1.addEventListener('change', updateSquareAxes);

// Cuboid axis handling
const cuboidAxis1 = document.getElementById('cuboidAxis1');
const cuboidAxis2 = document.getElementById('cuboidAxis2');
const cuboidAxis3 = document.getElementById('cuboidAxis3');

function updateCuboidAxes() {
    const allAxes = ['height', 'width', 'depth'];
    const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

    // Axis 1: all three options
    const axis1Val = cuboidAxis1.value;
    cuboidAxis1.innerHTML = allAxes.map(a =>
        `<option value="${a}"${a === axis1Val ? ' selected' : ''}>${cap(a)}</option>`
    ).join('');

    // Axis 2: the two options not chosen by axis 1
    const axis2Options = allAxes.filter(a => a !== axis1Val);
    let axis2Val = cuboidAxis2.value;
    if (!axis2Options.includes(axis2Val)) {
        // Default preference: width, then depth
        axis2Val = axis2Options.find(a => a === 'width') || axis2Options[0];
    }
    cuboidAxis2.innerHTML = axis2Options.map(a =>
        `<option value="${a}"${a === axis2Val ? ' selected' : ''}>${cap(a)}</option>`
    ).join('');

    // Axis 3: the only remaining option
    const axis3Val = allAxes.find(a => a !== axis1Val && a !== axis2Val);
    cuboidAxis3.innerHTML = `<option value="${axis3Val}">${cap(axis3Val)}</option>`;
}

cuboidAxis1.addEventListener('change', updateCuboidAxes);
cuboidAxis2.addEventListener('change', updateCuboidAxes);
cuboidAxis3.addEventListener('change', updateCuboidAxes);

// Hypercuboid projection levels
const problemDimensions = document.getElementById('problemDimensions');
const outputDimensions = document.getElementById('outputDimensions');
const projectionLevels = document.getElementById('projectionLevels');

function updateProjectionLevels() {
    const probDim = parseInt(problemDimensions.value);
    const outDim = parseInt(outputDimensions.value);
    const numProjections = probDim - outDim;
    
    projectionLevels.innerHTML = '';
    
    for (let i = 0; i < numProjections; i++) {
        const projItem = document.createElement('div');
        projItem.className = 'projection-item';
        projItem.innerHTML = `
            <div>
                <label>Projection Dimension ${i + 1}</label>
                <input type="number" min="1" value="1">
            </div>
            <div>
                <label>Projection Level ${i + 1}</label>
                <input type="number" min="0" value="0">
            </div>
        `;
        projectionLevels.appendChild(projItem);
    }
}

problemDimensions.addEventListener('change', updateProjectionLevels);
outputDimensions.addEventListener('change', updateProjectionLevels);

document.getElementById('probDimDown').addEventListener('click', () => {
    if (problemDimensions.value > 1) {
        problemDimensions.value = parseInt(problemDimensions.value) - 1;
        updateProjectionLevels();
    }
});

document.getElementById('probDimUp').addEventListener('click', () => {
    problemDimensions.value = parseInt(problemDimensions.value) + 1;
    updateProjectionLevels();
});

// Dual view toggle
const dualView = document.getElementById('dualView');
const singleMetric = document.getElementById('singleMetric');
const dualMetrics = document.getElementById('dualMetrics');

dualView.addEventListener('change', (e) => {
    if (e.target.checked) {
        singleMetric.classList.add('hidden');
        dualMetrics.classList.remove('hidden');
    } else {
        singleMetric.classList.remove('hidden');
        dualMetrics.classList.add('hidden');
    }
});

// Metric type handling with complex color configurations
function createRGBAInput(label, defaultColor = 'rgba(255, 255, 255, 1)') {
    const pickrId = `pickr-el-${pickrIdCounter++}`;
    pickrLabelMap.set(pickrId, label);
    return `
        <div class="pickr-form-item">
            <label>${label}</label>
            <div class="pickr-container" id="${pickrId}" data-default="${defaultColor}"></div>
        </div>
    `;
}

function updateMetricColorOptions(selectId, containerId) {
    const select = document.getElementById(selectId);
    const container = document.getElementById(containerId);
    const metricType = select.value;

    destroyPickrIn(container);
    container.innerHTML = '';
    
    // Colored iterations - simple case
    if (metricType === 'coloredIterations') {
        container.innerHTML = `
            <div class="form-group">
                <label>Number of Colors</label>
                <input type="number" min="2" value="5" class="num-colors-input" data-container="${containerId}">
            </div>
            <div class="color-inputs-container"></div>
        `;
        updateIterationColors(containerId);
        container.querySelector('.num-colors-input').addEventListener('change', () => updateIterationColors(containerId));
        return;
    }
    
    // For gradient-capable metrics
    if (['coloredMPI', 'coloredOutMPI', 'heatmapSpeeders', 'compHeatmapSpeeders', 'mpiHeatmapSpeedLoop', 'mpiHeatmapSpeedRun', 'compHeatmapSpeedLoop', 'compHeatmapSpeedRun', 'heatmapSpeed', 'papiHeatmap', 'distinguish'].includes(metricType)) {
        // "Use color gradients" doesn't apply to any of the heatmap metrics — they always use
        // their own percentile/pace-driven coloring described in their own UI below.
        const heatmapMetrics = ['heatmapSpeeders', 'compHeatmapSpeeders', 'mpiHeatmapSpeedLoop', 'mpiHeatmapSpeedRun', 'compHeatmapSpeedLoop', 'compHeatmapSpeedRun', 'heatmapSpeed', 'papiHeatmap'];
        const showGradientCheckbox = !heatmapMetrics.includes(metricType);
        container.innerHTML = `
            ${showGradientCheckbox ? `
            <div class="checkbox-group">
                <label>
                    <input type="checkbox" class="use-gradient-check" data-container="${containerId}">
                    Use color gradients
                    <span class="tooltip-icon" data-tooltip="Enable smooth color transitions">?</span>
                </label>
            </div>` : ''}
            <div class="gradient-options"></div>
        `;

        const gradientCheck = container.querySelector('.use-gradient-check');
        if (gradientCheck) {
            gradientCheck.addEventListener('change', () => updateGradientOptions(metricType, containerId, gradientCheck.checked));
        }
        updateGradientOptions(metricType, containerId, false);
    }
}

const ITERATION_COLOR_DEFAULTS = [
    'rgba(255, 255, 255, 1)',
    'rgba(0, 127, 255, 1)',
    'rgba(0, 255, 255, 1)',
    'rgba(0, 0, 255, 1)',
    'rgba(255, 0, 255, 1)',
];

function updateIterationColors(containerId) {
    const container = document.getElementById(containerId);
    const numColors = parseInt(container.querySelector('.num-colors-input').value) || 5;
    const colorContainer = container.querySelector('.color-inputs-container');

    destroyPickrIn(colorContainer);
    let html = '';
    for (let i = 0; i < numColors; i++) {
        const def = ITERATION_COLOR_DEFAULTS[i] ?? 'rgba(255, 255, 255, 1)';
        html += createRGBAInput(`Color #${i + 1}`, def);
    }
    colorContainer.innerHTML = html;
    initPickrIn(colorContainer);
}

function updateGradientOptions(metricType, containerId, useGradient) {
    const container = document.getElementById(containerId);
    const optionsDiv = container.querySelector('.gradient-options');
    
    if (metricType === 'coloredMPI') {
        updateColoredMPIOptions(optionsDiv, useGradient);
    } else if (metricType === 'coloredOutMPI') {
        updateColoredOutMPIOptions(optionsDiv, useGradient);
    } else if (metricType === 'heatmapSpeeders') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'mpi'); // percentile-only, no Time-based scheme
    } else if (metricType === 'compHeatmapSpeeders') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'comp'); // same as heatmapSpeeders, mirrored for out-of-MPI (Comp) periods
    } else if (metricType === 'mpiHeatmapSpeedLoop') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'mpi'); // same UI as heatmapSpeeders — ranks by duration instead of enter/leave time
    } else if (metricType === 'mpiHeatmapSpeedRun') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'mpi', true); // same UI plus a target-region-name field; ranks only that region's calls, pooled across the whole run
    } else if (metricType === 'compHeatmapSpeedLoop') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'comp'); // same UI as compHeatmapSpeeders — ranks Comp periods by duration instead of enter/leave time
    } else if (metricType === 'compHeatmapSpeedRun') {
        updateHeatmapSpeedersOptions(optionsDiv, useGradient, true, 'comp'); // same UI as compHeatmapSpeeders — no region-name field needed since every Comp period is pooled, unfiltered
    } else if (metricType === 'heatmapSpeed') {
        updateHeatmapSpeedOptions(optionsDiv);
    } else if (metricType === 'papiHeatmap') {
        updatePapiOptions(optionsDiv);
    } else if (metricType === 'distinguish') {
        updateDistinguishOptions(optionsDiv, useGradient);
    }
}

function updateColoredMPIOptions(container, useGradient) {
    if (useGradient) {
        container.innerHTML = `
            <div class="form-group">
                <label>Gradient Application</label>
                <div class="radio-group">
                    <label><input type="radio" name="gradient-app-${Date.now()}" value="within" checked> within MPI times</label>
                    <label><input type="radio" name="gradient-app-${Date.now()}" value="between"> between MPI times</label>
                    <label><input type="radio" name="gradient-app-${Date.now()}" value="both"> Both</label>
                </div>
            </div>
            <div class="gradient-colors"></div>
        `;
        
        const radios = container.querySelectorAll('input[type="radio"]');
        const updateColors = () => {
            const selected = container.querySelector('input[type="radio"]:checked').value;
            const colorDiv = container.querySelector('.gradient-colors');
            destroyPickrIn(colorDiv);

            if (selected === 'within') {
                colorDiv.innerHTML = createRGBAInput('MPI begin', 'rgba(255, 0, 0, 1)') + createRGBAInput('MPI end', 'rgba(0, 255, 0, 1)') + createRGBAInput('Comp', 'rgba(0, 0, 255, 0)');
            } else if (selected === 'between') {
                colorDiv.innerHTML = createRGBAInput('MPI', 'rgba(255, 0, 0, 1)') + createRGBAInput('Comp begin', 'rgba(0, 0, 255, 0)') + createRGBAInput('Comp end', 'rgba(0, 0, 255, 0)');
            } else {
                colorDiv.innerHTML = createRGBAInput('MPI begin', 'rgba(255, 0, 0, 1)') + createRGBAInput('MPI end', 'rgba(0, 255, 0, 1)') + createRGBAInput('Comp begin', 'rgba(0, 0, 255, 0)') + createRGBAInput('Comp end', 'rgba(0, 0, 255, 0)');
            }
            initPickrIn(colorDiv);
        };

        radios.forEach(r => r.addEventListener('change', updateColors));
        updateColors();
    } else {
        container.innerHTML = `
            <div class="checkbox-group">
                <label>
                    <input type="checkbox" class="duration-coding-check">
                    Duration color-coding
                    <span class="tooltip-icon" data-tooltip="Color based on MPI call duration">?</span>
                </label>
            </div>
            <div class="duration-options"></div>
        `;
        
        const durationCheck = container.querySelector('.duration-coding-check');
        const updateDuration = () => {
            const durationDiv = container.querySelector('.duration-options');
            
            if (!durationCheck.checked) {
                destroyPickrIn(durationDiv);
                durationDiv.innerHTML = `
                    ${createRGBAInput('MPI', 'rgba(255, 0, 0, 1)')}
                    ${createRGBAInput('Comp', 'rgba(0, 0, 255, 0)')}
                `;
                initPickrIn(durationDiv);
            } else {
                destroyPickrIn(durationDiv);
                durationDiv.innerHTML = `
                    <div class="form-group">
                        <label>Number of cutoffs</label>
                        <div class="slider-with-number">
                            <input type="range" min="1" max="10" step="1" value="3" class="cutoffs-slider">
                            <input type="number" min="1" value="3" class="cutoffs-input">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Scheme</label>
                        <select class="scheme-select">
                            <option value="time">Time-based</option>
                            <option value="percentile">Percentile-based</option>
                        </select>
                    </div>
                    <div class="cutoff-colors"></div>
                `;

                const updateCutoffColors = () => {
                    const numCutoffs = parseInt(durationDiv.querySelector('.cutoffs-input').value) || 1;
                    const cutoffDiv = durationDiv.querySelector('.cutoff-colors');
                    destroyPickrIn(cutoffDiv);
                    let html = '';

                    for (let i = 0; i <= numCutoffs; i++) {
                        if (i === 0) {
                            html += createRGBAInput(`Color ${i + 1}`);
                        } else {
                            html += `
                                <div class="cutoff-item">
                                    ${createRGBAInput(`Color ${i + 1}`)}
                                    <div class="form-group">
                                        <label>Duration/Percentile</label>
                                        <input type="number" step="0.01" value="${i * 0.1}">
                                    </div>
                                </div>
                            `;
                        }
                    }
                    cutoffDiv.innerHTML = html;
                    initPickrIn(cutoffDiv);
                };

                linkSliderAndNumber(durationDiv.querySelector('.cutoffs-slider'), durationDiv.querySelector('.cutoffs-input'));
                durationDiv.querySelector('.cutoffs-input').addEventListener('change', updateCutoffColors);
                durationDiv.querySelector('.cutoffs-slider').addEventListener('change', updateCutoffColors);
                updateCutoffColors();
            }
        };
        
        durationCheck.addEventListener('change', updateDuration);
        updateDuration();
    }
}

function updateColoredOutMPIOptions(container, useGradient) {
    // Similar to coloredMPI but for out-of-MPI
    updateColoredMPIOptions(container, useGradient);
}

function updateHeatmapSpeedersOptions(container, useGradient, hideTimeScheme = false, mode = 'mpi', regionNameField = false) {
    destroyPickrIn(container);
    // 'mpi' mode: ranks the MPI calls themselves, with a flat "Comp" color for the gaps between them.
    // 'comp' mode (mirrored, for Comp Heatmap speeders/laggers): ranks the out-of-MPI (Comp) periods
    // instead, with a flat "MPI" color for the calls that now sit *between* the ranked periods.
    const flatColorLabel = mode === 'comp' ? 'MPI' : 'Comp';
    const betweenLabel = mode === 'comp' ? 'in-between Comp calls' : 'in-between MPI calls';
    const keepOptionLabel = mode === 'comp' ? 'Keep Comp color during MPI' : 'Keep MPI color during Comp';
    const baseHTML = `
        ${regionNameField ? `
        <div class="form-group">
            <label>MPI region name</label>
            <input type="text" value="MPI_Waitall" class="mpi-region-name-input">
            <span class="tooltip-icon" data-tooltip="Only calls to this exact function name are ranked; all other time counts as the gap">?</span>
        </div>` : ''}
        <div class="form-group">
            <label>Number of cutoffs</label>
            <div class="slider-with-number">
                <input type="range" min="1" max="10" step="1" value="3" class="cutoffs-slider">
                <input type="number" min="1" value="3" class="cutoffs-input">
            </div>
        </div>
        ${hideTimeScheme ? '' : `
        <div class="form-group">
            <label>Scheme</label>
            <select class="scheme-select">
                <option value="time">Time-based</option>
                <option value="percentile">Percentile-based</option>
            </select>
        </div>`}
        <div class="comp-color-group">${createRGBAInput(flatColorLabel)}</div>
        <div class="form-group">
            <label>${betweenLabel}</label>
            <select class="between-mpi-select">
                <option value="custom">Custom color</option>
                <option value="keep" selected>${keepOptionLabel}</option>
                ${!useGradient ? '<option value="gradient">Gradient change</option>' : ''}
            </select>
        </div>
        ${useGradient ? `
            <div class="form-group">
                <label>Gradient type</label>
                <select class="gradient-type-select">
                    <option value="prev-curr">from previous to current</option>
                    <option value="curr-next">from current to next</option>
                </select>
            </div>
        ` : ''}
        <div class="cutoff-colors"></div>
    `;

    container.innerHTML = baseHTML;
    initPickrIn(container);

    const updateCutoffColors = () => {
        const numCutoffs = parseInt(container.querySelector('.cutoffs-input').value) || 1;
        const cutoffDiv = container.querySelector('.cutoff-colors');
        destroyPickrIn(cutoffDiv);
        let html = '';

        for (let i = 0; i <= numCutoffs; i++) {
            if (i === 0) {
                html += createRGBAInput(`Color ${i + 1}`);
            } else {
                const defaultValue = hideTimeScheme
                    ? Math.round(100 * (numCutoffs - i + 1) / (numCutoffs + 1))
                    : i * 0.1;
                html += `
                    <div class="cutoff-item">
                        ${createRGBAInput(`Color ${i + 1}`)}
                        <div class="form-group">
                            <label>${hideTimeScheme ? 'Percentile' : 'Duration/Percentile'}</label>
                            <input type="number" step="0.01" value="${defaultValue}" class="cutoff-percentile-input">
                        </div>
                    </div>
                `;
            }
        }
        cutoffDiv.innerHTML = html;
        initPickrIn(cutoffDiv);
    };

    const betweenSelect = container.querySelector('.between-mpi-select');
    const updateCompGroupVisibility = () => {
        const compGroup = container.querySelector('.comp-color-group');
        compGroup.style.display = betweenSelect.value === 'custom' ? 'block' : 'none';
    };
    betweenSelect.addEventListener('change', updateCompGroupVisibility);
    updateCompGroupVisibility();

    linkSliderAndNumber(container.querySelector('.cutoffs-slider'), container.querySelector('.cutoffs-input'));
    container.querySelector('.cutoffs-input').addEventListener('change', updateCutoffColors);
    container.querySelector('.cutoffs-slider').addEventListener('change', updateCutoffColors);
    updateCutoffColors();
}

function updateHeatmapSpeedOptions(container) {
    // Live, per-frame variant: ranks each rank's recent comp-time pace (over a trailing
    // window) against its peers right now — no Scheme/in-between/flat-color concepts
    // apply here, since every instant already has a well-defined pace value.
    destroyPickrIn(container);
    container.innerHTML = `
        <div class="form-group">
            <label>Number of cutoffs</label>
            <div class="slider-with-number">
                <input type="range" min="1" max="10" step="1" value="3" class="cutoffs-slider">
                <input type="number" min="1" value="3" class="cutoffs-input">
            </div>
        </div>
        <div class="form-group">
            <label>Window Size (µs)</label>
            <input type="number" min="1" value="1000" class="window-size-input">
        </div>
        <div class="cutoff-colors"></div>
    `;
    initPickrIn(container);

    const updateCutoffColors = () => {
        const numCutoffs = parseInt(container.querySelector('.cutoffs-input').value) || 1;
        const cutoffDiv = container.querySelector('.cutoff-colors');
        destroyPickrIn(cutoffDiv);
        let html = '';

        for (let i = 0; i <= numCutoffs; i++) {
            if (i === 0) {
                html += createRGBAInput(`Color ${i + 1}`);
            } else {
                const defaultValue = Math.round(100 * (numCutoffs - i + 1) / (numCutoffs + 1));
                html += `
                    <div class="cutoff-item">
                        ${createRGBAInput(`Color ${i + 1}`)}
                        <div class="form-group">
                            <label>Percentile</label>
                            <input type="number" step="0.01" value="${defaultValue}" class="cutoff-percentile-input">
                        </div>
                    </div>
                `;
            }
        }
        cutoffDiv.innerHTML = html;
        initPickrIn(cutoffDiv);
    };

    linkSliderAndNumber(container.querySelector('.cutoffs-slider'), container.querySelector('.cutoffs-input'));
    container.querySelector('.cutoffs-input').addEventListener('change', updateCutoffColors);
    container.querySelector('.cutoffs-slider').addEventListener('change', updateCutoffColors);
    updateCutoffColors();
}

// Sequential blue ramp for the PAPI percentile buckets, light -> dark, from the validated
// reference palette. A heatmap whose buckets all default to white shows nothing until the
// user colours it by hand, so this metric ships with a ramp already in place.
//
// Light = high percentile: the scene background is black, so the ranks that stand out are the
// ones at the top of the distribution. Steps are interpolated along the ramp rather than
// snapped to its named entries, which keeps the lightness gaps even.
//
// The ramp spans a total lightness of ~0.47, so it yields visually separable steps up to
// **8 buckets** (7 cutoffs); past that the defaults crowd together and want recolouring by
// hand. Verified with the palette validator in dark/ordinal mode.
const PAPI_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
                   '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95'];

function papiRampColor(index, count) {
    if (count <= 1) return 'rgba(205, 226, 251, 1)';
    const f = index * (PAPI_RAMP.length - 1) / (count - 1);
    const i = Math.min(PAPI_RAMP.length - 1, Math.floor(f));
    const j = Math.min(PAPI_RAMP.length - 1, i + 1);
    const t = f - i;
    const part = (h, k) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
    const mix = k => Math.round(part(PAPI_RAMP[i], k) + (part(PAPI_RAMP[j], k) - part(PAPI_RAMP[i], k)) * t);
    return `rgba(${mix(0)}, ${mix(1)}, ${mix(2)}, 1)`;
}

// Hardware-counter metric. One formula picked from a list, ranked across ranks every frame,
// so the cutoff/colour UI is the same percentile machinery as the other heatmaps. The extra
// "No value" swatch is not decoration: a rank has no counter reading at the edges of the
// trace or when the formula's denominator did not advance, and that must be visibly distinct
// from a genuine low reading rather than guessed at.
function updatePapiOptions(container) {
    destroyPickrIn(container);
    container.innerHTML = `
        <div class="form-group">
            <label>Counter metric</label>
            <select class="papi-formula-select">
                <option value="ipc" selected>IPC — instructions per cycle</option>
                <option value="clock">Effective clock (GHz) — cycles retired per unit time</option>
                <option value="flops">FLOP rate (GFLOP/s)</option>
                <option value="vec">Vector lane use — ops per vector instruction</option>
            </select>
        </div>
        <div class="form-group">
            <label>Number of cutoffs</label>
            <div class="slider-with-number">
                <input type="range" min="1" max="10" step="1" value="3" class="cutoffs-slider">
                <input type="number" min="1" value="3" class="cutoffs-input">
            </div>
        </div>
        <div class="cutoff-colors"></div>
        <div class="comp-color-group">
            ${createRGBAInput('No value', 'rgba(32, 32, 32, 0.25)')}
        </div>
    `;
    initPickrIn(container);

    const updateCutoffColors = () => {
        const numCutoffs = parseInt(container.querySelector('.cutoffs-input').value) || 1;
        const cutoffDiv = container.querySelector('.cutoff-colors');
        destroyPickrIn(cutoffDiv);
        let html = '';
        const buckets = numCutoffs + 1;
        for (let i = 0; i <= numCutoffs; i++) {
            const swatch = createRGBAInput(`Color ${i + 1}`, papiRampColor(i, buckets));
            if (i === 0) {
                html += swatch;
            } else {
                const defaultValue = Math.round(100 * (numCutoffs - i + 1) / (numCutoffs + 1));
                html += `
                    <div class="cutoff-item">
                        ${swatch}
                        <div class="form-group">
                            <label>Percentile</label>
                            <input type="number" step="0.01" value="${defaultValue}" class="cutoff-percentile-input">
                        </div>
                    </div>
                `;
            }
        }
        cutoffDiv.innerHTML = html;
        initPickrIn(cutoffDiv);
    };

    linkSliderAndNumber(container.querySelector('.cutoffs-slider'), container.querySelector('.cutoffs-input'));
    container.querySelector('.cutoffs-input').addEventListener('change', updateCutoffColors);
    container.querySelector('.cutoffs-slider').addEventListener('change', updateCutoffColors);
    updateCutoffColors();
}

function updateDistinguishOptions(container, useGradient) {
    destroyPickrIn(container);
    if (useGradient) {
        container.innerHTML = `
            ${createRGBAInput('Sends begin')}
            ${createRGBAInput('Sends end')}
            ${createRGBAInput('Receives begin')}
            ${createRGBAInput('Receives end')}
            ${createRGBAInput('Waitalls begin')}
            ${createRGBAInput('Waitalls end')}
            ${createRGBAInput('Other MPI calls begin')}
            ${createRGBAInput('Other MPI calls end')}
            <div class="comp-color-group">
                ${createRGBAInput('Comp begin')}
                ${createRGBAInput('Comp end')}
            </div>
            <div class="form-group">
                <label>in-between MPI calls</label>
                <select class="between-mpi-select">
                    <option value="custom">Custom color</option>
                    <option value="keep" selected>Keep MPI color during Comp</option>
                    <option value="gradient">Gradient change</option>
                </select>
            </div>
        `;
        initPickrIn(container);

        const betweenSelect = container.querySelector('.between-mpi-select');
        const updateCompGroupVisibility = () => {
            const compGroup = container.querySelector('.comp-color-group');
            compGroup.style.display = betweenSelect.value === 'custom' ? 'block' : 'none';
        };
        betweenSelect.addEventListener('change', updateCompGroupVisibility);
        updateCompGroupVisibility();
    } else {
        container.innerHTML = `
            ${createRGBAInput('Sends')}
            ${createRGBAInput('Receives')}
            ${createRGBAInput('Waitalls')}
            ${createRGBAInput('Other MPI calls')}
            <div class="comp-color-group">${createRGBAInput('Comp')}</div>
            <div class="form-group">
                <label>in-between MPI calls</label>
                <select class="between-mpi-select">
                    <option value="custom">Custom color</option>
                    <option value="keep" selected>Keep MPI color during Comp</option>
                    <option value="gradient">Gradient change</option>
                </select>
            </div>
        `;
        initPickrIn(container);

        const betweenSelect = container.querySelector('.between-mpi-select');
        const updateCompGroupVisibility = () => {
            const compGroup = container.querySelector('.comp-color-group');
            compGroup.style.display = betweenSelect.value === 'custom' ? 'block' : 'none';
        };
        betweenSelect.addEventListener('change', updateCompGroupVisibility);
        updateCompGroupVisibility();
    }
}

// Initialize metric options for all three panels
['metricType1', 'metricType2', 'metricType3'].forEach((id, index) => {
    const containerId = `colorOptions${index + 1}`;
    document.getElementById(id).addEventListener('change', () => {
        updateMetricColorOptions(id, containerId);
    });
    updateMetricColorOptions(id, containerId);
});

// Extract color picker values from a color options container, keyed by their label text
function extractPickrColors(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return {};
    const colors = {};
    pickrInstances.forEach((instance, id) => {
        const root = instance.getRoot().root;
        if (!container.contains(root)) return;
        const label = pickrLabelMap.get(id);
        if (!label) return;
        const rgba = instance.getColor()?.toRGBA();
        if (rgba) {
            colors[label] = {
                r: rgba[0] / 255,
                g: rgba[1] / 255,
                b: rgba[2] / 255,
                a: rgba[3]
            };
        }
    });
    return colors;
}

// Visualize button
const visualizeButton = document.getElementById('visualizeButton');
export let config;
visualizeButton.addEventListener('click', () => {
    // Collect all configuration data
    const getNum = (id) => {
        const raw = document.getElementById(id)?.value;
        const n = parseFloat(raw);
        return Number.isFinite(n) ? n : undefined;
    };
    config = {
        dataSource: document.querySelector('input[name="dataSource"]:checked').value,
        serverFile: serverFileSelect.value,
        localFileData: uploadedFileData,
        localFile: uploadedFile,
        geometry: {
            hideViewCores: document.getElementById('hideViewCores')?.checked,
            type: document.getElementById('geometryType')?.value,
            // Square options
            square: {
                axis1: document.getElementById('squareAxis1')?.value,
                axis2: document.getElementById('squareAxis2')?.value,
                dim1: document.getElementById('squareDim1')?.value,
                dim2: document.getElementById('squareDim2')?.value,
                hybridView: document.getElementById('squareHybridView')?.checked,
            },
            // Cuboid options
            cuboid: {
                axis1: document.getElementById('cuboidAxis1')?.value,
                axis2: document.getElementById('cuboidAxis2')?.value,
                axis3: document.getElementById('cuboidAxis3')?.value,
                dim1: document.getElementById('cuboidDim1')?.value,
                dim2: document.getElementById('cuboidDim2')?.value,
                dim3: document.getElementById('cuboidDim3')?.value,
                hybridView: document.getElementById('cuboidHybridView')?.checked,
            },
            // Sphere options
            sphere: {
                faces: document.getElementById('sphereFaces')?.value,
                doubleSide: document.getElementById('sphereDoubleSide')?.checked,
            },
            // Custom geometry options
            custom: {
                doubleSide: document.getElementById('customDoubleSide')?.checked,
            },
            // Hypercuboid options
            hypercuboid: {
                problemDimensions: document.getElementById('problemDimensions')?.value,
                outputDimensions: document.getElementById('outputDimensions')?.value,
                hybridView: document.getElementById('hypercuboidHybridView')?.checked,
            },
            // Machine topology
            machineTopology: {
                nodeDimensions: {
                    width: getNum('nodeDimWidth'),
                    height: getNum('nodeDimHeight'),
                    depth: getNum('nodeDimDepth'),
                },
                socketDimensions: {
                    width: getNum('socketDimWidth'),
                    height: getNum('socketDimHeight'),
                    depth: getNum('socketDimDepth'),
                },
                ccNumaDomainDimensions: {
                    width: getNum('ccNumaDimWidth'),
                    height: getNum('ccNumaDimHeight'),
                    depth: getNum('ccNumaDimDepth'),
                },
                extraSpacing: {
                    ccNuma: getNum('ccNumaGapNumber'),
                    socket: getNum('socketGapNumber'),
                    node: getNum('nodeGapNumber'),
                },
                displayMethod: document.getElementById('displayMethod')?.value,
                hideView: document.getElementById('hideViewMachineTopology')?.checked,
            },
            // Advanced options
            advanced: {
                rankShape: document.getElementById('rankShape')?.value,
                standardSpacing: document.getElementById('standardSpacing')?.value,
            },
        },
        metric: {
            type1: document.getElementById('metricType1')?.value,
            type2: document.getElementById('metricType2')?.value,
            type3: document.getElementById('metricType3')?.value,
            colors1: extractPickrColors('colorOptions1'),
            colors2: extractPickrColors('colorOptions2'),
            colors3: extractPickrColors('colorOptions3'),
            useGradient1: document.querySelector('#colorOptions1 .use-gradient-check')?.checked || false,
            useGradient2: document.querySelector('#colorOptions2 .use-gradient-check')?.checked || false,
            useGradient3: document.querySelector('#colorOptions3 .use-gradient-check')?.checked || false,
            gradientApp1: document.querySelector('#colorOptions1 input[type="radio"]:checked')?.value,
            gradientApp2: document.querySelector('#colorOptions2 input[type="radio"]:checked')?.value,
            gradientApp3: document.querySelector('#colorOptions3 input[type="radio"]:checked')?.value,
            heatmapCutoffs1: Array.from(document.querySelectorAll('#colorOptions1 .cutoff-percentile-input')).map(el => el.value),
            heatmapCutoffs2: Array.from(document.querySelectorAll('#colorOptions2 .cutoff-percentile-input')).map(el => el.value),
            heatmapCutoffs3: Array.from(document.querySelectorAll('#colorOptions3 .cutoff-percentile-input')).map(el => el.value),
            betweenMpi1: document.querySelector('#colorOptions1 .between-mpi-select')?.value,
            betweenMpi2: document.querySelector('#colorOptions2 .between-mpi-select')?.value,
            betweenMpi3: document.querySelector('#colorOptions3 .between-mpi-select')?.value,
            windowSize1: document.querySelector('#colorOptions1 .window-size-input')?.value,
            windowSize2: document.querySelector('#colorOptions2 .window-size-input')?.value,
            windowSize3: document.querySelector('#colorOptions3 .window-size-input')?.value,
            runTargetFn1: document.querySelector('#colorOptions1 .mpi-region-name-input')?.value,
            runTargetFn2: document.querySelector('#colorOptions2 .mpi-region-name-input')?.value,
            runTargetFn3: document.querySelector('#colorOptions3 .mpi-region-name-input')?.value,
            papiFormula1: document.querySelector('#colorOptions1 .papi-formula-select')?.value,
            papiFormula2: document.querySelector('#colorOptions2 .papi-formula-select')?.value,
            papiFormula3: document.querySelector('#colorOptions3 .papi-formula-select')?.value,
            dualView: document.getElementById('dualView').checked,
            includeTraditional: document.getElementById('includeTraditional').checked,
            showLegend: document.getElementById('showLegend').checked,
            excludeCollectives: document.getElementById('excludeCollectives').checked,
            excludeP2P: document.getElementById('excludeP2P').checked,
        },
        controls: {
            timeGranularity: document.getElementById('timeGranularity').value,
            playbackSpeed: document.getElementById('playbackSpeed').value,
            beginTime: document.getElementById('beginTime').value,
            endTime: document.getElementById('endTime').value,
            activateTimejumps: document.getElementById('activateTimejumps').checked,
            ignoreMpiInit: document.getElementById('ignoreMpiInit').checked,
        },
        hoverInfoOptions: {
            rankName: document.querySelector('.hover-options label:nth-child(1) input')?.checked,
            nodeName: document.querySelector('.hover-options label:nth-child(2) input')?.checked,
            ccNumaDomain: document.querySelector('.hover-options label:nth-child(3) input')?.checked,
            regionName: document.querySelector('.hover-options label:nth-child(4) input')?.checked,
            startTime: document.querySelector('.hover-options label:nth-child(5) input')?.checked,
            endTime: document.querySelector('.hover-options label:nth-child(6) input')?.checked,
            durationTime: document.querySelector('.hover-options label:nth-child(7) input')?.checked,
            startingTimestamp: document.querySelector('.hover-options label:nth-child(8) input')?.checked,
            endingTimestamp: document.querySelector('.hover-options label:nth-child(9) input')?.checked,
            cycleDurationCount: document.querySelector('.hover-options label:nth-child(10) input')?.checked,
            iterationNumber: document.querySelector('.hover-options label:nth-child(11) input')?.checked,
            transferredSize: document.querySelector('.hover-options label:nth-child(12) input')?.checked,
            messageNumber: document.querySelector('.hover-options label:nth-child(13) input')?.checked,
        }
    };
    
    // Config is now read by viz.js's handler on the same button click
});

// Add hover effect to visualize button
visualizeButton.addEventListener('mouseenter', () => {
    visualizeButton.style.background = '#218838';
});

visualizeButton.addEventListener('mouseleave', () => {
    visualizeButton.style.background = '#28a745';
});

// Traditional view toggle
document.getElementById('includeTraditional').addEventListener('change', (e) => {
    const traditionalOptions = document.getElementById('traditionalOptions');
    if (e.target.checked) {
        traditionalOptions.classList.remove('hidden');
    } else {
        traditionalOptions.classList.add('hidden');
    }
});

document.getElementById('activateTimejumps').addEventListener('change', (e) => {
    document.getElementById('timejumpThresholdGroup').style.display = e.target.checked ? 'block' : 'none';
});

// Initialize
updateProjectionLevels();

// Initialize traditional view Pickr
initPickrIn(document.getElementById('traditionalOptions'));
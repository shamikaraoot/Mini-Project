// ─────────────────────────────────────────────────────────────────────────────
//  NeuroEmotion AI — Full Frontend Logic
// ─────────────────────────────────────────────────────────────────────────────

// ── Global State ─────────────────────────────────────────────────────────────
const state = {
    subjects: [],
    labels: [],
    trialData: null,
    currentTime: 0,
    activeBand: 'alpha',
    showChannels: true,
    channelsMap: {},
    windowSize: '1s',        // '1s' or '4s'
    animationId: null,
    probChart: null,
};

// ── Canvas Setup ──────────────────────────────────────────────────────────────
const topoCanvas      = document.getElementById('topomap-canvas');
const ctx             = topoCanvas.getContext('2d');
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width  = 90;
offscreenCanvas.height = 90;
const offscreenCtx    = offscreenCanvas.getContext('2d');

// ── DOM References ────────────────────────────────────────────────────────────
const subjectSelect    = document.getElementById('subject-select');
const sessionSelect    = document.getElementById('session-select');
const trialSelect      = document.getElementById('trial-select');
const splitSelect      = document.getElementById('split-select');
const featureSelect    = document.getElementById('feature-select');
const classifierSelect = document.getElementById('classifier-select');
const runBtn           = document.getElementById('run-btn');
const loadingOverlay   = document.getElementById('loading-overlay');
const loaderTitle      = document.getElementById('loader-title');
const welcomePanel     = document.getElementById('welcome-panel');
const dashboardPanel   = document.getElementById('dashboard-panel');
const subHeader        = document.getElementById('sub-header');
const scrubber         = document.getElementById('timeline-scrubber');
const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('file-input');

// ── Emotion Color Map ─────────────────────────────────────────────────────────
const EMOTION_COLORS = {
    'Positive': '#4ade80',
    'Neutral':  '#facc15',
    'Negative': '#f87171',
};

const EMOTION_LABEL = { 1: 'Positive', 0: 'Neutral', '-1': 'Negative' };

// ── Trial Labels (SEED ground truth order) ───────────────────────────────────
const TRIAL_EMOTIONS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1];
const TRIAL_NAMES    = TRIAL_EMOTIONS.map(e =>
    e === 1 ? 'Positive' : e === 0 ? 'Neutral' : 'Negative');

// ── Colormap (Blue → Purple → Orange) ────────────────────────────────────────
function getInterpolatedColor(ratio) {
    const stops = [
        { r: 0,   g: 0,   b: 180 },
        { r: 100, g: 0,   b: 180 },
        { r: 180, g: 0,   b: 120 },
        { r: 230, g: 100, b: 0   },
        { r: 255, g: 200, b: 0   },
    ];
    const scaled = ratio * (stops.length - 1);
    const i      = Math.min(Math.floor(scaled), stops.length - 2);
    const t      = scaled - i;
    return {
        r: Math.round(stops[i].r + t * (stops[i+1].r - stops[i].r)),
        g: Math.round(stops[i].g + t * (stops[i+1].g - stops[i].g)),
        b: Math.round(stops[i].b + t * (stops[i+1].b - stops[i].b)),
    };
}

// ── Build Channel Position Map from SEED 9×9 grid ────────────────────────────
function buildChannelsMap(channelList, locationGrid) {
    const map  = {};
    const rows = locationGrid.length;
    const cols = locationGrid[0].length;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const label = locationGrid[r][c];
            if (label === '-') continue;
            const idx = channelList.indexOf(label);
            if (idx === -1) continue;
            map[label] = {
                x: (c / (cols - 1)) * 360,
                y: (r / (rows - 1)) * 360,
                index: idx,
            };
        }
    }
    return map;
}

// ── Draw Scalp Outline ────────────────────────────────────────────────────────
function drawScalpOutline() {
    const cx = 180, cy = 180, r = 145;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Nose indicator
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy - r + 4);
    ctx.lineTo(cx, cy - r - 14);
    ctx.lineTo(cx + 12, cy - r + 4);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth   = 2;
    ctx.stroke();
}

// ── Draw Electrode Dots & Labels ─────────────────────────────────────────────
function drawElectrodeSensors(rawValues) {
    for (const label in state.channelsMap) {
        const ele = state.channelsMap[label];
        ctx.beginPath();
        ctx.arc(ele.x, ele.y, 4, 0, Math.PI * 2);
        ctx.fillStyle   = 'rgba(255,255,255,0.85)';
        ctx.strokeStyle = 'rgba(0,0,0,0.4)';
        ctx.lineWidth   = 1;
        ctx.fill();
        ctx.stroke();

        // Electrode Label — major 10-20 positions only
        ctx.fillStyle  = '#ffffff';
        ctx.font       = '9px Inter';
        ctx.textAlign  = 'center';

        if (
            label.startsWith('Fp') ||
            label.startsWith('F')  ||
            label.startsWith('C')  ||
            label.startsWith('P')  ||
            label.startsWith('O')
        ) {
            ctx.fillText(label, ele.x, ele.y - 8);
        }
    }
}

// ── Topomap Renderer (Shepard IDW) ────────────────────────────────────────────
function renderTopomap() {
    if (!state.trialData) return;

    const t          = state.currentTime;
    const band       = state.activeBand;
    const rawValues  = state.trialData.features_by_band[band][t];
    const bandMatrix = state.trialData.features_by_band[band];

    let globalMin = Infinity, globalMax = -Infinity;
    for (let i = 0; i < bandMatrix.length; i++)
        for (let j = 0; j < bandMatrix[i].length; j++) {
            if (bandMatrix[i][j] < globalMin) globalMin = bandMatrix[i][j];
            if (bandMatrix[i][j] > globalMax) globalMax = bandMatrix[i][j];
        }
    const globalRange = globalMax - globalMin || 1;

    const offW = offscreenCanvas.width;
    const offH = offscreenCanvas.height;
    const imgData = offscreenCtx.createImageData(offW, offH);
    const data    = imgData.data;

    const factor        = 90 / 360;
    const headRadiusLow = 145 * factor;
    const headCenterLow = 180 * factor;
    const p = 2;

    const lowResEles = [];
    for (const label in state.channelsMap) {
        const ele = state.channelsMap[label];
        lowResEles.push({ x: ele.x * factor, y: ele.y * factor, val: rawValues[ele.index] });
    }

    for (let y = 0; y < offH; y++) {
        for (let x = 0; x < offW; x++) {
            const pixelIdx     = (y * offW + x) * 4;
            const dx           = x - headCenterLow, dy = y - headCenterLow;
            const distToCenter = Math.sqrt(dx*dx + dy*dy);

            if (distToCenter > headRadiusLow) { data[pixelIdx + 3] = 0; continue; }

            let numerator = 0, denominator = 0, exactMatch = false, exactVal = 0;
            for (let i = 0; i < lowResEles.length; i++) {
                const ele  = lowResEles[i];
                const dX   = x - ele.x, dY = y - ele.y;
                const dist = Math.sqrt(dX*dX + dY*dY);
                if (dist < 0.25) { exactMatch = true; exactVal = ele.val; break; }
                const w    = 1 / Math.pow(dist, p);
                numerator   += w * ele.val;
                denominator += w;
            }

            const finalVal = exactMatch ? exactVal : numerator / denominator;
            const ratio    = (finalVal - globalMin) / globalRange;
            const color    = getInterpolatedColor(ratio);

            data[pixelIdx]     = color.r;
            data[pixelIdx + 1] = color.g;
            data[pixelIdx + 2] = color.b;
            data[pixelIdx + 3] = distToCenter > headRadiusLow - 4
                ? Math.round(255 * (headRadiusLow - distToCenter) / 4)
                : 255;
        }
    }

    offscreenCtx.putImageData(imgData, 0, 0);
    ctx.clearRect(0, 0, 360, 360);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreenCanvas, 0, 0, 90, 90, 0, 0, 360, 360);
    drawScalpOutline();
    if (state.showChannels) drawElectrodeSensors(rawValues);
}

// ── Probability Chart ─────────────────────────────────────────────────────────
function initProbChart() {
    const canvas = document.getElementById('probability-chart');
    if (!canvas) return;
    if (state.probChart) { state.probChart.destroy(); state.probChart = null; }

    state.probChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Negative', data: [], borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.15)', fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Neutral',  data: [], borderColor: '#facc15', backgroundColor: 'rgba(250,204,21,0.15)',  fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Positive', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.15)', fill: true, tension: 0.3, pointRadius: 0 },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 0, max: 1, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { labels: { color: '#e2e8f0' } } }
        }
    });
}

function updateProbChart(probs) {
    if (!state.probChart || !probs) return;
    const T      = probs.length;
    const labels = Array.from({ length: T }, (_, i) => `${i}s`);

    state.probChart.data.labels           = labels;
    state.probChart.data.datasets[0].data = probs.map(p => p[0]); // Negative
    state.probChart.data.datasets[1].data = probs.map(p => p[1]); // Neutral
    state.probChart.data.datasets[2].data = probs.map(p => p[2]); // Positive
    state.probChart.update();
}

// ── Metrics: Precision & AUC-ROC ─────────────────────────────────────────────
function computeMetrics(predictions, probabilities, groundTruth) {
    const classes  = [-1, 0, 1];
    const classIdx = { '-1': 0, '0': 1, '1': 2 };

    // Precision (macro)
    let precisionSum = 0, precisionCount = 0;
    for (const cls of classes) {
        let TP = 0, FP = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === cls && predictions[i] === groundTruth) TP++;
            else if (predictions[i] === cls && predictions[i] !== groundTruth) FP++;
        }
        if (TP + FP > 0) { precisionSum += TP / (TP + FP); precisionCount++; }
    }
    const precision = precisionCount > 0 ? precisionSum / precisionCount : 0;

    // AUC-ROC (macro one-vs-rest)
    function trapezoidalAUC(scores, binaryLabels) {
        const paired = scores.map((s, i) => ({ score: s, label: binaryLabels[i] }));
        paired.sort((a, b) => b.score - a.score);
        let auc = 0, tp = 0, fp = 0, prevTp = 0, prevFp = 0;
        const totalPos = binaryLabels.filter(l => l === 1).length;
        const totalNeg = binaryLabels.length - totalPos;
        if (totalPos === 0 || totalNeg === 0) return 0.5;
        for (const p of paired) {
            if (p.label === 1) tp++; else fp++;
            auc   += (fp - prevFp) * (tp + prevTp) / 2;
            prevTp = tp; prevFp = fp;
        }
        return auc / (totalPos * totalNeg);
    }

    let aucSum = 0;
    for (const cls of classes) {
        const scores       = probabilities.map(p => p[classIdx[String(cls)]]);
        const binaryLabels = Array(predictions.length).fill(groundTruth === cls ? 1 : 0);
        aucSum += trapezoidalAUC(scores, binaryLabels);
    }
    const auc = aucSum / classes.length;

    return { precision, auc };
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function showLoading(msg) {
    loaderTitle.textContent = msg || 'Training Classifier...';
    loadingOverlay.classList.remove('hidden');
}
function hideLoading() { loadingOverlay.classList.add('hidden'); }

function setEmotionCard(cardId, valId, emotionName) {
    const card = document.getElementById(cardId);
    const val  = document.getElementById(valId);
    if (!card || !val) return;
    val.textContent = emotionName;
    card.className  = 'card card-metric';
    if (emotionName === 'Positive')      card.classList.add('emotion-positive');
    else if (emotionName === 'Negative') card.classList.add('emotion-negative');
    else                                 card.classList.add('emotion-neutral');
}

function populateTrialSelect(labels) {
    trialSelect.innerHTML = '';
    labels.forEach((lbl, i) => {
        const name      = lbl === 1 ? 'Positive' : lbl === 0 ? 'Neutral' : 'Negative';
        const opt       = document.createElement('option');
        opt.value       = i + 1;
        opt.textContent = `Trial ${i + 1} (${name})`;
        trialSelect.appendChild(opt);
    });
}

// ── Load Subjects from API ────────────────────────────────────────────────────
async function loadSubjects() {
    subjectSelect.innerHTML = '<option value="" disabled selected>Loading...</option>';
    sessionSelect.innerHTML = '<option value="" disabled selected>Select subject first</option>';
    sessionSelect.disabled  = true;

    try {
        const res  = await fetch(`/api/status?window_size=${state.windowSize}`);
        const data = await res.json();
        if (!data.success) throw new Error(data.error);

        state.subjects = data.subjects;
        state.labels   = data.labels;

        subjectSelect.innerHTML = '<option value="" disabled selected>Select subject</option>';
        data.subjects.forEach(s => {
            const opt       = document.createElement('option');
            opt.value       = s.subject_id;
            opt.textContent = s.name;
            subjectSelect.appendChild(opt);
        });

        populateTrialSelect(data.labels);

    } catch (err) {
        subjectSelect.innerHTML = `<option disabled selected>Error: ${err.message}</option>`;
        console.error('loadSubjects error:', err);
    }
}

// ── Subject Change → populate Sessions ───────────────────────────────────────
subjectSelect.addEventListener('change', () => {
    const subId   = parseInt(subjectSelect.value);
    const subject = state.subjects.find(s => s.subject_id === subId);
    sessionSelect.innerHTML = '';
    sessionSelect.disabled  = false;

    if (subject) {
        subject.sessions.forEach((sess, idx) => {
            const opt       = document.createElement('option');
            opt.value       = idx;
            opt.textContent = `Session ${sess.session_id} (${sess.date})`;
            sessionSelect.appendChild(opt);
        });
    }
});

// ── Band Tab Clicks ───────────────────────────────────────────────────────────
document.querySelectorAll('.band-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.band-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeBand = btn.dataset.band;
        renderTopomap();

        // highlight matching row in band table
        document.querySelectorAll('.band-ref-table tbody tr').forEach(row => {
            row.querySelectorAll('td').forEach(td => td.style.opacity = '0.6');
        });
        const activeRow = document.querySelector(`.band-row-${state.activeBand}`);
        if (activeRow) activeRow.querySelectorAll('td').forEach(td => td.style.opacity = '1');
    });
});

// ── Timeline Scrubber ─────────────────────────────────────────────────────────
scrubber.addEventListener('input', () => {
    state.currentTime = parseInt(scrubber.value);
    renderTopomap();
});

// ── Run Classification ────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
    const subjectId  = subjectSelect.value;
    const sessionIdx = sessionSelect.value;
    const trialId    = trialSelect.value;

    if (!subjectId || sessionIdx === '' || !trialId) {
        alert('Please select Subject, Session, and Trial first.');
        return;
    }

    const splitMode      = splitSelect.value;
    const featureType    = featureSelect.value;
    const classifierName = classifierSelect.value;
    const nClusters      = parseInt(document.getElementById('cluster-select')?.value || 8);

    showLoading('Training Classifier...');

    try {
        const res  = await fetch('/api/predict', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subject_id:   parseInt(subjectId),
                session_idx:  parseInt(sessionIdx),
                trial_id:     parseInt(trialId),
                feature_type: featureType,
                classifier:   classifierName,
                split_mode:   splitMode,
                window_size:  state.windowSize,
                n_clusters:   nClusters,
            })
        });

        const data = await res.json();
        if (!data.success) throw new Error(data.error);

        hideLoading();
        renderDashboard(data);

    } catch (err) {
        hideLoading();
        alert('Classification failed: ' + err.message);
        console.error(err);
    }
});

// ── Render Dashboard with Results ────────────────────────────────────────────
function renderDashboard(data) {
    welcomePanel.classList.add('hidden');
    dashboardPanel.classList.remove('hidden');

    // Header
    const windowLabel = state.windowSize === '4s' ? '4-second' : '1-second';
    subHeader.textContent =
        `Subject ${data.subject_id} · Session ${data.session_idx + 1} · ` +
        `Trial ${data.trial_id} · ${windowLabel} window · ${data.split_mode === '10_5_split' ? '10/5 Split' : 'LOO'}`;

    // Emotion cards
    setEmotionCard('card-ground-truth', 'val-ground-truth', data.ground_truth_name);
    setEmotionCard('card-prediction',   'val-prediction',   data.predicted_name);

    // Accuracy & Duration
    const valAcc = document.getElementById('val-accuracy');
    const valDur = document.getElementById('val-duration');
    if (valAcc) valAcc.textContent = (data.second_by_second_accuracy * 100).toFixed(1) + '%';
    if (valDur) valDur.textContent = data.duration_seconds + 's';

    // Precision & AUC-ROC
    const { precision, auc } = computeMetrics(
        data.second_by_second_predictions,
        data.second_by_second_probabilities,
        data.ground_truth_label
    );
    const valPrec = document.getElementById('val-precision');
    const valAuc  = document.getElementById('val-auc');
    if (valPrec) valPrec.textContent = (precision * 100).toFixed(1) + '%';
    if (valAuc)  valAuc.textContent  = auc.toFixed(3);

    // 10/5 split summary panel
    const summaryPanel = document.getElementById('test-summary-panel');
    const summaryGrid  = document.getElementById('test-trials-container');
    const valTestAcc   = document.getElementById('val-test-accuracy');

    if (data.split_mode === '10_5_split' && data.test_trials_summary) {
        summaryPanel.classList.remove('hidden');
        if (valTestAcc) valTestAcc.textContent = (data.overall_test_accuracy * 100).toFixed(0) + '%';
        summaryGrid.innerHTML = '';
        data.test_trials_summary.forEach(t => {
            const div = document.createElement('div');
            div.className = `trial-result-card ${t.is_correct ? 'correct' : 'incorrect'}`;
            div.innerHTML = `
                <div class="trial-num">Trial ${t.trial_id}</div>
                <div class="trial-truth">${t.ground_truth_name}</div>
                <div class="trial-pred">${t.predicted_name}</div>
                <div class="trial-acc">${(t.second_accuracy * 100).toFixed(0)}%</div>
            `;
            summaryGrid.appendChild(div);
        });
    } else {
        summaryPanel.classList.add('hidden');
    }

    // Build channels map & trial data
    state.channelsMap = buildChannelsMap(data.channels, data.location_grid);
    state.trialData   = data;
    state.currentTime = 0;

    // Reset playback
    clearInterval(playbackInterval);
    playbackInterval = null;

    // Scrubber
    scrubber.max   = data.duration_seconds - 1;
    scrubber.value = 0;

    // Topomap & chart
    renderTopomap();
    initProbChart();
    updateProbChart(data.second_by_second_probabilities);
}

// ── Drag & Drop .mat Upload ───────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleMatUpload(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleMatUpload(fileInput.files[0]);
});

async function handleMatUpload(file) {
    if (!file.name.endsWith('.mat')) {
        alert('Please upload a .mat file.');
        return;
    }

    showLoading('Uploading & Classifying...');

    const nClusters = document.getElementById('cluster-select')?.value || '8';
    const formData  = new FormData();
    formData.append('file',         file);
    formData.append('feature_type', featureSelect.value);
    formData.append('classifier',   classifierSelect.value);
    formData.append('split_mode',   splitSelect.value);
    formData.append('window_size',  state.windowSize);
    formData.append('n_clusters',   nClusters);

    try {
        const res  = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!data.success) throw new Error(data.error);
        hideLoading();
        renderDashboard(data);
    } catch (err) {
        hideLoading();
        alert('Upload failed: ' + err.message);
    }
}

// ── Window Size Selector ──────────────────────────────────────────────────────
function addWindowSizeSelector() {
    const sel = document.getElementById('window-select');
    if (!sel) return;
    sel.addEventListener('change', () => {
        state.windowSize = sel.value;
        loadSubjects();
    });
}

// ── EEG Playback Controls ─────────────────────────────────────────────────────
const playBtn  = document.getElementById('play-btn');
const pauseBtn = document.getElementById('pause-btn');
const resetBtn = document.getElementById('reset-btn');

let playbackInterval = null;

playBtn.addEventListener('click', () => {
    if (!state.trialData) return;

    const maxTime = parseInt(scrubber.max);

    if (state.currentTime >= maxTime) {
        state.currentTime = 0;
        scrubber.value    = 0;
    }

    clearInterval(playbackInterval);

    playbackInterval = setInterval(() => {
        if (state.currentTime >= maxTime) {
            clearInterval(playbackInterval);
            playbackInterval = null;
            return;
        }
        state.currentTime++;
        scrubber.value = state.currentTime;
        renderTopomap();
    }, 1000);
});

pauseBtn.addEventListener('click', () => {
    clearInterval(playbackInterval);
    playbackInterval = null;
});

resetBtn.addEventListener('click', () => {
    clearInterval(playbackInterval);
    playbackInterval  = null;
    state.currentTime = 0;
    scrubber.value    = 0;
    renderTopomap();
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    addWindowSizeSelector();
    loadSubjects();
    initProbChart();
});
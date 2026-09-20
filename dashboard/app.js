/*
 * ======================================================================
 * app.js - Real-Time Dashboard Logic & WebSocket Client
 * ======================================================================
 */

// Activity Configuration & Visual Metadata
const ACTIVITY_META = {
    "WALKING": { icon: "🚶", color: "#3b82f6", label: "Walking" },
    "WALKING_UPSTAIRS": { icon: "🧗", color: "#10b981", label: "Walking Upstairs" },
    "WALKING_DOWNSTAIRS": { icon: "⛷️", color: "#f59e0b", label: "Walking Downstairs" },
    "SITTING": { icon: "🪑", color: "#8b5cf6", label: "Sitting" },
    "STANDING": { icon: "🧍", color: "#ec4899", label: "Standing" },
    "LAYING": { icon: "🛌", color: "#64748b", label: "Laying" },
    "UNKNOWN": { icon: "🔄", color: "#94a3b8", label: "Analyzing..." }
};

const MODEL_SPECS = {
    "proposed": { acc: "96.88%", f1: "0.9685", latency: "0.72 ms", size: "960 KB", diff: "+5.78% vs Baseline" },
    "cnn_lstm": { acc: "93.25%", f1: "0.9318", latency: "0.85 ms", size: "1.2 MB", diff: "+2.15% vs Baseline" },
    "bilstm": { acc: "92.40%", f1: "0.9231", latency: "1.10 ms", size: "1.8 MB", diff: "+1.30% vs Baseline" },
    "cnn": { acc: "91.10%", f1: "0.9102", latency: "0.35 ms", size: "850 KB", diff: "Fast Baseline" },
    "lstm": { acc: "89.80%", f1: "0.8972", latency: "0.65 ms", size: "720 KB", diff: "Standard Baseline" }
};

let ws = null;
let sensorChart = null;
let maxChartPoints = 100;
let packetCounter = 0;
let lastFpsTime = performance.now();

// Attention Heatmap Bars Initialization
function initAttentionHeatmap() {
    const container = document.getElementById("attention-heatmap-container");
    container.innerHTML = "";
    for (let i = 0; i < 128; i++) {
        const bar = document.createElement("div");
        bar.className = "attn-bar";
        bar.id = `attn-bar-${i}`;
        container.appendChild(bar);
    }
}

// Chart.js 6-Channel Telemetry Setup
function initSensorChart() {
    const ctx = document.getElementById('sensorWaveChart').getContext('2d');
    
    const initialLabels = Array.from({ length: maxChartPoints }, (_, i) => "");

    sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: initialLabels,
            datasets: [
                { label: 'Acc-X', borderColor: '#60a5fa', borderWidth: 1.8, data: Array(maxChartPoints).fill(0), tension: 0.2, pointRadius: 0 },
                { label: 'Acc-Y', borderColor: '#34d399', borderWidth: 1.8, data: Array(maxChartPoints).fill(1.0), tension: 0.2, pointRadius: 0 },
                { label: 'Acc-Z', borderColor: '#fbbf24', borderWidth: 1.8, data: Array(maxChartPoints).fill(0), tension: 0.2, pointRadius: 0 },
                { label: 'Gyro-X', borderColor: '#a78bfa', borderWidth: 1.5, borderDash: [3, 3], data: Array(maxChartPoints).fill(0), tension: 0.2, pointRadius: 0 },
                { label: 'Gyro-Y', borderColor: '#f472b6', borderWidth: 1.5, borderDash: [3, 3], data: Array(maxChartPoints).fill(0), tension: 0.2, pointRadius: 0 },
                { label: 'Gyro-Z', borderColor: '#fb7185', borderWidth: 1.5, borderDash: [3, 3], data: Array(maxChartPoints).fill(0), tension: 0.2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#6b7280', font: { size: 10 } },
                    suggestedMin: -2.0,
                    suggestedMax: 2.5
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// WebSocket Connection Management
function connectWebSocket() {
    const host = window.location.host || "localhost:8000";
    const wsUrl = `ws://${host}/ws/stream`;

    const connPill = document.getElementById("connection-pill");
    const dot = connPill.querySelector(".dot");
    const statusText = document.getElementById("conn-status-text");

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        dot.className = "dot connected";
        statusText.textContent = "Live Telemetry 50Hz";
        console.log("[WebSocket] Connected to HAR Telemetry Stream.");
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);

        if (payload.type === "sensor_sample") {
            handleSensorSample(payload.data);
        } else if (payload.type === "inference_update") {
            handleInferenceUpdate(payload.data);
        }
    };

    ws.onclose = () => {
        dot.className = "dot disconnected";
        statusText.textContent = "Reconnecting...";
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error("[WebSocket Error]", err);
    };
}

// Sensor Sample Handler (50Hz chart update)
function handleSensorSample(sample) {
    if (!sensorChart) return;

    packetCounter++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        document.getElementById("telemetry-fps").textContent = packetCounter.toFixed(1);
        packetCounter = 0;
        lastFpsTime = now;
    }

    // Push new values to chart datasets
    sensorChart.data.datasets[0].data.push(sample.ax);
    sensorChart.data.datasets[0].data.shift();

    sensorChart.data.datasets[1].data.push(sample.ay);
    sensorChart.data.datasets[1].data.shift();

    sensorChart.data.datasets[2].data.push(sample.az);
    sensorChart.data.datasets[2].data.shift();

    sensorChart.data.datasets[3].data.push(sample.gx);
    sensorChart.data.datasets[3].data.shift();

    sensorChart.data.datasets[4].data.push(sample.gy);
    sensorChart.data.datasets[4].data.shift();

    sensorChart.data.datasets[5].data.push(sample.gz);
    sensorChart.data.datasets[5].data.shift();

    sensorChart.update('none'); // Update without animation for maximum FPS
}

// Inference Update Handler
function handleInferenceUpdate(pred) {
    const activity = pred.activity || "UNKNOWN";
    const meta = ACTIVITY_META[activity] || ACTIVITY_META["UNKNOWN"];

    // Update Posture & Title
    document.getElementById("activity-icon").textContent = meta.icon;
    document.getElementById("predicted-activity").textContent = meta.label;
    document.getElementById("predicted-activity").style.color = meta.color;

    const ring = document.getElementById("posture-ring");
    ring.style.borderColor = meta.color;
    ring.style.boxShadow = `0 0 30px ${meta.color}66`;

    document.getElementById("confidence-badge").textContent = `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;
    document.getElementById("activity-duration").textContent = `${pred.activity_duration_sec}s`;
    document.getElementById("telemetry-latency").textContent = `${pred.latency_ms} ms`;

    // Render Probabilities
    renderProbabilities(pred.probabilities);

    // Render Attention Heatmap
    if (pred.attention_weights && pred.attention_weights.length > 0) {
        renderAttentionHeatmap(pred.attention_weights);
    }
}

// Render Probability Bars
function renderProbabilities(probs) {
    const container = document.getElementById("probability-bars");
    container.innerHTML = "";

    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([name, val]) => {
        const meta = ACTIVITY_META[name] || { label: name, color: "#6366f1" };
        const pct = (val * 100).toFixed(1);

        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <div class="prob-info">
                <span>${meta.label}</span>
                <span style="color:${meta.color}; font-weight:700;">${pct}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${pct}%; background: ${meta.color};"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

// Render Attention Heatmap
function renderAttentionHeatmap(weights) {
    weights.forEach((w, i) => {
        const bar = document.getElementById(`attn-bar-${i}`);
        if (bar) {
            const h = Math.max(10, Math.min(100, w * 100));
            bar.style.height = `${h}%`;
            // Color gradient from purple to emerald based on intensity
            if (w > 0.7) {
                bar.style.background = "#10b981";
            } else if (w > 0.4) {
                bar.style.background = "#6366f1";
            } else {
                bar.style.background = "rgba(99, 102, 241, 0.3)";
            }
        }
    });
}

// User Action Controls
function setSource(mode) {
    document.getElementById("btn-mode-virtual").classList.toggle("active", mode === "virtual");
    document.getElementById("btn-mode-hardware").classList.toggle("active", mode === "hardware");

    document.getElementById("virtual-controls").classList.toggle("active", mode === "virtual");
    document.getElementById("hardware-controls").classList.toggle("active", mode === "hardware");

    document.getElementById("source-badge").textContent = mode === "virtual" ? "VIRTUAL SENSOR" : "PHYSICAL COM PORT";

    if (mode === "hardware") {
        refreshPorts();
    } else {
        fetch("/api/set_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: "virtual" })
        });
    }
}

function triggerActivity(actName) {
    document.querySelectorAll(".act-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = Array.from(document.querySelectorAll(".act-btn")).find(b => b.textContent.includes(actName.toLowerCase()) || b.getAttribute("onclick").includes(actName));
    if (activeBtn) activeBtn.classList.add("active");

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "set_activity", activity: actName }));
    }
}

function refreshPorts() {
    fetch("/api/ports")
        .then(res => res.json())
        .then(ports => {
            const select = document.getElementById("com-port-select");
            select.innerHTML = "";
            if (ports.length === 0) {
                select.innerHTML = '<option value="">No COM devices found</option>';
            } else {
                ports.forEach(p => {
                    const opt = document.createElement("option");
                    opt.value = p.device;
                    opt.textContent = `${p.device} - ${p.description}`;
                    select.appendChild(opt);
                });
            }
        });
}

function connectHardwarePort() {
    const port = document.getElementById("com-port-select").value;
    if (!port) {
        alert("Please select a valid COM port.");
        return;
    }
    fetch("/api/set_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "hardware", port: port })
    })
    .then(res => res.json())
    .then(data => {
        alert(`Connected to hardware port: ${port}`);
    })
    .catch(err => {
        alert("Could not connect to port: " + err.message);
    });
}

function switchModel(modelKey) {
    const specs = MODEL_SPECS[modelKey] || MODEL_SPECS["proposed"];
    document.getElementById("card-acc").textContent = specs.acc;
    document.getElementById("card-f1").textContent = specs.f1;
    document.getElementById("card-latency").textContent = specs.latency;
    document.getElementById("card-size").textContent = specs.size;

    fetch("/api/select_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelKey, backend: "keras" })
    });
}

// Initialization on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initAttentionHeatmap();
    initSensorChart();
    connectWebSocket();
});

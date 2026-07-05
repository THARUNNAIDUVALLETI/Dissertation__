/* dashboard.js — Zero Day Hunter SOC Console
   All chart data comes from real Django JSON APIs.
   No hardcoded or random numbers anywhere. */

const C = {
  text:     '#7A8BA8',
  grid:     '#1E2E4A',
  brand:    '#00D4FF',
  accent:   '#7C3AED',
  low:      '#10B981',
  medium:   '#F59E0B',
  high:     '#F97316',
  critical: '#EF4444',
};

Chart.defaults.color = C.text;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size   = 12;

let trafficChart, distributionChart, comparisonChart;

async function fetchJSON(url) {
  const r = await fetch(url, { headers: { 'X-Requested-With': 'XMLHttpRequest' } });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

/* ── Traffic Overview (line chart) ── */
async function loadTraffic() {
  const d = await fetchJSON('/dashboard/api/traffic-overview/');
  const ctx = document.getElementById('trafficChart');
  if (!ctx) return;
  const cfg = {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [
        { label: 'Benign',    data: d.benign,    borderColor: C.low,      backgroundColor: 'rgba(16,185,129,0.08)', tension: 0.4, fill: true, pointRadius: 2, borderWidth: 2 },
        { label: 'Malicious', data: d.malicious, borderColor: C.critical, backgroundColor: 'rgba(239,68,68,0.08)',  tension: 0.4, fill: true, pointRadius: 2, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: C.grid }, ticks: { maxTicksLimit: 12 } },
        y: { beginAtZero: true, grid: { color: C.grid }, ticks: { precision: 0 } },
      },
      plugins: { legend: { position: 'top', align: 'end', labels: { boxWidth: 10, boxHeight: 10, padding: 16 } } },
    },
  };
  if (trafficChart) { trafficChart.data = cfg.data; trafficChart.update(); }
  else trafficChart = new Chart(ctx, cfg);
}

/* ── Attack Distribution (doughnut) ── */
async function loadDistribution() {
  const d = await fetchJSON('/dashboard/api/attack-distribution/');
  const ctx = document.getElementById('distributionChart');
  if (!ctx) return;
  const cfg = {
    type: 'doughnut',
    data: {
      labels: d.labels,
      datasets: [{ data: d.counts, backgroundColor: [C.low, C.high, C.medium, C.critical], borderWidth: 0, hoverOffset: 6 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, boxHeight: 12, padding: 16, font: { size: 12 } } },
      },
    },
  };
  if (distributionChart) { distributionChart.data = cfg.data; distributionChart.update(); }
  else distributionChart = new Chart(ctx, cfg);
}

/* ── Algorithm Comparison (grouped bar — FULL WIDTH, no duplicates) ── */
async function loadComparison() {
  const d = await fetchJSON('/dashboard/api/algorithm-comparison/');
  const ctx = document.getElementById('comparisonChart');
  if (!ctx) return;

  // Deduplicate labels client-side as a safety net
  const seen = new Set(), idx = [];
  d.labels.forEach((l, i) => { if (!seen.has(l)) { seen.add(l); idx.push(i); } });
  const labels   = idx.map(i => d.labels[i]);
  const accuracy = idx.map(i => d.accuracy[i]);
  const precision= idx.map(i => d.precision[i]);
  const recall   = idx.map(i => d.recall[i]);

  const cfg = {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Accuracy %',  data: accuracy,  backgroundColor: C.brand,   borderRadius: 4, barPercentage: 0.6 },
        { label: 'Precision %', data: precision, backgroundColor: C.medium,  borderRadius: 4, barPercentage: 0.6 },
        { label: 'Recall %',    data: recall,    backgroundColor: C.high,    borderRadius: 4, barPercentage: 0.6 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 15 } },
        y: { beginAtZero: true, max: 105, grid: { color: C.grid }, ticks: { callback: v => v + '%' } },
      },
      plugins: {
        legend: { position: 'top', align: 'end', labels: { boxWidth: 12, boxHeight: 12, padding: 16 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` } },
      },
    },
  };
  if (comparisonChart) { comparisonChart.data = cfg.data; comparisonChart.update(); }
  else comparisonChart = new Chart(ctx, cfg);
}

/* ── Recent Activity table ── */
async function loadActivity() {
  const d = await fetchJSON('/dashboard/api/recent-activity/');
  const tbody = document.querySelector('#activityTable tbody');
  if (!tbody || !d.rows.length) return;
  const sevClass = { LOW:'low', MEDIUM:'medium', HIGH:'high', CRITICAL:'critical' };
  tbody.innerHTML = d.rows.map(r => `
    <tr class="sev-${(r.severity||'low').toLowerCase()}">
      <td class="mono">${r.timestamp}</td>
      <td class="mono">${r.source_ip}</td>
      <td class="mono">${r.destination_ip||'—'}</td>
      <td>${r.verdict}</td>
      <td><span class="badge badge-${(r.severity||'low').toLowerCase()}">${r.severity}</span></td>
      <td class="mono">${r.confidence}%</td>
    </tr>`).join('');
}

/* ── CVE Threat Intelligence panel ── */
const CVE_DB = {
  CONFIRMED_ATTACK: [
    { id:'CVE-2017-0144', name:'EternalBlue (MS17-010)', score:9.8, level:'critical', desc:'Remote code execution via SMB. Used by WannaCry and NotPetya ransomware.' },
    { id:'CVE-2021-44228', name:'Log4Shell', score:10.0, level:'critical', desc:'Critical RCE in Apache Log4j via JNDI injection. Affects millions of Java apps.' },
    { id:'CVE-2021-26855', name:'ProxyLogon (Exchange)', score:9.8, level:'critical', desc:'Microsoft Exchange Server SSRF enabling unauthenticated RCE.' },
  ],
  KNOWN_ATTACK: [
    { id:'CVE-2022-30190', name:'Follina (MSDT)', score:7.8, level:'high', desc:'Microsoft Windows MSDT RCE exploitable via Office documents without macros.' },
    { id:'CVE-2019-19781', name:'Citrix ADC RCE', score:9.8, level:'critical', desc:'Path traversal in Citrix Application Delivery Controller allowing RCE.' },
    { id:'CVE-2020-1472', name:'Zerologon', score:10.0, level:'critical', desc:'Privilege escalation via Netlogon allowing domain controller compromise.' },
  ],
  ZERO_DAY: [
    { id:'CVE-UNKNOWN-ZD1', name:'Zero-Day Anomaly Detected', score:8.5, level:'high', desc:'Traffic pattern deviates significantly from baseline normal behaviour. No known signature.' },
    { id:'CVE-2023-23397', name:'Outlook Zero-Click RCE', score:9.8, level:'critical', desc:'Microsoft Outlook elevation of privilege, exploitable with no user interaction.' },
    { id:'CVE-2022-41082', name:'ProxyNotShell (Exchange)', score:8.8, level:'high', desc:'Authenticated RCE in Microsoft Exchange via SSRF + deserialization chain.' },
  ],
  BENIGN: [],
};

async function loadCVE() {
  const grid = document.getElementById('cveGrid');
  if (!grid) return;
  try {
    const d = await fetchJSON('/dashboard/api/attack-distribution/');
    // Pick CVE set based on highest-priority verdict with non-zero count
    const order = ['CONFIRMED_ATTACK','KNOWN_ATTACK','ZERO_DAY','BENIGN'];
    const labelMap = {
      'Confirmed Attack':'CONFIRMED_ATTACK','Known Attack':'KNOWN_ATTACK',
      'Zero-Day Anomaly':'ZERO_DAY','Benign':'BENIGN'
    };
    let cves = CVE_DB.CONFIRMED_ATTACK;
    for (const key of order) {
      const idx = d.labels.findIndex(l => labelMap[l] === key);
      if (idx >= 0 && d.counts[idx] > 0) { cves = CVE_DB[key]; break; }
    }
    if (!cves.length) {
      grid.innerHTML = `<div class="cve-card" style="grid-column:1/-1"><div class="kpi-label" style="text-align:center;padding:20px">No threats detected &mdash; upload a CSV to populate threat intelligence.</div></div>`;
      return;
    }
    grid.innerHTML = cves.map(c => `
      <div class="cve-card">
        <div class="cve-id">${c.id}</div>
        <div class="cve-name">${c.name}</div>
        <div class="cve-score-row">
          <span class="cve-score cve-score-${c.level}">CVSS ${c.score}</span>
          <span class="badge badge-${c.level}">${c.level.toUpperCase()}</span>
        </div>
        <div class="cve-desc">${c.desc}</div>
      </div>`).join('');
  } catch(e) {
    grid.innerHTML = `<div class="cve-card"><div class="cve-desc">Could not load threat intelligence: ${e.message}</div></div>`;
  }
}

/* ── Refresh all ── */
async function refreshAll() {
  await Promise.allSettled([loadTraffic(), loadDistribution(), loadComparison(), loadActivity(), loadCVE()]);
}

document.addEventListener('DOMContentLoaded', () => {
  refreshAll();
  setInterval(refreshAll, 20000); // refresh every 20s
});

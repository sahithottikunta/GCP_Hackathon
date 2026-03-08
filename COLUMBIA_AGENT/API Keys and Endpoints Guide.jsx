import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as d3 from "d3";
import _ from "lodash";

// ─── SYNTHETIC DATA ENGINE ───────────────────────────────────────────────────
const generateMarketData = (days = 90) => {
  const data = [];
  let price = 142.5;
  let volume = 2400000;
  const now = Date.now();

  for (let i = days; i >= 0; i--) {
    const timestamp = now - i * 86400000;
    const trend = Math.sin(i / 15) * 3;
    const volatility = 0.015 + Math.random() * 0.025;
    const shock = Math.random() > 0.92 ? (Math.random() - 0.5) * 12 : 0;
    price = Math.max(80, price + trend * 0.3 + (Math.random() - 0.48) * price * volatility + shock);

    const volMultiplier = shock !== 0 ? 2.5 + Math.random() * 2 : 0.7 + Math.random() * 0.8;
    volume = Math.max(500000, volume * volMultiplier * (0.85 + Math.random() * 0.3));

    const isAnomaly = Math.abs(shock) > 4 || volume > 6000000;

    data.push({
      timestamp,
      date: new Date(timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      price: +price.toFixed(2),
      open: +(price - Math.random() * 3).toFixed(2),
      high: +(price + Math.random() * 4).toFixed(2),
      low: +(price - Math.random() * 4).toFixed(2),
      volume: Math.round(volume),
      volatility: +(volatility * 100).toFixed(2),
      anomaly: isAnomaly,
      anomalyScore: isAnomaly ? +(0.7 + Math.random() * 0.3).toFixed(2) : +(Math.random() * 0.35).toFixed(2),
    });
  }
  return data;
};

const generateTransactions = () => {
  const types = ["BUY", "SELL", "SHORT", "COVER"];
  const exchanges = ["NYSE", "NASDAQ", "CBOE", "DARK POOL", "OTC"];
  const entities = [
    "Citadel Securities", "Jane Street", "Two Sigma", "DE Shaw",
    "Bridgewater", "Renaissance Tech", "Point72", "Millennium",
    "Unknown Entity A", "Unknown Entity B", "Offshore LP-7",
    "Shell Corp Delta", "Bermuda Holding III"
  ];

  return Array.from({ length: 40 }, (_, i) => {
    const isSuspicious = Math.random() > 0.75;
    const size = isSuspicious ? Math.round(50000 + Math.random() * 450000) : Math.round(500 + Math.random() * 15000);
    return {
      id: `TXN-${String(1000 + i).padStart(6, "0")}`,
      time: `${String(9 + Math.floor(Math.random() * 7)).padStart(2, "0")}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`,
      type: types[Math.floor(Math.random() * types.length)],
      entity: entities[Math.floor(Math.random() * entities.length)],
      exchange: exchanges[Math.floor(Math.random() * exchanges.length)],
      size,
      price: +(140 + Math.random() * 10).toFixed(2),
      suspicious: isSuspicious,
      flag: isSuspicious
        ? ["WASH TRADE", "SPOOFING", "LAYERING", "FRONT-RUN", "INSIDER"][Math.floor(Math.random() * 5)]
        : null,
    };
  }).sort((a, b) => a.time.localeCompare(b.time));
};

const generateAlerts = () => [
  { id: 1, severity: "CRITICAL", time: "10:32:18", title: "Volume Spike Anomaly", desc: "Trading volume exceeded 3.2σ above 20-day mean. Concentrated in dark pool activity.", score: 0.94 },
  { id: 2, severity: "HIGH", time: "11:15:42", title: "Correlation Breakdown", desc: "SPY-QQQ correlation dropped from 0.87 to 0.31 in 15-minute window. Sector rotation or dislocation suspected.", score: 0.82 },
  { id: 3, severity: "CRITICAL", time: "12:01:07", title: "Potential Spoofing Pattern", desc: "Repeated large bid placement and cancellation detected. 47 orders placed/cancelled in 90 seconds.", score: 0.97 },
  { id: 4, severity: "MEDIUM", time: "13:45:33", title: "Unusual Options Flow", desc: "Heavy OTM put buying on no news catalyst. Volume 8x average for strike/expiry combination.", score: 0.71 },
  { id: 5, severity: "HIGH", time: "14:22:11", title: "Cross-Exchange Arbitrage", desc: "Price discrepancy of $0.47 persisted for 12 seconds across NYSE and NASDAQ — atypical latency.", score: 0.86 },
  { id: 6, severity: "LOW", time: "15:10:55", title: "After-Hours Cluster", desc: "Concentrated block trades from same counterparty in final 10 minutes. Pattern matches prior event-driven positioning.", score: 0.45 },
];

// ─── MINI SPARKLINE COMPONENT ────────────────────────────────────────────────
const Sparkline = ({ data, width = 120, height = 36, color = "#00ffc8", showArea = true }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const x = d3.scaleLinear().domain([0, data.length - 1]).range([2, width - 2]);
    const y = d3.scaleLinear().domain(d3.extent(data)).range([height - 2, 2]);

    const line = d3.line().x((_, i) => x(i)).y(d => y(d)).curve(d3.curveCatmullRom);

    if (showArea) {
      const area = d3.area().x((_, i) => x(i)).y0(height).y1(d => y(d)).curve(d3.curveCatmullRom);
      svg.append("path").datum(data).attr("d", area).attr("fill", color).attr("fill-opacity", 0.08);
    }

    svg.append("path").datum(data).attr("d", line).attr("fill", "none")
      .attr("stroke", color).attr("stroke-width", 1.5).attr("stroke-linecap", "round");

    svg.append("circle").attr("cx", x(data.length - 1)).attr("cy", y(data[data.length - 1]))
      .attr("r", 2.5).attr("fill", color);
  }, [data, width, height, color, showArea]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ─── PRICE CHART COMPONENT ──────────────────────────────────────────────────
const PriceChart = ({ data, width = 700, height = 300 }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 30, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, data.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([d3.min(data, d => d.low) * 0.98, d3.max(data, d => d.high) * 1.02]).range([h, 0]);

    // Grid
    g.selectAll(".grid-y").data(y.ticks(5)).enter().append("line")
      .attr("x1", 0).attr("x2", w).attr("y1", d => y(d)).attr("y2", d => y(d))
      .attr("stroke", "#1a2332").attr("stroke-width", 1);

    // Area
    const area = d3.area().x((_, i) => x(i)).y0(h).y1(d => y(d.price)).curve(d3.curveCatmullRom);
    const gradient = svg.append("defs").append("linearGradient").attr("id", "priceGrad").attr("x1", "0").attr("y1", "0").attr("x2", "0").attr("y2", "1");
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "#00ffc8").attr("stop-opacity", 0.25);
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "#00ffc8").attr("stop-opacity", 0);
    g.append("path").datum(data).attr("d", area).attr("fill", "url(#priceGrad)");

    // Line
    const line = d3.line().x((_, i) => x(i)).y(d => y(d.price)).curve(d3.curveCatmullRom);
    g.append("path").datum(data).attr("d", line).attr("fill", "none")
      .attr("stroke", "#00ffc8").attr("stroke-width", 2).attr("stroke-linecap", "round");

    // Anomaly markers
    data.forEach((d, i) => {
      if (d.anomaly) {
        g.append("circle").attr("cx", x(i)).attr("cy", y(d.price))
          .attr("r", 5).attr("fill", "#ff3860").attr("fill-opacity", 0.8)
          .attr("stroke", "#ff3860").attr("stroke-width", 2).attr("stroke-opacity", 0.3);
        g.append("circle").attr("cx", x(i)).attr("cy", y(d.price))
          .attr("r", 10).attr("fill", "none")
          .attr("stroke", "#ff3860").attr("stroke-width", 1).attr("stroke-opacity", 0.4)
          .attr("stroke-dasharray", "2,2");
      }
    });

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(8).tickFormat(i => data[Math.round(i)]?.date || ""))
      .selectAll("text").attr("fill", "#5a7a9a").style("font-size", "10px");
    g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d => `$${d.toFixed(0)}`))
      .selectAll("text").attr("fill", "#5a7a9a").style("font-size", "10px");
    g.selectAll(".domain").attr("stroke", "#1a2332");
    g.selectAll(".tick line").attr("stroke", "#1a2332");

  }, [data, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ─── VOLUME CHART ───────────────────────────────────────────────────────────
const VolumeChart = ({ data, width = 700, height = 120 }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 5, right: 20, bottom: 25, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(data.map((_, i) => i)).range([0, w]).padding(0.3);
    const y = d3.scaleLinear().domain([0, d3.max(data, d => d.volume)]).range([h, 0]);

    g.selectAll("rect").data(data).enter().append("rect")
      .attr("x", (_, i) => x(i)).attr("y", d => y(d.volume))
      .attr("width", x.bandwidth()).attr("height", d => h - y(d.volume))
      .attr("rx", 1)
      .attr("fill", d => d.anomaly ? "#ff3860" : "#1e3a5f")
      .attr("fill-opacity", d => d.anomaly ? 0.9 : 0.6);

    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % 12 === 0)).tickFormat(i => data[i]?.date || ""))
      .selectAll("text").attr("fill", "#5a7a9a").style("font-size", "10px");
    g.append("g").call(d3.axisLeft(y).ticks(3).tickFormat(d => `${(d / 1000000).toFixed(1)}M`))
      .selectAll("text").attr("fill", "#5a7a9a").style("font-size", "10px");
    g.selectAll(".domain").attr("stroke", "#1a2332");
    g.selectAll(".tick line").attr("stroke", "#1a2332");

  }, [data, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ─── ANOMALY SCORE GAUGE ────────────────────────────────────────────────────
const AnomalyGauge = ({ score, size = 140 }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const cx = size / 2, cy = size / 2, r = size * 0.38;
    const startAngle = -Math.PI * 0.75, endAngle = Math.PI * 0.75;
    const angleRange = endAngle - startAngle;

    const arc = d3.arc().innerRadius(r - 6).outerRadius(r).startAngle(startAngle).cornerRadius(3);

    // Background arc
    svg.append("path").attr("transform", `translate(${cx},${cy})`)
      .attr("d", arc.endAngle(endAngle)()).attr("fill", "#1a2332");

    // Score arc
    const color = score > 0.8 ? "#ff3860" : score > 0.5 ? "#ffaa00" : "#00ffc8";
    svg.append("path").attr("transform", `translate(${cx},${cy})`)
      .attr("d", arc.endAngle(startAngle + angleRange * score)())
      .attr("fill", color);

    // Center text
    svg.append("text").attr("x", cx).attr("y", cy - 4).attr("text-anchor", "middle")
      .attr("fill", color).style("font-size", "24px").style("font-weight", "700")
      .style("font-family", "'JetBrains Mono', monospace").text(score.toFixed(2));
    svg.append("text").attr("x", cx).attr("y", cy + 16).attr("text-anchor", "middle")
      .attr("fill", "#5a7a9a").style("font-size", "10px").style("font-family", "'JetBrains Mono', monospace")
      .text("ANOMALY SCORE");

  }, [score, size]);

  return <svg ref={svgRef} width={size} height={size} />;
};

// ─── CORRELATION MATRIX ─────────────────────────────────────────────────────
const CorrelationMatrix = () => {
  const assets = ["SPY", "QQQ", "IWM", "VIX", "TLT", "GLD"];
  const matrix = assets.map((_, i) =>
    assets.map((_, j) => {
      if (i === j) return 1;
      if (i < j) return +(0.3 + Math.random() * 0.6).toFixed(2) * (Math.random() > 0.3 ? 1 : -1);
      return 0;
    })
  );
  matrix.forEach((row, i) => row.forEach((val, j) => { if (i > j) matrix[i][j] = matrix[j][i]; }));

  const getColor = (v) => {
    if (v === 1) return "#0d1b2a";
    const abs = Math.abs(v);
    if (v > 0) return `rgba(0, 255, 200, ${abs * 0.7})`;
    return `rgba(255, 56, 96, ${abs * 0.7})`;
  };

  return (
    <div style={{ display: "inline-block" }}>
      <div style={{ display: "flex", marginLeft: 40 }}>
        {assets.map(a => (
          <div key={a} style={{ width: 44, textAlign: "center", fontSize: 10, color: "#5a7a9a", fontFamily: "'JetBrains Mono', monospace" }}>{a}</div>
        ))}
      </div>
      {matrix.map((row, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center" }}>
          <div style={{ width: 40, textAlign: "right", paddingRight: 8, fontSize: 10, color: "#5a7a9a", fontFamily: "'JetBrains Mono', monospace" }}>{assets[i]}</div>
          {row.map((v, j) => (
            <div key={j} style={{
              width: 40, height: 32, margin: 2, borderRadius: 4,
              background: getColor(v), display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 9, color: i === j ? "#2a3f5f" : "#c0d8f0", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
            }}>
              {i === j ? "—" : v.toFixed(2)}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

// ─── MAIN DASHBOARD ─────────────────────────────────────────────────────────
export default function FinancialAnomalyDashboard() {
  const [marketData] = useState(() => generateMarketData(90));
  const [transactions] = useState(() => generateTransactions());
  const [alerts] = useState(() => generateAlerts());
  const [activeTab, setActiveTab] = useState("overview");
  const [currentTime, setCurrentTime] = useState(new Date());
  const [scanProgress, setScanProgress] = useState(0);
  const [selectedAlert, setSelectedAlert] = useState(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setScanProgress(p => p >= 100 ? 0 : p + Math.random() * 3);
    }, 150);
    return () => clearInterval(timer);
  }, []);

  const latestPrice = marketData[marketData.length - 1];
  const prevPrice = marketData[marketData.length - 2];
  const priceChange = latestPrice.price - prevPrice.price;
  const pctChange = ((priceChange / prevPrice.price) * 100).toFixed(2);
  const totalAnomalies = marketData.filter(d => d.anomaly).length;
  const suspiciousTx = transactions.filter(t => t.suspicious).length;
  const avgScore = (marketData.reduce((s, d) => s + d.anomalyScore, 0) / marketData.length).toFixed(2);

  const severityColors = { CRITICAL: "#ff3860", HIGH: "#ff8c00", MEDIUM: "#ffaa00", LOW: "#5a7a9a" };
  const typeColors = { BUY: "#00ffc8", SELL: "#ff3860", SHORT: "#ff8c00", COVER: "#6c5ce7" };

  const styles = {
    root: {
      background: "#0a1628", color: "#c0d8f0", fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      minHeight: "100vh", padding: 0, margin: 0,
      position: "relative", overflow: "hidden",
    },
    scanline: {
      position: "fixed", top: 0, left: 0, right: 0, height: 2,
      background: `linear-gradient(90deg, transparent, #00ffc8, transparent)`,
      transform: `translateX(${scanProgress - 50}%)`,
      opacity: 0.6, zIndex: 50, transition: "transform 0.15s linear",
    },
    header: {
      padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between",
      borderBottom: "1px solid #0f2640", background: "rgba(10, 22, 40, 0.95)",
      backdropFilter: "blur(20px)", position: "sticky", top: 0, zIndex: 40,
    },
    logo: { display: "flex", alignItems: "center", gap: 12 },
    logoIcon: {
      width: 36, height: 36, borderRadius: 8, background: "linear-gradient(135deg, #00ffc8, #0088ff)",
      display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, fontWeight: 900, color: "#0a1628",
    },
    logoText: { fontSize: 16, fontWeight: 700, letterSpacing: 2, color: "#00ffc8" },
    logoSub: { fontSize: 9, color: "#5a7a9a", letterSpacing: 3, marginTop: 2 },
    headerRight: { display: "flex", alignItems: "center", gap: 20 },
    liveBadge: {
      display: "flex", alignItems: "center", gap: 6, padding: "4px 12px",
      borderRadius: 20, background: "rgba(255, 56, 96, 0.15)", border: "1px solid rgba(255, 56, 96, 0.3)",
      fontSize: 10, fontWeight: 700, letterSpacing: 1.5, color: "#ff3860",
    },
    liveDot: {
      width: 7, height: 7, borderRadius: "50%", background: "#ff3860",
      animation: "pulse 2s infinite",
    },
    clock: { fontSize: 13, color: "#5a7a9a", letterSpacing: 1 },
    tabs: {
      display: "flex", gap: 2, padding: "12px 24px",
      borderBottom: "1px solid #0f2640", background: "#0c1a2e",
    },
    tab: (active) => ({
      padding: "8px 20px", borderRadius: 6, fontSize: 11, fontWeight: 600,
      letterSpacing: 1, cursor: "pointer", transition: "all 0.2s",
      background: active ? "rgba(0, 255, 200, 0.1)" : "transparent",
      color: active ? "#00ffc8" : "#5a7a9a",
      border: active ? "1px solid rgba(0, 255, 200, 0.2)" : "1px solid transparent",
    }),
    content: { padding: 24 },
    grid: (cols) => ({ display: "grid", gridTemplateColumns: cols, gap: 16, marginBottom: 16 }),
    card: {
      background: "linear-gradient(145deg, #0d1f35, #0a1628)",
      border: "1px solid #122a45", borderRadius: 12, padding: 20,
      position: "relative", overflow: "hidden",
    },
    cardGlow: (color) => ({
      position: "absolute", top: -40, right: -40, width: 100, height: 100,
      borderRadius: "50%", background: color, opacity: 0.05, filter: "blur(30px)",
    }),
    cardLabel: { fontSize: 9, fontWeight: 700, letterSpacing: 2, color: "#5a7a9a", marginBottom: 8, textTransform: "uppercase" },
    cardValue: (color = "#c0d8f0") => ({ fontSize: 28, fontWeight: 700, color, lineHeight: 1.1 }),
    cardDelta: (positive) => ({
      display: "inline-flex", alignItems: "center", gap: 4, marginTop: 6,
      padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600,
      background: positive ? "rgba(0, 255, 200, 0.1)" : "rgba(255, 56, 96, 0.1)",
      color: positive ? "#00ffc8" : "#ff3860",
    }),
    sectionTitle: {
      fontSize: 11, fontWeight: 700, letterSpacing: 2, color: "#5a7a9a",
      textTransform: "uppercase", marginBottom: 16, display: "flex",
      alignItems: "center", gap: 8,
    },
    titleDot: (color) => ({ width: 6, height: 6, borderRadius: "50%", background: color }),
    alertCard: (severity, selected) => ({
      padding: "14px 16px", borderRadius: 8, marginBottom: 8, cursor: "pointer",
      background: selected ? "rgba(0, 255, 200, 0.05)" : "rgba(13, 31, 53, 0.6)",
      border: `1px solid ${selected ? "rgba(0, 255, 200, 0.2)" : "#122a45"}`,
      borderLeft: `3px solid ${severityColors[severity]}`,
      transition: "all 0.2s",
    }),
    txRow: (suspicious) => ({
      display: "grid", gridTemplateColumns: "80px 55px 140px 80px 80px 90px 80px",
      padding: "10px 12px", borderRadius: 6, marginBottom: 4, fontSize: 11,
      alignItems: "center",
      background: suspicious ? "rgba(255, 56, 96, 0.06)" : "transparent",
      borderLeft: suspicious ? "2px solid #ff3860" : "2px solid transparent",
    }),
    badge: (color) => ({
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: `${color}18`, color, fontSize: 9, fontWeight: 700, letterSpacing: 0.5,
    }),
    keyframeStyle: `
      @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      @keyframes slideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; }
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: #0a1628; }
      ::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 3px; }
    `,
  };

  const renderOverview = () => (
    <>
      <div style={styles.grid("repeat(4, 1fr)")}>
        {[
          { label: "LAST PRICE", value: `$${latestPrice.price}`, delta: `${priceChange > 0 ? "+" : ""}${pctChange}%`, positive: priceChange > 0, color: "#00ffc8", sparkData: marketData.slice(-20).map(d => d.price) },
          { label: "ANOMALIES DETECTED", value: totalAnomalies, delta: `${suspiciousTx} suspicious txns`, positive: false, color: "#ff3860", sparkData: marketData.slice(-20).map(d => d.anomalyScore) },
          { label: "AVG ANOMALY SCORE", value: avgScore, delta: avgScore > 0.5 ? "ELEVATED" : "NORMAL", positive: avgScore <= 0.5, color: "#ffaa00", sparkData: marketData.slice(-20).map(d => d.anomalyScore) },
          { label: "VOLUME (TODAY)", value: `${(latestPrice.volume / 1e6).toFixed(1)}M`, delta: `${((latestPrice.volume / prevPrice.volume - 1) * 100).toFixed(0)}% vs prev`, positive: latestPrice.volume < prevPrice.volume * 1.5, color: "#6c5ce7", sparkData: marketData.slice(-20).map(d => d.volume) },
        ].map((item, i) => (
          <div key={i} style={{ ...styles.card, animation: `slideIn 0.4s ease ${i * 0.1}s both` }}>
            <div style={styles.cardGlow(item.color)} />
            <div style={styles.cardLabel}>{item.label}</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
              <div>
                <div style={styles.cardValue(item.color)}>{item.value}</div>
                <div style={styles.cardDelta(item.positive)}>{item.positive ? "▲" : "▼"} {item.delta}</div>
              </div>
              <Sparkline data={item.sparkData} color={item.color} width={100} height={40} />
            </div>
          </div>
        ))}
      </div>

      <div style={styles.grid("2fr 1fr")}>
        <div style={styles.card}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#00ffc8")} />
            PRICE ACTION & ANOMALY OVERLAY
          </div>
          <PriceChart data={marketData} width={580} height={260} />
          <VolumeChart data={marketData} width={580} height={100} />
        </div>

        <div style={styles.card}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#ff3860")} />
            THREAT SCORE
          </div>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}>
            <AnomalyGauge score={parseFloat(avgScore) + 0.25} size={160} />
          </div>
          <div style={{ padding: "0 8px" }}>
            {[
              { label: "Wash Trading Risk", val: 0.82, color: "#ff3860" },
              { label: "Spoofing Probability", val: 0.91, color: "#ff3860" },
              { label: "Insider Activity", val: 0.45, color: "#ffaa00" },
              { label: "Market Manipulation", val: 0.67, color: "#ff8c00" },
            ].map((item, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 4 }}>
                  <span style={{ color: "#5a7a9a" }}>{item.label}</span>
                  <span style={{ color: item.color, fontWeight: 700 }}>{(item.val * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "#0f2640", overflow: "hidden" }}>
                  <div style={{ width: `${item.val * 100}%`, height: "100%", borderRadius: 2, background: item.color, transition: "width 1s ease" }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.grid("1fr 1fr")}>
        <div style={styles.card}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#6c5ce7")} />
            CORRELATION MATRIX
          </div>
          <CorrelationMatrix />
        </div>

        <div style={styles.card}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#ffaa00")} />
            LIVE ALERTS
          </div>
          <div style={{ maxHeight: 300, overflowY: "auto" }}>
            {alerts.map(alert => (
              <div key={alert.id} style={styles.alertCard(alert.severity, selectedAlert === alert.id)} onClick={() => setSelectedAlert(selectedAlert === alert.id ? null : alert.id)}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={styles.badge(severityColors[alert.severity])}>{alert.severity}</span>
                  <span style={{ fontSize: 10, color: "#5a7a9a" }}>{alert.time}</span>
                </div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#c0d8f0", marginBottom: 4 }}>{alert.title}</div>
                {selectedAlert === alert.id && (
                  <div style={{ fontSize: 11, color: "#5a7a9a", lineHeight: 1.5, marginTop: 6, paddingTop: 6, borderTop: "1px solid #122a45" }}>
                    {alert.desc}
                    <div style={{ marginTop: 6, fontSize: 10, color: severityColors[alert.severity] }}>
                      Confidence: {(alert.score * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );

  const renderTransactions = () => (
    <div style={styles.card}>
      <div style={styles.sectionTitle}>
        <div style={styles.titleDot("#ff3860")} />
        TRANSACTION SURVEILLANCE FEED
        <span style={{ marginLeft: "auto", fontSize: 10, color: "#5a7a9a" }}>
          {suspiciousTx} flagged of {transactions.length} total
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "80px 55px 140px 80px 80px 90px 80px", padding: "8px 12px", fontSize: 9, fontWeight: 700, letterSpacing: 1.5, color: "#5a7a9a", borderBottom: "1px solid #122a45", marginBottom: 8 }}>
        <div>TIME</div><div>TYPE</div><div>ENTITY</div><div>SIZE</div><div>PRICE</div><div>EXCHANGE</div><div>FLAG</div>
      </div>
      <div style={{ maxHeight: 520, overflowY: "auto" }}>
        {transactions.map((tx, i) => (
          <div key={i} style={styles.txRow(tx.suspicious)}>
            <div style={{ color: "#5a7a9a", fontSize: 11 }}>{tx.time}</div>
            <div style={styles.badge(typeColors[tx.type])}>{tx.type}</div>
            <div style={{ color: tx.suspicious ? "#ff8c00" : "#c0d8f0", fontSize: 11, fontWeight: tx.suspicious ? 600 : 400 }}>{tx.entity}</div>
            <div style={{ color: tx.size > 100000 ? "#ff3860" : "#c0d8f0", fontSize: 11, fontWeight: tx.size > 100000 ? 700 : 400 }}>
              {tx.size > 999 ? `${(tx.size / 1000).toFixed(1)}K` : tx.size}
            </div>
            <div style={{ color: "#c0d8f0", fontSize: 11 }}>${tx.price}</div>
            <div style={{ color: tx.exchange === "DARK POOL" ? "#6c5ce7" : "#5a7a9a", fontSize: 10, fontWeight: tx.exchange === "DARK POOL" ? 600 : 400 }}>{tx.exchange}</div>
            <div>
              {tx.flag && <span style={styles.badge("#ff3860")}>{tx.flag}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAnalysis = () => (
    <div style={styles.grid("1fr 1fr")}>
      <div style={styles.card}>
        <div style={styles.sectionTitle}>
          <div style={styles.titleDot("#00ffc8")} />
          AI ANALYST BRIEF
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.8, color: "#8ea8c8" }}>
          <p style={{ marginBottom: 16 }}>
            <span style={{ color: "#00ffc8", fontWeight: 700 }}>SUMMARY:</span> Today's session shows
            elevated anomaly activity concentrated in the mid-morning window. What stands out here is
            the convergence of volume spikes with dark pool order routing — a pattern that typically
            warrants closer investigation.
          </p>
          <p style={{ marginBottom: 16 }}>
            <span style={{ color: "#ffaa00", fontWeight: 700 }}>KEY FINDING:</span> The spoofing
            probability score of 91% is driven by a sequence of 47 large bid placements and rapid
            cancellations within a 90-second window. This pattern is atypical because legitimate
            market-making activity would show a more stochastic cancellation distribution.
          </p>
          <p style={{ marginBottom: 16 }}>
            <span style={{ color: "#ff3860", fontWeight: 700 }}>RISK VECTOR:</span> Cross-referencing
            the correlation breakdown between SPY and QQQ (dropping from 0.87 to 0.31 in 15 minutes)
            with the unusual options flow on OTM puts suggests potential information asymmetry.
            The 8x average volume on specific strike/expiry combinations without a catalyst is notable.
          </p>
          <p>
            <span style={{ color: "#6c5ce7", fontWeight: 700 }}>RECOMMENDATION:</span> Flag entities
            "Unknown Entity A" and "Offshore LP-7" for enhanced surveillance. The concentration of
            their activity in dark pools during the anomaly window exceeds normal positioning behavior
            by 2.8 standard deviations.
          </p>
        </div>
      </div>

      <div>
        <div style={{ ...styles.card, marginBottom: 16 }}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#ff8c00")} />
            VOLATILITY REGIME
          </div>
          <Sparkline data={marketData.map(d => d.volatility)} width={340} height={80} color="#ff8c00" />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12, fontSize: 10, color: "#5a7a9a" }}>
            <span>30D Mean: {(marketData.slice(-30).reduce((s, d) => s + d.volatility, 0) / 30).toFixed(2)}%</span>
            <span>Current: {latestPrice.volatility}%</span>
            <span>Peak: {Math.max(...marketData.map(d => d.volatility)).toFixed(2)}%</span>
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.sectionTitle}>
            <div style={styles.titleDot("#ff3860")} />
            FLAGGED ENTITIES
          </div>
          {[
            { name: "Unknown Entity A", txns: 12, vol: "$4.2M", risk: 0.89 },
            { name: "Offshore LP-7", txns: 8, vol: "$2.8M", risk: 0.76 },
            { name: "Shell Corp Delta", txns: 5, vol: "$1.1M", risk: 0.62 },
            { name: "Bermuda Holding III", txns: 3, vol: "$890K", risk: 0.44 },
          ].map((entity, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 0", borderBottom: i < 3 ? "1px solid #122a45" : "none" }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#c0d8f0" }}>{entity.name}</div>
                <div style={{ fontSize: 10, color: "#5a7a9a" }}>{entity.txns} txns · {entity.vol} volume</div>
              </div>
              <div style={{
                padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700,
                background: entity.risk > 0.7 ? "rgba(255,56,96,0.15)" : entity.risk > 0.5 ? "rgba(255,170,0,0.15)" : "rgba(90,122,154,0.15)",
                color: entity.risk > 0.7 ? "#ff3860" : entity.risk > 0.5 ? "#ffaa00" : "#5a7a9a",
              }}>
                {(entity.risk * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.root}>
      <style>{styles.keyframeStyle}</style>
      <div style={styles.scanline} />

      {/* Header */}
      <div style={styles.header}>
        <div style={styles.logo}>
          <div style={styles.logoIcon}>◈</div>
          <div>
            <div style={styles.logoText}>SENTINEL</div>
            <div style={styles.logoSub}>ANOMALY DETECTION ENGINE</div>
          </div>
        </div>
        <div style={styles.headerRight}>
          <div style={styles.liveBadge}>
            <div style={styles.liveDot} />
            LIVE MONITORING
          </div>
          <div style={styles.clock}>
            {currentTime.toLocaleTimeString("en-US", { hour12: false })}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        {["overview", "transactions", "analysis"].map(tab => (
          <div key={tab} style={styles.tab(activeTab === tab)} onClick={() => setActiveTab(tab)}>
            {tab.toUpperCase()}
          </div>
        ))}
      </div>

      {/* Content */}
      <div style={styles.content}>
        {activeTab === "overview" && renderOverview()}
        {activeTab === "transactions" && renderTransactions()}
        {activeTab === "analysis" && renderAnalysis()}
      </div>

      {/* Footer */}
      <div style={{ padding: "16px 24px", borderTop: "1px solid #0f2640", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 10, color: "#2a3f5f" }}>
        <span>SENTINEL v3.1.0 · ANOMALY DETECTION ENGINE · ALL DATA SIMULATED FOR DEMONSTRATION</span>
        <span>SCAN CYCLE: {scanProgress.toFixed(0)}% · NEXT REFRESH: {Math.max(0, 100 - scanProgress).toFixed(0)}%</span>
      </div>
    </div>
  );
}

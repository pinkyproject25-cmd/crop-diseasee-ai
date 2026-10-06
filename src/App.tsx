import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock3,
  CloudSun,
  FileImage,
  Gauge,
  HeartPulse,
  History as HistoryIcon,
  Info,
  Languages,
  Leaf,
  Menu,
  Mic2,
  Play,
  ShieldCheck,
  Sprout,
  Square,
  Trash2,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Link, NavLink, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { analyzeImage, ApiError, getReportAudio, translateReport } from "./lib/api";
import {
  clearDraftImage,
  clearHistory,
  deleteReport,
  getActiveReport,
  getDraftImage,
  loadHistory,
  saveReport,
  setActiveReport,
  setDraftImage,
} from "./lib/storage";
import { cropCoverage } from "./data/crops";
import type { AnalysisReport, Language } from "./types";

const nav = [
  { to: "/", label: "Home", icon: Leaf },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/crops", label: "Supported crops", icon: Sprout },
  { to: "/about", label: "About", icon: Info },
];

function Layout({ children }: { children: React.ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <Link className="brand" to="/" aria-label="Crop Disease AI home">
          <span className="brand-mark"><Leaf size={24} /></span>
          <span><strong>Crop Disease</strong><small>AI</small></span>
        </Link>
        <nav className="desktop-nav" aria-label="Main navigation">
          {nav.map(({ to, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
        <button className="icon-button mobile-menu" onClick={() => setMenuOpen((open) => !open)} aria-label="Open menu">
          {menuOpen ? <X /> : <Menu />}
        </button>
      </header>
      {menuOpen && (
        <nav className="mobile-nav" aria-label="Mobile navigation">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to}><Icon size={19} />{label}</NavLink>
          ))}
        </nav>
      )}
      <main>{children}</main>
      <footer>
        <div><Leaf size={18} /> Crop Disease AI</div>
        <p>AI-assisted visual screening. Confirm critical crop decisions with a qualified local expert.</p>
      </footer>
    </div>
  );
}

function FilePicker({ compact = false }: { compact?: boolean }) {
  const navigate = useNavigate();
  const uploadRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");

  const acceptFile = (file?: File) => {
    setError("");
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Choose a JPG, PNG, or WebP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("The image is larger than 10 MB. Choose a smaller photo.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setDraftImage(reader.result);
        navigate("/analyze");
      }
    };
    reader.onerror = () => setError("The image could not be read.");
    reader.readAsDataURL(file);
  };

  return (
    <div className={compact ? "picker compact" : "picker"}>
      <input ref={uploadRef} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => acceptFile(e.target.files?.[0])} />
      <input ref={cameraRef} hidden type="file" accept="image/*" capture="environment" onChange={(e) => acceptFile(e.target.files?.[0])} />
      <button className="primary-button" onClick={() => uploadRef.current?.click()}><Upload size={20} /> Upload image</button>
      <button className="secondary-button" onClick={() => cameraRef.current?.click()}><Camera size={20} /> Take photo</button>
      {error && <p className="inline-error"><TriangleAlert size={16} /> {error}</p>}
    </div>
  );
}

function HomePage() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow"><ShieldCheck size={16} /> Honest, image-based screening</span>
          <h1>Understand what your crop leaf is showing.</h1>
          <p>Upload a clear leaf photo or take one now. Crop Disease AI checks the visual image and returns a simple report with uncertainty shown clearly.</p>
          <FilePicker />
          <div className="trust-row">
            <span><CheckCircle2 /> Filename ignored</span>
            <span><CheckCircle2 /> Unknown images rejected</span>
            <span><CheckCircle2 /> No account required</span>
          </div>
        </div>
        <div className="hero-visual" aria-label="Crop leaf analysis illustration">
          <div className="leaf-orbit orbit-one" />
          <div className="leaf-orbit orbit-two" />
          <div className="scan-card">
            <div className="scan-header"><span>Visual scan</span><Activity size={18} /></div>
            <div className="leaf-illustration"><Leaf size={126} strokeWidth={1.15} /></div>
            <div className="scan-line" />
            <div className="scan-status"><span className="pulse" /> Ready for a clear leaf photo</div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <span className="kicker">How it works</span>
          <h2>Three clear steps</h2>
          <p>The report keeps model confidence, visible damage, and crop health as separate measurements.</p>
        </div>
        <div className="steps-grid">
          {[
            [FileImage, "1", "Add a leaf photo", "Use good light and keep one leaf in focus."],
            [HeartPulse, "2", "Run visual analysis", "The service reads image pixels and checks whether it can make a supported prediction."],
            [Gauge, "3", "Review the report", "See the result, confidence, visible severity when measurable, weather, and practical next steps."],
          ].map(([Icon, number, title, body]) => {
            const StepIcon = Icon as typeof FileImage;
            return (
              <article className="step-card" key={String(number)}>
                <div className="step-icon"><StepIcon /></div>
                <span className="step-number">{String(number)}</span>
                <h3>{String(title)}</h3>
                <p>{String(body)}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="crop-strip">
        <div>
          <span className="kicker">Scope</span>
          <h2>14 target crops</h2>
          <p>Coverage is reported per crop and disease. A crop name does not imply that every disease is recognizable.</p>
        </div>
        <div className="crop-cloud">
          {cropCoverage.map(({ crop }) => <span key={crop}>{crop}</span>)}
        </div>
        <Link className="text-link" to="/crops">Review coverage <ChevronRight size={17} /></Link>
      </section>
    </>
  );
}

function dataUrlToFile(dataUrl: string): File {
  const [header, encoded] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);/)?.[1] || "image/jpeg";
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], "leaf-capture." + (mime.split("/")[1] || "jpg"), { type: mime });
}

function getLocation(): Promise<{ latitude?: number; longitude?: number }> {
  if (!navigator.geolocation) return Promise.resolve({});
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (position) => resolve({ latitude: position.coords.latitude, longitude: position.coords.longitude }),
      () => resolve({}),
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 600000 },
    );
  });
}

function AnalyzePage() {
  const navigate = useNavigate();
  const [image] = useState(getDraftImage);
  const [includeWeather, setIncludeWeather] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  if (!image) {
    return (
      <section className="narrow-section empty-state">
        <FileImage size={48} />
        <h1>No leaf image selected</h1>
        <p>Add a clear photograph before starting analysis.</p>
        <FilePicker compact />
      </section>
    );
  }

  const runAnalysis = async () => {
    setLoading(true);
    setError("");
    try {
      const location = includeWeather ? await getLocation() : {};
      const report = await analyzeImage(dataUrlToFile(image), location.latitude, location.longitude);
      report.thumbnailDataUrl = image;
      saveReport(report);
      clearDraftImage();
      navigate("/result");
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="narrow-section analyze-page">
      <div className="page-heading">
        <span className="kicker">Analyze</span>
        <h1>Check the photo before analysis</h1>
        <p>A single leaf in focus, natural color, and an uncluttered background usually give the strongest input.</p>
      </div>
      <div className="preview-panel">
        <img src={image} alt="Selected crop leaf preview" />
        <div className="preview-actions">
          <FilePicker compact />
          <label className="check-row">
            <input type="checkbox" checked={includeWeather} onChange={(e) => setIncludeWeather(e.target.checked)} />
            <span><CloudSun size={19} /><strong>Add today’s local weather</strong><small>Location is requested only for this report.</small></span>
          </label>
          {error && <div className="error-panel"><TriangleAlert /><div><strong>No prediction was created</strong><p>{error}</p></div></div>}
          <button className="primary-button wide" disabled={loading} onClick={runAnalysis}>
            {loading ? <><span className="spinner" /> Analyzing the actual image…</> : <><HeartPulse size={20} /> Analyze leaf</>}
          </button>
          <p className="privacy-note"><ShieldCheck size={15} /> The filename is excluded from the prediction request.</p>
        </div>
      </div>
    </section>
  );
}

function MetricCard({ label, value, helper, tone = "green" }: { label: string; value: string; helper: string; tone?: string }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{helper}</small>
    </article>
  );
}

function ListPanel({ title, items, icon: Icon }: { title: string; items: string[]; icon: typeof Info }) {
  return (
    <article className="report-panel">
      <h3><Icon size={20} /> {title}</h3>
      {items.length ? <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="none-value">None</p>}
    </article>
  );
}

function ResultPage() {
  const navigate = useNavigate();
  const [baseReport] = useState(getActiveReport);
  const [report, setReport] = useState(baseReport);
  const [language, setLanguage] = useState<Language>("en");
  const [languageError, setLanguageError] = useState("");
  const [audioState, setAudioState] = useState<"idle" | "loading" | "playing">("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const stopAudio = () => {
    audioRef.current?.pause();
    if (audioRef.current) audioRef.current.currentTime = 0;
    setAudioState("idle");
  };

  useEffect(() => () => {
    stopAudio();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  if (!report || !baseReport) {
    return (
      <section className="narrow-section empty-state">
        <Activity size={48} />
        <h1>No current report</h1>
        <p>Analyze a leaf or reopen a saved report from History.</p>
        <Link className="primary-button" to="/">Start analysis</Link>
      </section>
    );
  }

  const changeLanguage = async (next: Language) => {
    stopAudio();
    setLanguage(next);
    setLanguageError("");
    try {
      setReport(next === "en" ? baseReport : await translateReport(baseReport, next));
    } catch (caught) {
      setReport(baseReport);
      setLanguage("en");
      setLanguageError(caught instanceof Error ? caught.message : "Translation failed.");
    }
  };

  const playAudio = async () => {
    stopAudio();
    setLanguageError("");
    setAudioState("loading");
    try {
      const blob = await getReportAudio(report, language);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = URL.createObjectURL(blob);
      const audio = new Audio(objectUrlRef.current);
      audioRef.current = audio;
      audio.onended = () => setAudioState("idle");
      audio.onerror = () => {
        setAudioState("idle");
        setLanguageError("Audio playback failed. You can still read the full report.");
      };
      await audio.play();
      setAudioState("playing");
    } catch (caught) {
      setAudioState("idle");
      setLanguageError(caught instanceof Error ? caught.message : "Audio generation failed.");
    }
  };

  const unknown = report.state === "unknown";
  const healthy = report.state === "healthy";
  const stateTitle = unknown ? "Analysis uncertain" : healthy ? "No supported disease detected" : report.disease || "Disease detected";
  const confidenceValue = report.confidence === null ? "Undefined" : `${Math.round(report.confidence * 100)}%`;
  const predictionData = report.topPredictions.map((item) => ({ name: item.label, value: Math.round(item.probability * 100) }));

  return (
    <section className="result-section">
      <div className="result-topline">
        <div>
          <span className="kicker">Analysis result</span>
          <h1>{stateTitle}</h1>
          <p>{new Date(report.createdAt).toLocaleString()} · Model {report.modelVersion}</p>
        </div>
        <button className="secondary-button" onClick={() => navigate("/")}><Camera size={19} /> Scan another leaf</button>
      </div>

      <div className="language-bar">
        <span><Languages size={19} /> Report language</span>
        <div className="language-options">
          {([["en", "English"], ["te", "తెలుగు"], ["hi", "हिन्दी"]] as const).map(([code, label]) => (
            <button key={code} className={language === code ? "selected" : ""} onClick={() => changeLanguage(code)}>{label}</button>
          ))}
        </div>
        <div className="audio-actions">
          <button onClick={playAudio} disabled={audioState !== "idle"}><Play size={17} /> {audioState === "loading" ? "Preparing…" : "Listen"}</button>
          <button onClick={stopAudio} disabled={audioState === "idle"}><Square size={15} /> Stop</button>
        </div>
      </div>
      {languageError && <p className="inline-error"><TriangleAlert size={16} /> {languageError}</p>}

      <div className={`condition-banner ${report.state}`}>
        {unknown ? <TriangleAlert /> : healthy ? <CheckCircle2 /> : <HeartPulse />}
        <div>
          <span>{report.condition}</span>
          <h2>{unknown ? report.uncertaintyReason || "The image did not meet the threshold for a supported prediction." : `${report.crop} · ${healthy ? "Healthy" : report.disease}`}</h2>
        </div>
        {report.thumbnailDataUrl && <img src={report.thumbnailDataUrl} alt="Analyzed leaf" />}
      </div>

      <div className="metrics-grid">
        <MetricCard label="Crop" value={report.crop || "Undefined"} helper="Visual crop prediction" />
        <MetricCard label="AI confidence" value={confidenceValue} helper="Calibrated model confidence" tone="blue" />
        <MetricCard label="Visible disease rate" value={report.diseaseRate === null ? (unknown ? "Undefined" : "Unable to estimate") : `${report.diseaseRate}%`} helper="Visible leaf area only" tone="amber" />
        <MetricCard label="Severity" value={report.severity || (unknown ? "Undefined" : "None")} helper="Independent of confidence" tone="rose" />
      </div>

      {!unknown && (
        <>
          <div className="report-grid">
            <ListPanel title="Observed in this photo" items={report.observedSymptoms} icon={Activity} />
            <ListPanel title="Typical symptoms to check" items={healthy ? [] : report.typicalSymptoms} icon={Info} />
            <ListPanel title="Known causes or contributors" items={healthy ? [] : report.causes} icon={TriangleAlert} />
            <ListPanel title={healthy ? "Crop-care recommendations" : "Recommended next steps"} items={report.recommendations} icon={CheckCircle2} />
          </div>

          <div className="visual-grid">
            <article className="chart-card">
              <h3>Top model predictions</h3>
              <p>Original model probabilities; the displayed items are not renormalized.</p>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={predictionData} layout="vertical" margin={{ left: 5, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} unit="%" />
                    <YAxis type="category" dataKey="name" width={116} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(value) => [`${value}%`, "Probability"]} />
                    <Bar dataKey="value" fill="#2e7d55" radius={[0, 7, 7, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </article>
            <article className="chart-card status-visual">
              <h3>Visible leaf health</h3>
              {report.healthScore === null ? (
                <div className="not-measured"><Gauge /><strong>Unable to estimate</strong><p>A validated leaf-area measurement was not available for this result.</p></div>
              ) : (
                <div className="score-ring" style={{ "--score": report.healthScore } as React.CSSProperties}>
                  <span>{report.healthScore}%</span><small>visibly unaffected area</small>
                </div>
              )}
            </article>
          </div>
        </>
      )}

      {unknown && (
        <div className="unknown-help">
          <h3>Try another photograph</h3>
          <ul><li>Use one leaf in natural light.</li><li>Keep the leaf sharp and fill most of the frame.</li><li>Photograph both sides if symptoms are unclear.</li></ul>
        </div>
      )}

      <article className="weather-card">
        <div><CloudSun size={30} /><span>Weather recorded with this report</span></div>
        {report.weather ? (
          <div className="weather-values">
            <strong>{report.weather.location}</strong>
            <span>{report.weather.temperatureC ?? "—"}°C</span>
            <span>{report.weather.humidityPercent ?? "—"}% humidity</span>
            <span>{report.weather.precipitationMm ?? "—"} mm rain</span>
            <small>{report.weather.description} · {new Date(report.weather.observedAt).toLocaleString()}</small>
          </div>
        ) : <p>Weather was unavailable or location access was not provided. The image analysis remains independent.</p>}
      </article>
    </section>
  );
}

function DashboardPage() {
  const [reports] = useState(loadHistory);
  const summary = useMemo(() => {
    const counts = { healthy: 0, diseased: 0, unknown: 0 };
    reports.forEach((report) => { counts[report.state] += 1; });
    return counts;
  }, [reports]);
  const chartData = [
    { name: "Healthy", value: summary.healthy, color: "#2e7d55" },
    { name: "Diseased", value: summary.diseased, color: "#d47d2b" },
    { name: "Unknown", value: summary.unknown, color: "#718078" },
  ];
  const measured = reports.filter((report) => report.healthScore !== null).slice(0, 8).reverse();

  return (
    <section className="page-section">
      <div className="page-heading"><span className="kicker">Dashboard</span><h1>Your browser’s scan overview</h1><p>These summaries describe only the reports saved on this device.</p></div>
      <div className="metrics-grid dashboard-metrics">
        <MetricCard label="Saved scans" value={String(reports.length)} helper="Up to 30 compact reports" />
        <MetricCard label="Healthy" value={String(summary.healthy)} helper="No supported disease detected" />
        <MetricCard label="Diseased" value={String(summary.diseased)} helper="Accepted disease predictions" tone="amber" />
        <MetricCard label="Unknown" value={String(summary.unknown)} helper="Rejected or uncertain images" tone="blue" />
      </div>
      {reports.length ? (
        <div className="visual-grid dashboard-charts">
          <article className="chart-card">
            <h3>Result mix</h3><p>Local reports only; this is not regional prevalence.</p>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3}>
                    {chartData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="legend">{chartData.map((item) => <span key={item.name}><i style={{ background: item.color }} />{item.name}: {item.value}</span>)}</div>
          </article>
          <article className="chart-card">
            <h3>Measured visible health</h3><p>Reports without a valid score are excluded.</p>
            {measured.length ? (
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={measured.map((item) => ({ name: item.crop || "Unknown", score: item.healthScore }))}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} unit="%" />
                    <Tooltip />
                    <Bar dataKey="score" fill="#75a88d" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : <div className="not-measured"><Gauge /><strong>No validated health scores yet</strong><p>Scores appear only when visible leaf area was measured.</p></div>}
          </article>
        </div>
      ) : <EmptyHistory />}
    </section>
  );
}

function EmptyHistory() {
  return (
    <div className="empty-history"><HistoryIcon size={42} /><h3>No saved analyses yet</h3><p>Your reports will appear here after a real analysis completes.</p><Link className="primary-button" to="/">Analyze a leaf</Link></div>
  );
}

function HistoryPage() {
  const navigate = useNavigate();
  const [reports, setReports] = useState(loadHistory);
  const remove = (id: string) => { deleteReport(id); setReports(loadHistory()); };
  const clear = () => {
    if (window.confirm("Clear all Crop Disease AI history from this browser?")) {
      clearHistory();
      setReports([]);
    }
  };
  const open = (report: AnalysisReport) => {
    setActiveReport(report);
    navigate("/result");
  };

  return (
    <section className="page-section">
      <div className="page-heading history-heading">
        <div><span className="kicker">History</span><h1>Saved leaf reports</h1><p>Stored only in this browser. Clearing site data removes them.</p></div>
        {reports.length > 0 && <button className="danger-button" onClick={clear}><Trash2 size={18} /> Clear all</button>}
      </div>
      {!reports.length ? <EmptyHistory /> : (
        <div className="history-list">
          {reports.map((report) => (
            <article className="history-card" key={report.id}>
              {report.thumbnailDataUrl ? <img src={report.thumbnailDataUrl} alt="" /> : <div className="history-placeholder"><Leaf /></div>}
              <div className="history-info">
                <span className={`status-pill ${report.state}`}>{report.condition}</span>
                <h3>{report.crop || "Undefined"}</h3>
                <p>{report.disease || (report.state === "healthy" ? "No supported disease detected" : "Undefined")}</p>
                <small><Clock3 size={14} /> {new Date(report.createdAt).toLocaleString()}</small>
              </div>
              <div className="history-actions">
                <button onClick={() => open(report)}>Open <ChevronRight size={17} /></button>
                <button aria-label="Delete report" onClick={() => remove(report.id)}><Trash2 size={17} /></button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function CropsPage() {
  return (
    <section className="page-section">
      <div className="page-heading"><span className="kicker">Coverage</span><h1>Target crops and baseline taxonomy</h1><p>This is the PlantVillage baseline being evaluated. A class becomes advertised support only after the production model passes validation.</p></div>
      <div className="coverage-notice"><TriangleAlert /><div><strong>Coverage is still under validation</strong><p>The table exposes known dataset gaps instead of hiding them. Blueberry, Raspberry, and Soybean have no disease class in the baseline; Orange and Squash have no healthy class.</p></div></div>
      <div className="crops-grid">
        {cropCoverage.map((item) => (
          <article className="crop-card" key={item.crop}>
            <div className="crop-card-top"><span><Sprout /></span><div><h3>{item.crop}</h3><small>{item.healthy ? "Healthy class available" : "Healthy class missing"}</small></div></div>
            <div className="disease-tags">
              {item.diseases.length ? item.diseases.map((disease) => <span key={disease}>{disease}</span>) : <em>No disease class in baseline dataset</em>}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function AboutPage() {
  return (
    <section className="page-section about-page">
      <div className="page-heading"><span className="kicker">About</span><h1>Useful screening with visible limits</h1><p>Crop Disease AI is designed to help a farmer understand a leaf photograph and decide what to inspect next.</p></div>
      <div className="about-grid">
        <article><HeartPulse /><h3>What it analyzes</h3><p>The production service decodes the uploaded image and evaluates its pixels. Filenames and user labels are excluded from model input.</p></article>
        <article><ShieldCheck /><h3>How uncertainty works</h3><p>Unsupported plants, poor photographs, and unfamiliar conditions must be returned as Undefined and Unknown instead of being forced into the closest class.</p></article>
        <article><Gauge /><h3>How scores differ</h3><p>Confidence describes the classifier. Disease rate describes measured visible leaf damage. One value is never substituted for the other.</p></article>
        <article><CloudSun /><h3>Weather context</h3><p>Weather comes from the chosen location and time. It adds context but does not prove what caused a disease.</p></article>
        <article><Languages /><h3>Language and voice</h3><p>The complete report can be translated and spoken in English, Telugu, or Hindi using server-generated audio.</p></article>
        <article><Info /><h3>Practical limit</h3><p>A photograph cannot confirm laboratory pathogens, soil conditions, whole-field prevalence, or treatment suitability for every location.</p></article>
      </div>
      <div className="privacy-card">
        <div><ShieldCheck size={28} /><h3>Privacy and local history</h3></div>
        <p>Compact reports and thumbnails are saved in this browser for History and Dashboard. They do not synchronize across devices. The deployed backend’s image retention policy will be documented before public release.</p>
      </div>
    </section>
  );
}

function ReportRedirect() {
  const { id } = useParams();
  const navigate = useNavigate();
  useEffect(() => {
    const report = loadHistory().find((item) => item.id === id);
    if (report) setActiveReport(report);
    navigate(report ? "/result" : "/history", { replace: true });
  }, [id, navigate]);
  return null;
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/result" element={<ResultPage />} />
        <Route path="/result/:id" element={<ReportRedirect />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/crops" element={<CropsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<HomePage />} />
      </Routes>
    </Layout>
  );
}

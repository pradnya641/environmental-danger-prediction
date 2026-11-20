
import { useEffect, useRef, useState, useCallback } from "react";
import "./LiveMonitoring.css";

const API = "http://127.0.0.1:8000";

// ─── Configurable intervals ──────────────────────────────────────────────────
const ANALYSIS_INTERVAL_MS = 6000;   // capture interval (balanced for Gemini latency)
const SAVE_COOLDOWN_MS = 45000;      // min gap between DB persistence writes (45s)

// ─── Types ───────────────────────────────────────────────────────────────────

export type BBox = {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
};

export type Hazard = {
  hazard: string;
  category: string;
  visual_evidence: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "UNCERTAIN" | string;
  confidence: number;
  observation_type: string;
  bbox?: BBox | null;
};

export type AnalysisResult = {
  scene_analysis: {
    scene_type: string;
    scene_description: string;
    potential_hazards: Hazard[];
    spatial_relations: string[];
  };
  risk_result: {
    risk_level: "LOW" | "MEDIUM" | "HIGH" | "UNCERTAIN" | string;
    status: string;
    explanation: string;
    hazards_for_review: Hazard[];
    uncertain_hazards: Hazard[];
    recommendations: string[];
  };
  error: string | null;
  incident_id?: number;
};

// Track a unique real-world hazard tracked across consecutive frames
export type TrackedHazard = {
  id: string;
  category: string;
  hazardName: string;
  severity: string;
  confidence: number;
  visualEvidence: string;
  firstDetected: Date;
  lastDetected: Date;
  detectionCount: number;
  latestBbox?: BBox | null;
  isLocalized: boolean;
  incidentId?: number;
};

export type SystemStatus = {
  camera: "connected" | "disconnected" | "requesting";
  backend: "connected" | "disconnected" | "unknown";
  ai: "running" | "idle";
  lastAnalyzedAt: Date | null;
};

// ─── Color maps ───────────────────────────────────────────────────────────────

const SEVERITY_COLOR: Record<string, string> = {
  HIGH: "#e53e3e",
  MEDIUM: "#dd6b20",
  LOW: "#d69e2e",
  UNCERTAIN: "#805ad5",
};

const RISK_BORDER: Record<string, string> = {
  HIGH: "#e53e3e",
  MEDIUM: "#dd6b20",
  LOW: "#d69e2e",
  UNCERTAIN: "#805ad5",
};

// ─── Bbox validation ──────────────────────────────────────────────────────────
export function isBboxValid(b: BBox | null | undefined): b is BBox {
  if (!b) return false;
  if (typeof b.x_min !== "number" || typeof b.y_min !== "number") return false;
  if (typeof b.x_max !== "number" || typeof b.y_max !== "number") return false;
  if (b.x_min < 0 || b.y_min < 0 || b.x_max > 1 || b.y_max > 1) return false;
  if (b.x_min >= b.x_max || b.y_min >= b.y_max) return false;
  const area = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  return area >= 0.001; // reject micro-boxes (< 0.1% of frame area)
}

// ─── Spatial IoU (Intersection over Union) ──────────────────────────────────
export function calculateIoU(a: BBox, b: BBox): number {
  const xLeft = Math.max(a.x_min, b.x_min);
  const yTop = Math.max(a.y_min, b.y_min);
  const xRight = Math.min(a.x_max, b.x_max);
  const yBottom = Math.min(a.y_max, b.y_max);

  if (xRight <= xLeft || yBottom <= yTop) return 0.0;

  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  const areaA = (a.x_max - a.x_min) * (a.y_max - a.y_min);
  const areaB = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  const unionArea = areaA + areaB - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0.0;
}

// ─── Canvas helper ───────────────────────────────────────────────────────────
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function LiveMonitoring() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);           // hidden capture
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);   // visible overlay
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAnalyzingRef = useRef(false);
  const lastSavedAtRef = useRef<number>(0);
  const lastResultRef = useRef<AnalysisResult | null>(null);

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("");

  const [latestResult, setLatestResult] = useState<AnalysisResult | null>(null);

  // Tracked distinct hazards
  const [trackedHazards, setTrackedHazards] = useState<TrackedHazard[]>([]);

  // Detailed statistics
  const [stats, setStats] = useState({
    totalFramesAnalyzed: 0,
    totalFrameDetections: 0,
    uniqueHazards: 0,
    localizedHazards: 0,
    highRisk: 0,
    mediumRisk: 0,
    lowRisk: 0,
  });

  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    camera: "disconnected",
    backend: "unknown",
    ai: "idle",
    lastAnalyzedAt: null,
  });

  const [cameraError, setCameraError] = useState<string>("");

  // ── Camera enumeration ────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const videoDevices = devices.filter((d) => d.kind === "videoinput");
      setCameras(videoDevices);
      if (videoDevices.length > 0) setSelectedCamera(videoDevices[0].deviceId);
    }).catch(() => setCameraError("Could not enumerate cameras."));
  }, []);

  // ── Backend health check ──────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setSystemStatus((s) => ({ ...s, backend: r.ok ? "connected" : "disconnected" })))
      .catch(() => setSystemStatus((s) => ({ ...s, backend: "disconnected" })));
  }, []);

  // ── Draw overlay ──────────────────────────────────────────────────────────
  const drawOverlay = useCallback(
    (result: AnalysisResult | null, videoEl: HTMLVideoElement, overlayEl: HTMLCanvasElement) => {
      const ctx = overlayEl.getContext("2d");
      if (!ctx) return;

      const rect = videoEl.getBoundingClientRect();
      const W = rect.width || videoEl.clientWidth;
      const H = rect.height || videoEl.clientHeight;
      overlayEl.width = W;
      overlayEl.height = H;

      ctx.clearRect(0, 0, W, H);
      if (!result || result.error) return;

      const videoW = videoEl.videoWidth || W;
      const videoH = videoEl.videoHeight || H;
      const videoRatio = videoW / videoH;
      const containerRatio = W / H;

      // Handle object-fit: cover mapping
      let renderW = W;
      let renderH = H;
      let offsetX = 0;
      let offsetY = 0;

      if (containerRatio > videoRatio) {
        renderW = W;
        renderH = W / videoRatio;
        offsetY = (H - renderH) / 2;
      } else {
        renderH = H;
        renderW = H * videoRatio;
        offsetX = (W - renderW) / 2;
      }

      const riskLevel = result.risk_result?.risk_level ?? "LOW";
      const allHazards: Hazard[] = [
        ...(result.risk_result?.hazards_for_review ?? []),
        ...(result.risk_result?.uncertain_hazards ?? []),
      ];

      if (allHazards.length === 0) return;

      const localizedHazards = allHazards.filter((h) => isBboxValid(h.bbox));
      const unlocatedHazards = allHazards.filter((h) => !isBboxValid(h.bbox));

      // ── Per-hazard bounding boxes (only when valid coords exist) ───────────
      if (localizedHazards.length > 0) {
        localizedHazards.forEach((h) => {
          const bbox = h.bbox!;
          const color = SEVERITY_COLOR[h.severity] ?? "#805ad5";

          const rawPx = offsetX + bbox.x_min * renderW;
          const rawPy = offsetY + bbox.y_min * renderH;
          const rawPw = (bbox.x_max - bbox.x_min) * renderW;
          const rawPh = (bbox.y_max - bbox.y_min) * renderH;

          // Clip to canvas visible boundaries
          const px = Math.max(0, rawPx);
          const py = Math.max(0, rawPy);
          const pw = Math.min(W - px, rawPw - (px - rawPx));
          const ph = Math.min(H - py, rawPh - (py - rawPy));

          if (pw <= 0 || ph <= 0) return;

          // Glow effect
          ctx.shadowColor = color;
          ctx.shadowBlur = 8;

          // Translucent fill
          ctx.fillStyle = color + "22";
          ctx.fillRect(px, py, pw, ph);

          // Solid border
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.shadowBlur = 0;
          ctx.strokeRect(px, py, pw, ph);

          // Top label pill: displays Category, Confidence & Severity
          const cat = (h.category || "HAZARD").toUpperCase();
          const sev = (h.severity || "LOW").toUpperCase();
          const label = `${h.hazard} [${cat} • ${sev}] • ${Math.round(h.confidence * 100)}%`;
          ctx.font = "bold 11px DM Sans, sans-serif";
          const labelW = ctx.measureText(label).width + 16;
          const labelH = 20;
          const labelX = Math.min(px, Math.max(4, W - labelW - 4));
          const labelY = py > labelH + 4 ? py - labelH - 3 : py + 3;

          ctx.fillStyle = color;
          roundRect(ctx, labelX, labelY, labelW, labelH, 4);
          ctx.fill();

          ctx.fillStyle = "white";
          ctx.fillText(label, labelX + 8, labelY + 14);
        });
      }

      // ── Fallback frame border when NO hazard has valid coords ─────────────
      if (localizedHazards.length === 0) {
        const borderColor = RISK_BORDER[riskLevel] ?? "#805ad5";
        const t = 4;
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = t;
        ctx.shadowColor = borderColor;
        ctx.shadowBlur = 8;
        ctx.strokeRect(t / 2, t / 2, W - t, H - t);
        ctx.shadowBlur = 0;
      }

      // ── Risk-level corner badge ───────────────────────────────────────────
      const badgeColor = RISK_BORDER[riskLevel] ?? "#805ad5";
      const badgeText = `${riskLevel} RISK`;
      ctx.font = "bold 11px DM Sans, sans-serif";
      const bw = ctx.measureText(badgeText).width + 18;
      const bh = 24;
      ctx.fillStyle = badgeColor;
      roundRect(ctx, 10, 10, bw, bh, 5);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.fillText(badgeText, 19, 10 + 16);

      // ── Unlocated hazard count badge ──────────────────────────────────────
      if (unlocatedHazards.length > 0) {
        const uText = `${unlocatedHazards.length} not localized`;
        ctx.font = "11px DM Sans, sans-serif";
        const uw = ctx.measureText(uText).width + 16;
        const uh = 22;
        const ux = W - uw - 10;
        ctx.fillStyle = "rgba(20, 36, 58, 0.75)";
        roundRect(ctx, ux, 10, uw, uh, 5);
        ctx.fill();
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(uText, ux + 8, 10 + 15);
      }
    },
    []
  );

  // ── ResizeObserver: redraw overlay when video element is resized ──────────
  useEffect(() => {
    const videoEl = videoRef.current;
    const overlayEl = overlayCanvasRef.current;
    if (!videoEl || !overlayEl) return;

    const observer = new ResizeObserver(() => {
      if (lastResultRef.current) {
        drawOverlay(lastResultRef.current, videoEl, overlayEl);
      }
    });
    observer.observe(videoEl);
    return () => observer.disconnect();
  }, [drawOverlay]);

  // ── Deduplicate & track hazards across consecutive frames ─────────────────
  const updateTrackedHazards = useCallback(
    (newHazards: Hazard[], incidentId?: number) => {
      setTrackedHazards((prevTracked) => {
        const updated = [...prevTracked];
        const now = new Date();

        newHazards.forEach((nh) => {
          const nhHasBox = isBboxValid(nh.bbox);
          const nhCat = (nh.category || "OTHER").toUpperCase();
          const nhName = (nh.hazard || "").toLowerCase().trim();

          // Try matching with an existing tracked hazard
          let matchedIdx = -1;

          for (let i = 0; i < updated.length; i++) {
            const existing = updated[i];
            const exCat = existing.category.toUpperCase();
            const exName = existing.hazardName.toLowerCase().trim();

            if (nhHasBox && existing.latestBbox && isBboxValid(existing.latestBbox)) {
              // Spatial matching: same category or name + significant IoU overlap
              const iou = calculateIoU(nh.bbox!, existing.latestBbox);
              if (iou >= 0.30 || (iou >= 0.15 && (nhCat === exCat || nhName === exName))) {
                matchedIdx = i;
                break;
              }
            } else if (!nhHasBox && !existing.isLocalized) {
              // Non-localized matching: match if same category and same core hazard name
              if (nhCat === exCat && nhName === exName) {
                // If detected recently (within last 35 seconds), treat as same persistent hazard
                const diffSec = (now.getTime() - existing.lastDetected.getTime()) / 1000;
                if (diffSec < 35) {
                  matchedIdx = i;
                  break;
                }
              }
            }
          }

          if (matchedIdx >= 0) {
            // Update existing tracked hazard
            const target = updated[matchedIdx];
            updated[matchedIdx] = {
              ...target,
              lastDetected: now,
              detectionCount: target.detectionCount + 1,
              severity: nh.severity,
              confidence: Math.max(target.confidence, nh.confidence),
              visualEvidence: nh.visual_evidence || target.visualEvidence,
              latestBbox: nhHasBox ? nh.bbox : target.latestBbox,
              isLocalized: target.isLocalized || nhHasBox,
              incidentId: incidentId || target.incidentId,
            };
          } else {
            // Register a newly detected unique hazard
            const newTracked: TrackedHazard = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              category: nh.category || "OTHER",
              hazardName: nh.hazard,
              severity: nh.severity,
              confidence: nh.confidence,
              visualEvidence: nh.visual_evidence,
              firstDetected: now,
              lastDetected: now,
              detectionCount: 1,
              latestBbox: nhHasBox ? nh.bbox : null,
              isLocalized: nhHasBox,
              incidentId,
            };
            updated.unshift(newTracked);
          }
        });

        // Keep maximum 30 unique events in history
        return updated.slice(0, 30);
      });
    },
    []
  );

  // ── Recompute statistics whenever tracked hazards or frames update ────────
  useEffect(() => {
    let high = 0;
    let med = 0;
    let low = 0;
    let localized = 0;

    trackedHazards.forEach((th) => {
      const sev = (th.severity || "").toUpperCase();
      if (sev === "HIGH") high++;
      else if (sev === "MEDIUM") med++;
      else if (sev === "LOW") low++;

      if (th.isLocalized) localized++;
    });

    setStats((prev) => ({
      ...prev,
      uniqueHazards: trackedHazards.length,
      localizedHazards: localized,
      highRisk: high,
      mediumRisk: med,
      lowRisk: low,
    }));
  }, [trackedHazards]);

  // ── Analyse one frame ─────────────────────────────────────────────────────
  const analyzeFrame = useCallback(async () => {
    if (isAnalyzingRef.current) return;
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    if (video.readyState < 2) return;

    isAnalyzingRef.current = true;
    setSystemStatus((s) => ({ ...s, ai: "running" }));

    try {
      // Capture JPEG from current video frame
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0);

      const blob: Blob | null = await new Promise((res) =>
        canvas.toBlob(res, "image/jpeg", 0.82)
      );
      if (!blob) return;

      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      const response = await fetch(`${API}/api/analyze/frame`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.warn(`Analysis API returned status ${response.status}:`, errorText);
        setSystemStatus((s) => ({ ...s, backend: "connected", ai: "idle" }));
        return;
      }

      const result: AnalysisResult = await response.json();

      lastResultRef.current = result;
      setLatestResult(result);
      setSystemStatus((s) => ({
        ...s,
        backend: "connected",
        ai: "idle",
        lastAnalyzedAt: new Date(),
      }));

      // Draw bounding boxes on canvas
      if (overlayCanvasRef.current) {
        drawOverlay(result, video, overlayCanvasRef.current);
      }

      const allHazards: Hazard[] = [
        ...(result.risk_result?.hazards_for_review ?? []),
        ...(result.risk_result?.uncertain_hazards ?? []),
      ];

      // Update frame-level counters
      setStats((prev) => ({
        ...prev,
        totalFramesAnalyzed: prev.totalFramesAnalyzed + 1,
        totalFrameDetections: prev.totalFrameDetections + allHazards.length,
      }));

      if (allHazards.length > 0) {
        const riskLevel = result.risk_result?.risk_level ?? "UNCERTAIN";
        let savedIncidentId: number | undefined;

        // DB persistence with 60s cooldown
        const shouldSave =
          (riskLevel === "HIGH" || riskLevel === "MEDIUM") &&
          Date.now() - lastSavedAtRef.current > SAVE_COOLDOWN_MS;

        if (shouldSave) {
          try {
            const saveForm = new FormData();
            saveForm.append("file", blob, "frame.jpg");
            const saveRes = await fetch(`${API}/api/analyze/image`, {
              method: "POST",
              body: saveForm,
            });
            if (saveRes.ok) {
              const saved: AnalysisResult = await saveRes.json();
              savedIncidentId = saved.incident_id;
              lastSavedAtRef.current = Date.now();
            }
          } catch {
            // Non-critical persistence failure
          }
        }

        // Deduplicate & track unique hazard events
        updateTrackedHazards(allHazards, savedIncidentId);
      } else {
        // Clear overlay when no hazards detected
        if (overlayCanvasRef.current) {
          const oc = overlayCanvasRef.current;
          oc.getContext("2d")?.clearRect(0, 0, oc.width, oc.height);
        }
      }
    } catch (err) {
      console.error("Frame analysis error:", err);
      setSystemStatus((s) => ({ ...s, backend: "disconnected", ai: "idle" }));
    } finally {
      isAnalyzingRef.current = false;
    }
  }, [drawOverlay, updateTrackedHazards]);

  // ── Start monitoring ──────────────────────────────────────────────────────
  const startMonitoring = useCallback(async () => {
    setCameraError("");
    setSystemStatus((s) => ({ ...s, camera: "requesting" }));

    try {
      const constraints: MediaStreamConstraints = {
        video: selectedCamera
          ? { deviceId: { exact: selectedCamera }, width: 1280, height: 720 }
          : { width: 1280, height: 720 },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setSystemStatus((s) => ({ ...s, camera: "connected" }));
      setIsMonitoring(true);

      intervalRef.current = setInterval(analyzeFrame, ANALYSIS_INTERVAL_MS);
      setTimeout(analyzeFrame, 1500);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Could not access camera.";
      setCameraError(msg);
      setSystemStatus((s) => ({ ...s, camera: "disconnected" }));
    }
  }, [selectedCamera, analyzeFrame]);

  // ── Stop monitoring ───────────────────────────────────────────────────────
  const stopMonitoring = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }

    lastResultRef.current = null;
    setIsMonitoring(false);
    setLatestResult(null);
    setSystemStatus((s) => ({ ...s, camera: "disconnected", ai: "idle" }));
  }, []);

  useEffect(() => () => stopMonitoring(), [stopMonitoring]);

  // ── Derived values ────────────────────────────────────────────────────────
  const allActiveHazards: Hazard[] = latestResult
    ? [
        ...(latestResult.risk_result?.hazards_for_review ?? []),
        ...(latestResult.risk_result?.uncertain_hazards ?? []),
      ]
    : [];

  const riskLevel = latestResult?.risk_result?.risk_level ?? null;
  const localizedCount = allActiveHazards.filter((h) => isBboxValid(h.bbox)).length;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="live-page">
      {/* ── TOP: video + side panel ──────────────────────────────────────── */}
      <div className="live-top-grid">
        {/* Camera panel */}
        <div className="camera-panel">
          <div className="camera-panel-heading">
            <div>
              <h2>Live Camera Feed</h2>
              <p>Webcam capture • Gemini AI analysis every {ANALYSIS_INTERVAL_MS / 1000}s</p>
            </div>
            <div className="camera-controls">
              {cameras.length > 1 && (
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  disabled={isMonitoring}
                  className="camera-select"
                >
                  {cameras.map((cam, i) => (
                    <option key={cam.deviceId} value={cam.deviceId}>
                      {cam.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              )}
              {!isMonitoring ? (
                <button className="btn-start" onClick={startMonitoring}>▶ Start Monitoring</button>
              ) : (
                <button className="btn-stop" onClick={stopMonitoring}>■ Stop Monitoring</button>
              )}
            </div>
          </div>

          {cameraError && <div className="live-error">{cameraError}</div>}

          <div className="video-wrapper">
            <video ref={videoRef} className="live-video" autoPlay muted playsInline />

            {/* Responsive overlay canvas */}
            <canvas ref={overlayCanvasRef} className="overlay-canvas" />

            {/* Hidden capture canvas */}
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {!isMonitoring && (
              <div className="video-placeholder">
                <div className="placeholder-icon">📷</div>
                <p>Camera not active</p>
                <span>Click "Start Monitoring" to begin</span>
              </div>
            )}

            {isMonitoring && (
              <div className="live-badge">
                <span className="live-dot" /> LIVE
              </div>
            )}

            {systemStatus.ai === "running" && (
              <div className="analyzing-badge">AI Analyzing…</div>
            )}
          </div>

          {/* Legend */}
          {isMonitoring && (
            <div className="bbox-legend">
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "#e53e3e" }} />
                High Risk
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "#dd6b20" }} />
                Medium Risk
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "#d69e2e" }} />
                Low Risk
              </div>
              {localizedCount > 0 ? (
                <span className="legend-localized">
                  {localizedCount} of {allActiveHazards.length} spatially localized
                </span>
              ) : allActiveHazards.length > 0 ? (
                <span className="legend-unlocated">
                  {allActiveHazards.length} hazard{allActiveHazards.length > 1 ? "s" : ""} detected (Not localized)
                </span>
              ) : null}
            </div>
          )}
        </div>

        {/* Detection info panel */}
        <div className="info-panel">
          <div className={`risk-header risk-${(riskLevel ?? "none").toLowerCase()}`}>
            <div className="risk-header-label">Current Risk Level</div>
            <div className="risk-header-value">{riskLevel ?? "—"}</div>
            {riskLevel && (
              <div className="risk-header-sub">
                {allActiveHazards.length} hazard{allActiveHazards.length !== 1 ? "s" : ""} in current frame
                {localizedCount > 0 ? ` • ${localizedCount} localized` : ` • 0 localized`}
              </div>
            )}
          </div>

          <div className="info-section">
            <div className="info-section-title">
              Current Frame Hazards
              <span className="count-badge">{allActiveHazards.length}</span>
            </div>

            {allActiveHazards.length === 0 ? (
              <p className="live-empty">
                {isMonitoring ? "No hazards detected in current frame." : "Start monitoring to begin analysis."}
              </p>
            ) : (
              <div className="hazard-cards">
                {allActiveHazards.map((h, i) => (
                  <HazardCard key={i} hazard={h} />
                ))}
              </div>
            )}
          </div>

          {latestResult?.scene_analysis?.scene_type && (
            <div className="info-section">
              <div className="info-section-title">Scene Type</div>
              <div className="scene-info">
                <strong>{latestResult.scene_analysis.scene_type}</strong>
                <p>{latestResult.scene_analysis.scene_description}</p>
              </div>
            </div>
          )}

          {(latestResult?.risk_result?.recommendations ?? []).length > 0 && (
            <div className="info-section">
              <div className="info-section-title">Safety Recommendations</div>
              <ul className="recommendation-list">
                {latestResult!.risk_result.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* ── BOTTOM: stats + history + system status ───────────────────────── */}
      <div className="live-bottom-grid">
        <div className="live-stats-panel panel">
          <div className="panel-heading">
            <div>
              <h2>Monitoring Statistics</h2>
              <p>Session-wide frame detections & deduplicated unique hazards</p>
            </div>
          </div>
          <div className="stat-grid-6">
            <StatTile label="Frame Detections" value={stats.totalFrameDetections} sub="Total across all frames" color="blue" />
            <StatTile label="Unique Hazards" value={stats.uniqueHazards} sub="Deduplicated distinct hazards" color="purple" />
            <StatTile label="Localized Hazards" value={stats.localizedHazards} sub="With visual bounding boxes" color="green" />
            <StatTile label="High-Risk Hazards" value={stats.highRisk} sub="Critical unique threats" color="red" />
            <StatTile label="Medium-Risk Hazards" value={stats.mediumRisk} sub="Moderate unique threats" color="orange" />
            <StatTile label="Low-Risk Hazards" value={stats.lowRisk} sub="Low / minor unique issues" color="yellow" />
          </div>
        </div>

        <div className="live-history-panel panel">
          <div className="panel-heading">
            <div>
              <h2>Unique Hazard History</h2>
              <p>{trackedHazards.length} distinct hazard event{trackedHazards.length === 1 ? "" : "s"} tracked</p>
            </div>
          </div>
          {trackedHazards.length === 0 ? (
            <p className="live-empty">No hazard events recorded yet.</p>
          ) : (
            <div className="history-list">
              {trackedHazards.map((th) => (
                <TrackedHazardRow key={th.id} item={th} />
              ))}
            </div>
          )}
        </div>

        <div className="live-system-panel panel">
          <div className="panel-heading">
            <div>
              <h2>System Status</h2>
              <p>Live health & telemetry</p>
            </div>
          </div>
          <StatusRow label="Camera" status={systemStatus.camera} ok={systemStatus.camera === "connected"} warn={systemStatus.camera === "requesting"} />
          <StatusRow label="Backend" status={systemStatus.backend} ok={systemStatus.backend === "connected"} warn={systemStatus.backend === "unknown"} />
          <StatusRow label="AI Analysis" status={systemStatus.ai} ok={systemStatus.ai === "idle"} warn={systemStatus.ai === "running"} />
          <div className="status-row">
            <span className="status-label">Frames Analyzed</span>
            <span className="status-value">{stats.totalFramesAnalyzed}</span>
          </div>
          <div className="status-row">
            <span className="status-label">Last Analysis</span>
            <span className="status-value">
              {systemStatus.lastAnalyzedAt
                ? systemStatus.lastAnalyzedAt.toLocaleTimeString("en-IN")
                : "—"}
            </span>
          </div>
          <div className="status-row">
            <span className="status-label">Monitoring State</span>
            <span className={`status-value ${isMonitoring ? "status-ok" : "status-off"}`}>
              {isMonitoring ? "ACTIVE" : "INACTIVE"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function HazardCard({ hazard }: { hazard: Hazard }) {
  const color = SEVERITY_COLOR[hazard.severity] ?? "#805ad5";
  const hasBox = isBboxValid(hazard.bbox);

  return (
    <div className="hazard-card" style={{ borderLeftColor: color }}>
      <div className="hazard-card-header">
        <strong style={{ color }}>{hazard.hazard}</strong>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {hasBox ? (
            <span className="bbox-tag" title="Spatially localized with bounding box">
              📍 Localized
            </span>
          ) : (
            <span className="unlocated-tag" title="Visual region coordinates unavailable">
              Not localized
            </span>
          )}
          <span className="severity-badge" style={{ background: color + "22", color }}>
            {hazard.severity}
          </span>
        </div>
      </div>
      <div className="hazard-card-meta">
        <span>Category: {hazard.category}</span>
        <span>Confidence: {Math.round(hazard.confidence * 100)}%</span>
        <span>Type: {hazard.observation_type}</span>
      </div>
      {hazard.visual_evidence && (
        <div className="hazard-evidence">{hazard.visual_evidence}</div>
      )}
      {!hasBox && (
        <div className="no-bbox-note">Not localized — no reliable bounding coordinates returned</div>
      )}
    </div>
  );
}

function TrackedHazardRow({ item }: { item: TrackedHazard }) {
  const color = RISK_BORDER[item.severity] ?? "#805ad5";
  const firstTime = item.firstDetected.toLocaleTimeString("en-IN");
  const lastTime = item.lastDetected.toLocaleTimeString("en-IN");

  return (
    <div className="history-row">
      <div className="history-dot" style={{ background: color }} />
      <div className="history-content">
        <div className="history-top">
          <span className="history-hazards" title={item.hazardName}>
            {item.hazardName}
          </span>
          <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
            {item.isLocalized ? (
              <span className="history-loc-badge">📍 Boxed</span>
            ) : (
              <span className="history-unloc-badge">Unlocated</span>
            )}
            <span className="history-risk" style={{ color, background: color + "1a" }}>
              {item.severity}
            </span>
          </div>
        </div>
        <div className="history-sub-meta">
          <span>Cat: {item.category}</span>
          <span>• Seen in {item.detectionCount} frame{item.detectionCount > 1 ? "s" : ""}</span>
          {item.incidentId && <span className="history-saved"> • Saved #{item.incidentId}</span>}
        </div>
        <div className="history-time">
          First: {firstTime} {firstTime !== lastTime && `| Last: ${lastTime}`}
        </div>
      </div>
    </div>
  );
}

function StatTile({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: number;
  sub?: string;
  color: string;
}) {
  return (
    <div className={`stat-tile stat-${color}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function StatusRow({ label, status, ok, warn }: { label: string; status: string; ok: boolean; warn: boolean }) {
  const cls = ok ? "status-ok" : warn ? "status-warn" : "status-off";
  return (
    <div className="status-row">
      <span className="status-label">{label}</span>
      <span className={`status-value ${cls}`}>
        <span className={`status-indicator ${cls}`} />
        {status.toUpperCase()}
      </span>
    </div>
  );
}

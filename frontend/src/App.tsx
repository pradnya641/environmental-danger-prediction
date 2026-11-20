
import { useEffect, useState } from "react";
import "./App.css";
import LiveMonitoring from "./LiveMonitoring";

const API = "http://127.0.0.1:8000";

type Incident = {
  incident_id: number;
  scene_type: string;
  scene_description: string;
  risk_level: string;
  status: string;
  analyzed_at: string;
};

type DashboardStats = {
  total_incidents: number;
  total_hazards: number;
  high_risk_incidents: number;
  review_required: number;
  risk_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  hazard_distribution: Record<string, number>;
  recent_incidents: Incident[];
};

type IncidentResponse = {
  total: number;
  page: number;
  limit: number;
  incidents: Incident[];
};

type Page = "dashboard" | "incidents" | "live";

function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [total, setTotal] = useState(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");

  const formatDate = (date: string) =>
    new Date(date).toLocaleString("en-IN");

  async function loadDashboard() {
    try {
      const response = await fetch(`${API}/api/dashboard/stats`);
      if (!response.ok) throw new Error("Dashboard request failed.");

      const data: DashboardStats = await response.json();
      setStats(data);
      setError("");
    } catch {
      setError("Unable to connect to the backend.");
    }
  }

  async function loadIncidents() {
    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        page: "1",
        limit: "100",
      });

      if (riskFilter !== "ALL") {
        params.set("risk_level", riskFilter);
      }

      if (statusFilter !== "ALL") {
        params.set("status", statusFilter);
      }

      const response = await fetch(
        `${API}/api/incidents?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error("Unable to load incident history.");
      }

      const data: IncidentResponse = await response.json();

      setIncidents(data.incidents);
      setTotal(data.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Incident history could not be loaded."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
    const interval = setInterval(loadDashboard, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (page === "incidents") {
      loadIncidents();
    }
  }, [page, riskFilter, statusFilter]);

  const filteredIncidents = incidents.filter((incident) => {
    const query = search.toLowerCase();

    return (
      incident.scene_type.toLowerCase().includes(query) ||
      incident.scene_description.toLowerCase().includes(query) ||
      String(incident.incident_id).includes(query)
    );
  });

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">ED</div>
          <div>
            <h2>ENVIRONSAFE</h2>
            <span>Danger Prediction System</span>
          </div>
        </div>

        <nav>
          <button
            className={`nav-item ${page === "dashboard" ? "active" : ""}`}
            onClick={() => setPage("dashboard")}
          >
            ▦ Dashboard
          </button>

          <button
            className={`nav-item ${page === "incidents" ? "active" : ""}`}
            onClick={() => setPage("incidents")}
          >
            ◷ Incident History
          </button>

          <button
            className={`nav-item ${page === "live" ? "active" : ""}`}
            onClick={() => setPage("live")}
          >
            ⚠ Hazard Monitoring
          </button>

          <button className="nav-item" disabled>
            ▤ Analytics
          </button>
        </nav>

        <div className="sidebar-bottom">
          <span className="system-dot"></span>
          System Dashboard
          <small>Version 1.0</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">
              ENVIRONMENTAL SAFETY / {page.toUpperCase()}
            </p>

            <h1>
              {page === "dashboard"
                ? "Dashboard Overview"
                : page === "incidents"
                ? "Incident History"
                : "Live Hazard Monitoring"}
            </h1>

            <p className="subtitle">
              {page === "dashboard"
                ? "Monitor environmental hazards and incident activity."
                : page === "incidents"
                ? "Search, filter and review recorded incidents."
                : "Real-time webcam analysis powered by Gemini AI."}
            </p>
          </div>

          <div className="topbar-right">
            <span className="live-indicator">
              <span className="system-dot"></span> LIVE DATA
            </span>
            <div className="profile">AD</div>
          </div>
        </header>

        {error && (
          <div className="error-message">
            {error}
            <button
              onClick={
                page === "dashboard"
                  ? loadDashboard
                  : loadIncidents
              }
            >
              Retry
            </button>
          </div>
        )}

        {page === "dashboard" && (
          <Dashboard
            stats={stats}
            formatDate={formatDate}
            onViewIncidents={() => setPage("incidents")}
          />
        )}

        {page === "live" && <LiveMonitoring />}

        {page === "incidents" && (
          <section className="panel incidents-panel">
            <div className="panel-heading">
              <div>
                <h2>All Incidents</h2>
                <p>Recorded observations from the MySQL database</p>
              </div>

              <button
                className="refresh-button"
                onClick={loadIncidents}
              >
                ↻ Refresh
              </button>
            </div>

            <div className="incident-filters">
              <input
                type="search"
                placeholder="Search by ID, scene or description..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />

              <select
                value={riskFilter}
                onChange={(event) => setRiskFilter(event.target.value)}
              >
                <option value="ALL">All Risk Levels</option>
                <option value="HIGH">High Risk</option>
                <option value="MEDIUM">Medium Risk</option>
                <option value="LOW">Low Risk</option>
                <option value="UNCERTAIN">Uncertain</option>
              </select>

              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
              >
                <option value="ALL">All Statuses</option>
                <option value="NO_RULE_TRIGGERED">No Rule Triggered</option>
                <option value="REVIEW_REQUIRED">Review Required</option>
                <option value="MANUAL_REVIEW">Manual Review</option>
                <option value="ACKNOWLEDGED">Acknowledged</option>
                <option value="RESOLVED">Resolved</option>
              </select>
            </div>

            <div className="incident-summary">
              Showing {filteredIncidents.length} of {total} matching records
            </div>

            {loading ? (
              <p className="empty-state">Loading incidents...</p>
            ) : (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>INCIDENT</th>
                      <th>SCENE</th>
                      <th>RISK LEVEL</th>
                      <th>STATUS</th>
                      <th>DATE & TIME</th>
                    </tr>
                  </thead>

                  <tbody>
                    {filteredIncidents.map((incident) => (
                      <tr key={incident.incident_id}>
                        <td className="incident-id">
                          #{incident.incident_id}
                        </td>

                        <td>
                          <strong>{incident.scene_type}</strong>
                          <p className="description">
                            {incident.scene_description}
                          </p>
                        </td>

                        <td>
                          <RiskBadge level={incident.risk_level} />
                        </td>

                        <td>
                          <span className="status-text">
                            {incident.status.replaceAll("_", " ")}
                          </span>
                        </td>

                        <td className="date-cell">
                          {formatDate(incident.analyzed_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {!filteredIncidents.length && (
                  <p className="empty-state">
                    No matching incidents found.
                  </p>
                )}
              </div>
            )}
          </section>
        )}

        <footer>
          Environmental Danger Prediction System
          <span>Powered by Gemini AI · FastAPI · MySQL</span>
        </footer>
      </main>
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  return (
    <span className={`risk-badge ${level.toLowerCase()}`}>
      {level}
    </span>
  );
}

function Dashboard({
  stats,
  formatDate,
  onViewIncidents,
}: {
  stats: DashboardStats | null;
  formatDate: (date: string) => string;
  onViewIncidents: () => void;
}) {
  if (!stats) {
    return <p className="message">Loading dashboard statistics...</p>;
  }

  return (
    <>
      <section className="metrics-grid">
        <Metric label="Total Incidents" value={stats.total_incidents} icon="◈" color="blue" />
        <Metric label="Total Hazards" value={stats.total_hazards} icon="⚠" color="orange" />
        <Metric label="High Risk Incidents" value={stats.high_risk_incidents} icon="!" color="red" />
        <Metric label="Review Required" value={stats.review_required} icon="◷" color="purple" />
      </section>

      <section className="content-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Risk Distribution</h2>
              <p>Incidents grouped by assessed risk level</p>
            </div>
          </div>

          <RiskBar label="High Risk" value={stats.risk_distribution.HIGH || 0} total={stats.total_incidents} color="red" />
          <RiskBar label="Medium Risk" value={stats.risk_distribution.MEDIUM || 0} total={stats.total_incidents} color="orange" />
          <RiskBar label="Low Risk" value={stats.risk_distribution.LOW || 0} total={stats.total_incidents} color="green" />
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Hazard Summary</h2>
              <p>Detected environmental hazard categories</p>
            </div>
          </div>

          {Object.entries(stats.hazard_distribution).length === 0 ? (
            <p className="empty-state">No hazards recorded.</p>
          ) : (
            Object.entries(stats.hazard_distribution).map(([name, count]) => (
              <div className="hazard-row" key={name}>
                <span className="hazard-icon">⚠</span>
                <span>{name.replaceAll("_", " ")}</span>
                <strong>{count}</strong>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel incidents-panel">
        <div className="panel-heading">
          <div>
            <h2>Recent Incidents</h2>
            <p>Latest recorded environmental observations</p>
          </div>

          <button className="refresh-button" onClick={onViewIncidents}>
            View All →
          </button>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>INCIDENT</th>
                <th>SCENE</th>
                <th>RISK LEVEL</th>
                <th>STATUS</th>
                <th>DATE & TIME</th>
              </tr>
            </thead>

            <tbody>
              {stats.recent_incidents.map((incident) => (
                <tr key={incident.incident_id}>
                  <td className="incident-id">#{incident.incident_id}</td>
                  <td>
                    <strong>{incident.scene_type}</strong>
                    <p className="description">{incident.scene_description}</p>
                  </td>
                  <td><RiskBadge level={incident.risk_level} /></td>
                  <td>
                    <span className="status-text">
                      {incident.status.replaceAll("_", " ")}
                    </span>
                  </td>
                  <td className="date-cell">{formatDate(incident.analyzed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function Metric({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: number;
  icon: string;
  color: string;
}) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${color}`}>{icon}</div>
      <p>{label}</p>
      <h2>{value}</h2>
      <span>Database records</span>
    </div>
  );
}

function RiskBar({
  label,
  value,
  total,
  color,
}: {
  label: string;
  value: number;
  total: number;
  color: string;
}) {
  const percentage = total > 0 ? (value / total) * 100 : 0;

  return (
    <div className="risk-row">
      <div className="risk-label">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div className="progress-track">
        <div
          className={`progress-fill ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default App;

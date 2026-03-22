import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useAppData } from "./hooks/useAppData";
import { MatchExplorer } from "./pages/MatchExplorer";
import { ModelExplainer } from "./pages/ModelExplainer";
import { PlayerDashboard } from "./pages/PlayerDashboard";
import { TeamComparison } from "./pages/TeamComparison";

export default function App() {
  const { data, error } = useAppData();

  if (error) {
    return (
      <div className="page" style={{ padding: "2rem" }}>
        <h2>Could not load data</h2>
        <p className="muted">{error}</p>
        <p className="muted">
          Place <code>app_data.json</code> in <code>webapp/public/</code> (copy from{" "}
          <code>data/predictions/app_data.json</code> after training).
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page" style={{ padding: "2rem" }}>
        <p>Loading…</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>PL xG</h1>
        <nav>
          <NavLink end className={({ isActive }) => (isActive ? "active" : "")} to="/">
            Match explorer
          </NavLink>
          <NavLink className={({ isActive }) => (isActive ? "active" : "")} to="/players">
            Player dashboard
          </NavLink>
          <NavLink className={({ isActive }) => (isActive ? "active" : "")} to="/teams">
            Team comparison
          </NavLink>
          <NavLink className={({ isActive }) => (isActive ? "active" : "")} to="/model">
            Model explainer
          </NavLink>
        </nav>
        <p className="muted" style={{ marginTop: "2rem", fontSize: "0.75rem" }}>
          {data.meta.label}
        </p>
      </aside>

      <Routes>
        <Route path="/" element={<MatchExplorer data={data} />} />
        <Route path="/players" element={<PlayerDashboard data={data} />} />
        <Route path="/teams" element={<TeamComparison data={data} />} />
        <Route path="/model" element={<ModelExplainer data={data} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AppDataPayload } from "../types";

type Props = { data: AppDataPayload };

function fmt(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(3);
}

export function ModelExplainer({ data }: Props) {
  const ev = data.evaluation;
  const [sim, setSim] = useState({
    x: 95,
    y: 38,
    body_part: "Right Foot",
    under_pressure: false,
    defenders_in_cone: 3,
    goal_diff: 0,
  });
  const [apiXg, setApiXg] = useState<number | null>(null);
  const [apiErr, setApiErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const calPts = useMemo(() => {
    if (!ev?.calibration) return [];
    const { prob_true, prob_pred } = ev.calibration;
    return prob_true.map((t, i) => ({
      bin: i + 1,
      empirical: t,
      predicted: prob_pred[i] ?? 0,
    }));
  }, [ev]);

  const sbCompare = useMemo(() => {
    const rows: { model: number; statsbomb: number }[] = [];
    for (const s of data.shots) {
      if (s.statsbombXg == null) continue;
      rows.push({ model: s.xg, statsbomb: s.statsbombXg });
      if (rows.length >= 400) break;
    }
    return rows;
  }, [data.shots]);

  async function runPredict() {
    setLoading(true);
    setApiErr(null);
    try {
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: sim.x,
          y: sim.y,
          body_part: sim.body_part,
          under_pressure: sim.under_pressure,
          defenders_in_cone: sim.defenders_in_cone,
          goal_diff: sim.goal_diff,
        }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      const j = (await r.json()) as { xg: number };
      setApiXg(j.xg);
    } catch (e) {
      setApiXg(null);
      setApiErr(
        e instanceof Error
          ? e.message
          : "Is the API running? uvicorn api.main:app --port 8000",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <h2>Model & methodology</h2>
      <p className="muted">
        Expected goals (xG) estimates the probability that a shot becomes a goal. This build adds
        <strong> context</strong> beyond location: pressure, freeze-frame geometry (defenders in the shot cone,
        goalkeeper positioning), and game state — features many public models skip.
      </p>

      <div className="panel">
        <h3 style={{ marginTop: 0 }}>Interactive shot simulator</h3>
        <p className="muted">
          Uses the same logistic model as training via <code>POST /api/predict</code>. Start the API:{" "}
          <code>uvicorn api.main:app --reload --port 8000</code> (Vite proxies <code>/api</code> in dev).
        </p>
        <div style={{ display: "grid", gap: "0.75rem", maxWidth: 480 }}>
          <label>
            x (0–120){" "}
            <input
              type="number"
              step={0.1}
              value={sim.x}
              onChange={(e) => setSim({ ...sim, x: Number(e.target.value) })}
            />
          </label>
          <label>
            y (0–80){" "}
            <input
              type="number"
              step={0.1}
              value={sim.y}
              onChange={(e) => setSim({ ...sim, y: Number(e.target.value) })}
            />
          </label>
          <label>
            Body part{" "}
            <select
              value={sim.body_part}
              onChange={(e) => setSim({ ...sim, body_part: e.target.value })}
            >
              <option>Right Foot</option>
              <option>Left Foot</option>
              <option>Head</option>
              <option>Other</option>
            </select>
          </label>
          <label>
            Defenders in cone{" "}
            <input
              type="number"
              min={0}
              max={20}
              value={sim.defenders_in_cone}
              onChange={(e) => setSim({ ...sim, defenders_in_cone: Number(e.target.value) })}
            />
          </label>
          <label>
            Goal diff (shooting team){" "}
            <input
              type="number"
              min={-5}
              max={5}
              value={sim.goal_diff}
              onChange={(e) => setSim({ ...sim, goal_diff: Number(e.target.value) })}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={sim.under_pressure}
              onChange={(e) => setSim({ ...sim, under_pressure: e.target.checked })}
            />{" "}
            Under pressure
          </label>
          <button type="button" className="primary" disabled={loading} onClick={() => void runPredict()}>
            {loading ? "Predicting…" : "Get xG"}
          </button>
          {apiXg != null && (
            <p>
              <strong>Model xG:</strong> {apiXg.toFixed(3)}
            </p>
          )}
          {apiErr && <p className="muted">{apiErr}</p>}
        </div>
      </div>

      {ev && (
        <>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>Results (from last training export)</h3>
            <p className="muted">
              Prefer <strong>cross-validated</strong> metrics (grouped by match) for honest reporting.{" "}
              {ev.calibration.based_on === "out_of_fold" ? "Calibration uses out-of-fold predictions." : "Calibration uses in-sample predictions (use more matches for grouped CV)."}
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <div>
                <div className="muted">CV log-loss (context)</div>
                <strong>{fmt(ev.cross_val_grouped_by_match.context_model?.log_loss)}</strong>
              </div>
              <div>
                <div className="muted">CV log-loss (baseline)</div>
                <strong>{fmt(ev.cross_val_grouped_by_match.baseline_location_only?.log_loss)}</strong>
              </div>
              <div>
                <div className="muted">CV ROC-AUC (context)</div>
                <strong>{fmt(ev.cross_val_grouped_by_match.context_model?.roc_auc)}</strong>
              </div>
              <div>
                <div className="muted">CV Brier (context)</div>
                <strong>{fmt(ev.cross_val_grouped_by_match.context_model?.brier)}</strong>
              </div>
            </div>
          </div>

          <div className="panel">
            <h3 style={{ marginTop: 0 }}>Calibration</h3>
            <p className="muted">Predicted probability vs observed goal rate (uniform bins).</p>
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <LineChart data={calPts} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#2a3b44" strokeDasharray="3 3" />
                  <XAxis dataKey="bin" stroke="#8b9a92" label={{ value: "Bin", fill: "#8b9a92" }} />
                  <YAxis domain={[0, 1]} stroke="#8b9a92" />
                  <Tooltip contentStyle={{ background: "#111820", border: "1px solid #2a3b44" }} />
                  <Legend />
                  <Line type="monotone" dataKey="predicted" name="Mean predicted" stroke="#74c0fc" dot />
                  <Line type="monotone" dataKey="empirical" name="Observed rate" stroke="#7ee787" dot />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {ev.model_vs_statsbomb && (
            <div className="panel">
              <h3 style={{ marginTop: 0 }}>Your model vs StatsBomb xG</h3>
              <p className="muted">
                Mean absolute error vs StatsBomb’s published xG on the same shots:{" "}
                <strong>{fmt(ev.model_vs_statsbomb.mean_absolute_error_vs_statsbomb_xg)}</strong>. Correlation:{" "}
                <strong>{fmt(ev.model_vs_statsbomb.pearson_r_model_vs_statsbomb)}</strong> ({ev.model_vs_statsbomb.n_shots_compared}{" "}
                shots).
              </p>
              {sbCompare.length > 5 && (
                <div style={{ width: "100%", height: 320 }}>
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                      <CartesianGrid stroke="#2a3b44" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="model" name="This model" domain={[0, 1]} stroke="#8b9a92" />
                      <YAxis type="number" dataKey="statsbomb" name="StatsBomb" domain={[0, 1]} stroke="#8b9a92" />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#111820", border: "1px solid #2a3b44" }} />
                      <Legend />
                      <Scatter name="Shots" data={sbCompare} fill="#7ee787" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {!ev && (
        <div className="panel">
          <p className="muted">
            No <code>evaluation</code> block in <code>app_data.json</code>. Run{" "}
            <code>python scripts/train_and_export.py</code> and copy the JSON to <code>webapp/public/</code>.
          </p>
        </div>
      )}
    </main>
  );
}

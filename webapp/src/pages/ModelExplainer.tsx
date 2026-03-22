export function ModelExplainer() {
  return (
    <main className="page">
      <h2>What is xG?</h2>
      <div className="panel">
        <p>
          Expected goals (xG) estimates the probability that a shot becomes a goal using features of the chance —
          typically distance, angle, and body part. This project adds <strong>context</strong>: defensive pressure,
          freeze-frame geometry (defenders in the shot cone, goalkeeper positioning), and game state.
        </p>
        <p>
          Models are trained on StatsBomb open data (logistic regression + XGBoost). The web app reads static JSON
          produced by <code>scripts/train_and_export.py</code> — no backend required for the MVP.
        </p>
        <p className="muted">
          See the repository README for metrics (log-loss, Brier, calibration) and methodology.
        </p>
      </div>
    </main>
  );
}

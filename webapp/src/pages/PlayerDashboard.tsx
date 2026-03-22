import type { AppDataPayload } from "../types";

type Props = { data: AppDataPayload };

/**
 * Player-level shots require joining StatsBomb lineups / player ids in the export.
 * Placeholder view until `player_id` is added to the pipeline.
 */
export function PlayerDashboard({ data }: Props) {
  const n = data.shots.length;
  return (
    <main className="page">
      <h2>Player dashboard</h2>
      <p className="muted">
        Shot-level player metadata is not in the MVP export yet. Extend <code>build_feature_frame</code> and the
        training script to retain <code>player_id</code> from StatsBomb events, then aggregate xG vs goals per player
        here.
      </p>
      <div className="panel">
        <p>
          Loaded dataset: <strong>{n}</strong> shots across <strong>{data.matches.length}</strong> matches (
          {data.meta.label}).
        </p>
      </div>
    </main>
  );
}

export type AppMeta = {
  competitionId: number;
  seasonId: number;
  label: string;
};

export type MatchRow = {
  matchId: number;
  date: string;
  homeTeam: string;
  awayTeam: string;
  homeTeamId: number | null;
  awayTeamId: number | null;
  homeScore: number | null;
  awayScore: number | null;
};

export type ShotRow = {
  matchId: number;
  teamId: number | null;
  minute: number;
  xg: number;
  goal: number;
  x: number;
  y: number;
  distance: number;
  angle: number;
  statsbombXg?: number;
};

export type MetricBundle = {
  log_loss?: number;
  brier?: number;
  roc_auc?: number | null;
  note?: string;
};

export type EvaluationPayload = {
  n_shots: number;
  n_matches: number;
  cross_val_grouped_by_match: {
    context_model: MetricBundle;
    baseline_location_only: MetricBundle;
    n_splits: number | null;
    note?: string;
  };
  in_sample_full_context: MetricBundle;
  calibration: {
    strategy: string;
    prob_true: number[];
    prob_pred: number[];
    based_on: string;
  };
  statsbomb_official_xg?: {
    metrics_on_same_shots: MetricBundle;
    note?: string;
  };
  model_vs_statsbomb?: {
    mean_absolute_error_vs_statsbomb_xg: number;
    pearson_r_model_vs_statsbomb: number | null;
    n_shots_compared: number;
    predictions_used: string;
  };
};

export type AppDataPayload = {
  meta: AppMeta;
  matches: MatchRow[];
  shots: ShotRow[];
  evaluation?: EvaluationPayload;
};

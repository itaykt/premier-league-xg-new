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
};

export type AppDataPayload = {
  meta: AppMeta;
  matches: MatchRow[];
  shots: ShotRow[];
};

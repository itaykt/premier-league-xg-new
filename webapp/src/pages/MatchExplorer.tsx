import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PitchShotMap } from "../components/PitchShotMap";
import type { AppDataPayload, MatchRow, ShotRow } from "../types";

function cumulativeTimeline(
  shots: ShotRow[],
  homeId: number | null,
  awayId: number | null,
): { minute: number; home: number; away: number }[] {
  const sorted = [...shots].sort((a, b) => a.minute - b.minute);
  let home = 0;
  let away = 0;
  const pts: { minute: number; home: number; away: number }[] = [{ minute: 0, home: 0, away: 0 }];
  for (const s of sorted) {
    if (s.teamId != null && s.teamId === homeId) home += s.xg;
    else if (s.teamId != null && s.teamId === awayId) away += s.xg;
    else continue;
    pts.push({ minute: s.minute, home, away });
  }
  return pts;
}

type Props = { data: AppDataPayload };

export function MatchExplorer({ data }: Props) {
  const [matchId, setMatchId] = useState<number>(() => data.matches[0]?.matchId ?? 0);

  const match: MatchRow | undefined = useMemo(
    () => data.matches.find((m) => m.matchId === matchId),
    [data.matches, matchId],
  );

  const shots = useMemo(
    () => data.shots.filter((s) => s.matchId === matchId),
    [data.shots, matchId],
  );

  const chartData = useMemo(() => {
    if (!match) return [];
    return cumulativeTimeline(shots, match.homeTeamId, match.awayTeamId);
  }, [match, shots]);

  const totalHome = shots.filter((s) => s.teamId === match?.homeTeamId).reduce((a, s) => a + s.xg, 0);
  const totalAway = shots.filter((s) => s.teamId === match?.awayTeamId).reduce((a, s) => a + s.xg, 0);

  if (data.matches.length === 0) {
    return (
      <main className="page">
        <h2>Match explorer</h2>
        <p className="muted">No matches in app_data.json — run the training export script.</p>
      </main>
    );
  }

  return (
    <main className="page">
      <h2>Match explorer</h2>
      <p className="muted">Cumulative xG timeline, shot map, and match totals.</p>

      <div className="panel">
        <label htmlFor="match-select">Match </label>
        <select
          id="match-select"
          value={matchId}
          onChange={(e) => setMatchId(Number(e.target.value))}
        >
          {data.matches.map((m) => (
            <option key={m.matchId} value={m.matchId}>
              {m.date} · {m.homeTeam} vs {m.awayTeam}
            </option>
          ))}
        </select>
      </div>

      {match && (
        <>
          <div className="panel">
            <strong>
              {match.homeTeam} {match.homeScore ?? "—"} – {match.awayScore ?? "—"} {match.awayTeam}
            </strong>
            <p className="muted">
              Model xG: {totalHome.toFixed(2)} – {totalAway.toFixed(2)} · Shots: {shots.length}
            </p>
          </div>

          <div className="panel">
            <h3 style={{ marginTop: 0 }}>xG timeline</h3>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <AreaChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#2a3b44" strokeDasharray="3 3" />
                  <XAxis dataKey="minute" stroke="#8b9a92" />
                  <YAxis stroke="#8b9a92" />
                  <Tooltip
                    contentStyle={{ background: "#111820", border: "1px solid #2a3b44" }}
                    labelStyle={{ color: "#e8f0e8" }}
                  />
                  <Legend />
                  <Area type="stepAfter" dataKey="home" name={match.homeTeam} stroke="#7ee787" fill="#1a3d2a" />
                  <Area type="stepAfter" dataKey="away" name={match.awayTeam} stroke="#74c0fc" fill="#1a2e3d" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel">
            <h3 style={{ marginTop: 0 }}>Shot map</h3>
            <PitchShotMap shots={shots} />
          </div>
        </>
      )}
    </main>
  );
}

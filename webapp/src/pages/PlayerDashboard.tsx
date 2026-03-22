import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AppDataPayload, ShotRow } from "../types";

type Props = { data: AppDataPayload };

type PlayerAgg = {
  playerId: number;
  playerName: string;
  teamId: number | null;
  shots: ShotRow[];
  goals: number;
  xg: number;
};

function aggregateByPlayer(shots: ShotRow[]): PlayerAgg[] {
  const byId = new Map<number, { name: string; list: ShotRow[] }>();
  for (const s of shots) {
    if (s.playerId == null) continue;
    const id = s.playerId;
    let e = byId.get(id);
    if (!e) {
      e = { name: s.playerName?.trim() || `Player ${id}`, list: [] };
      byId.set(id, e);
    }
    if (s.playerName?.trim()) e.name = s.playerName.trim();
    e.list.push(s);
  }
  const out: PlayerAgg[] = [];
  for (const [playerId, { name, list }] of byId) {
    const goals = list.reduce((a, x) => a + x.goal, 0);
    const xg = list.reduce((a, x) => a + x.xg, 0);
    const teamCounts = new Map<number, number>();
    for (const sh of list) {
      if (sh.teamId != null) teamCounts.set(sh.teamId, (teamCounts.get(sh.teamId) ?? 0) + 1);
    }
    let teamId: number | null = null;
    let best = 0;
    for (const [tid, c] of teamCounts) {
      if (c > best) {
        best = c;
        teamId = tid;
      }
    }
    out.push({ playerId, playerName: name, teamId, shots: list, goals, xg });
  }
  return out.sort((a, b) => b.xg - a.xg);
}

export function PlayerDashboard({ data }: Props) {
  const teamName = useMemo(() => {
    const m = new Map<number, string>();
    for (const row of data.matches) {
      if (row.homeTeamId != null) m.set(row.homeTeamId, row.homeTeam);
      if (row.awayTeamId != null) m.set(row.awayTeamId, row.awayTeam);
    }
    return m;
  }, [data.matches]);

  const matchLine = useMemo(() => {
    const m = new Map<number, string>();
    for (const r of data.matches) {
      m.set(r.matchId, `${r.date} · ${r.homeTeam} vs ${r.awayTeam}`);
    }
    return m;
  }, [data.matches]);

  const players = useMemo(() => aggregateByPlayer(data.shots), [data.shots]);

  const [selectedId, setSelectedId] = useState<number | null>(null);

  const selected = useMemo(() => {
    if (selectedId == null) return null;
    return players.find((p) => p.playerId === selectedId) ?? null;
  }, [players, selectedId]);

  const chartData = useMemo(() => {
    if (!selected) return [];
    const byMin = new Map<number, number>();
    for (const s of selected.shots) {
      const bucket = Math.min(90, Math.max(0, Math.floor(s.minute / 5) * 5));
      byMin.set(bucket, (byMin.get(bucket) ?? 0) + s.xg);
    }
    return [...byMin.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([minute, xg]) => ({ minute: `${minute}′`, xg }));
  }, [selected]);

  const hasPlayerIds = data.shots.some((s) => s.playerId != null);

  if (!hasPlayerIds) {
    return (
      <main className="page">
        <h2>Player dashboard</h2>
        <p className="muted">
          No <code>playerId</code> on shots in this <code>app_data.json</code>. Re-run{" "}
          <code>python scripts/train_and_export.py</code> and copy{" "}
          <code>data/predictions/app_data.json</code> to <code>webapp/public/</code> so exports include player
          metadata from StatsBomb.
        </p>
        <div className="panel">
          <p>
            Loaded: <strong>{data.shots.length}</strong> shots · <strong>{data.matches.length}</strong> matches
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <h2>Player dashboard</h2>
      <p className="muted">
        Shot-level xG and goals aggregated by player (same model xG as elsewhere). Team is the side this player shot
        for most often in the export.
      </p>

      <div className="panel player-dash__layout">
        <div className="player-dash__table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th className="num">Shots</th>
                <th className="num">Goals</th>
                <th className="num">xG</th>
              </tr>
            </thead>
            <tbody>
              {players.map((p) => (
                <tr
                  key={p.playerId}
                  className={selectedId === p.playerId ? "selected" : undefined}
                  onClick={() => setSelectedId(p.playerId)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelectedId(p.playerId);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <td>{p.playerName}</td>
                  <td className="muted">{p.teamId != null ? teamName.get(p.teamId) ?? p.teamId : "—"}</td>
                  <td className="num">{p.shots.length}</td>
                  <td className="num">{p.goals}</td>
                  <td className="num">{p.xg.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="player-dash__detail">
          {selected ? (
            <>
              <h3 style={{ marginTop: 0 }}>{selected.playerName}</h3>
              <p className="muted">
                {selected.teamId != null ? teamName.get(selected.teamId) ?? `Team ${selected.teamId}` : "—"} ·{" "}
                {selected.shots.length} shots · {selected.goals} goals · <strong>{selected.xg.toFixed(2)}</strong> xG
              </p>
              {chartData.length > 0 && (
                <div style={{ width: "100%", height: 220, marginTop: "0.75rem" }}>
                  <ResponsiveContainer>
                    <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="#2a3b44" strokeDasharray="3 3" />
                      <XAxis dataKey="minute" stroke="#8b9a92" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                      <YAxis stroke="#8b9a92" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: "#111820", border: "1px solid #2a3b44" }}
                        labelStyle={{ color: "#e8f0e8" }}
                      />
                      <Bar dataKey="xg" name="xG" fill="#7ee787" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
              <p className="muted" style={{ fontSize: "0.82rem", marginTop: "0.5rem" }}>
                xG by 5-minute bucket (season)
              </p>
              <h4 style={{ marginBottom: "0.5rem" }}>Shots</h4>
              <div className="player-dash__shots">
                <table className="data-table data-table--compact">
                  <thead>
                    <tr>
                      <th>Match</th>
                      <th className="num">Min</th>
                      <th className="num">xG</th>
                      <th className="num">G</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...selected.shots]
                      .sort((a, b) => a.matchId - b.matchId || a.minute - b.minute)
                      .map((s, i) => (
                        <tr key={`${s.matchId}-${i}`}>
                          <td className="ellipsis" title={matchLine.get(s.matchId) ?? ""}>
                            {matchLine.get(s.matchId) ?? s.matchId}
                          </td>
                          <td className="num">{s.minute.toFixed(0)}′</td>
                          <td className="num">{s.xg.toFixed(2)}</td>
                          <td className="num">{s.goal ? "1" : "0"}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">Select a player from the table.</p>
          )}
        </div>
      </div>
    </main>
  );
}

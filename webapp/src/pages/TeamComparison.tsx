import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AppDataPayload } from "../types";

type Props = { data: AppDataPayload };

export function TeamComparison({ data }: Props) {
  const teamIds = useMemo(() => {
    const s = new Set<number>();
    for (const sh of data.shots) {
      if (sh.teamId != null) s.add(sh.teamId);
    }
    return [...s].sort((a, b) => a - b);
  }, [data.shots]);

  const idToName = useMemo(() => {
    const m = new Map<number, string>();
    for (const row of data.matches) {
      if (row.homeTeamId != null) m.set(row.homeTeamId, row.homeTeam);
      if (row.awayTeamId != null) m.set(row.awayTeamId, row.awayTeam);
    }
    return m;
  }, [data.matches]);

  const [ta, setTa] = useState<number | "">(teamIds[0] ?? "");
  const [tb, setTb] = useState<number | "">(teamIds[1] ?? "");

  const chartData = useMemo(() => {
    const aid = typeof ta === "number" ? ta : null;
    const bid = typeof tb === "number" ? tb : null;
    if (aid == null || bid == null) return [];
    const shotsA = data.shots.filter((s) => s.teamId === aid);
    const shotsB = data.shots.filter((s) => s.teamId === bid);
    const xg = (xs: typeof data.shots) => xs.reduce((acc, s) => acc + s.xg, 0);
    const goals = (xs: typeof data.shots) => xs.reduce((acc, s) => acc + s.goal, 0);
    return [
      { metric: "xG (for)", teamA: xg(shotsA), teamB: xg(shotsB) },
      { metric: "Goals", teamA: goals(shotsA), teamB: goals(shotsB) },
      { metric: "Shots", teamA: shotsA.length, teamB: shotsB.length },
    ];
  }, [ta, tb, data.shots]);

  const nameA = typeof ta === "number" ? idToName.get(ta) ?? `Team ${ta}` : "Team A";
  const nameB = typeof tb === "number" ? idToName.get(tb) ?? `Team ${tb}` : "Team B";

  return (
    <main className="page">
      <h2>Team comparison</h2>
      <p className="muted">Compare aggregated xG and shot volume between two teams in the loaded export.</p>

      <div className="panel" style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
        <label>
          Team A{" "}
          <select value={ta === "" ? "" : ta} onChange={(e) => setTa(e.target.value === "" ? "" : Number(e.target.value))}>
            <option value="">—</option>
            {teamIds.map((id) => (
              <option key={id} value={id}>
                {idToName.get(id) ?? id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Team B{" "}
          <select value={tb === "" ? "" : tb} onChange={(e) => setTb(e.target.value === "" ? "" : Number(e.target.value))}>
            <option value="">—</option>
            {teamIds.map((id) => (
              <option key={id} value={id}>
                {idToName.get(id) ?? id}
              </option>
            ))}
          </select>
        </label>
      </div>

      {chartData.length > 0 && (
        <div className="panel">
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid stroke="#2a3b44" strokeDasharray="3 3" />
                <XAxis dataKey="metric" stroke="#8b9a92" />
                <YAxis stroke="#8b9a92" />
                <Tooltip contentStyle={{ background: "#111820", border: "1px solid #2a3b44" }} />
                <Legend />
                <Bar dataKey="teamA" name={nameA} fill="#7ee787" />
                <Bar dataKey="teamB" name={nameB} fill="#74c0fc" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </main>
  );
}

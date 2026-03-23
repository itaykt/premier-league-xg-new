import type { ShotRow } from "../types";

/** StatsBomb pitch: length 120, width 80; goals at y∈[36,44], centered at y=40. */
const PW = 120;
const PH = 80;
const CY = 40;
const GY_LO = 36;
const GY_HI = 44;
/** Center circle radius (~9.15m scaled to pitch length). */
const CR = 9.15;
/** Penalty area: ~18 from goal line, 44 wide on 80 pitch → y 18–62. */
const PA_DEPTH = 18;
const PA_Y0 = 18;
const PA_Y1 = 62;
/** Six-yard box ~6 deep, ~20 wide → y 30–50. */
const SA_DEPTH = 6;
const SA_Y0 = 30;
const SA_Y1 = 50;
/** Penalty mark ~12 from goal line. */
const PM = 12;

type PitchShotMapProps = {
  shots: ShotRow[];
  homeTeamId: number | null;
  awayTeamId: number | null;
  homeTeamName: string;
  awayTeamName: string;
  /** Sum of shot-level model xG for this team in this match. */
  homeXg: number;
  awayXg: number;
};

/** Home: StatsBomb (x, y). Away: x mirrored so R→L attack reads on the same pitch — x_plot = pitch_length − x; y unchanged. */
function plotX(s: ShotRow, isHome: boolean): number {
  if (isHome) return PW - s.x;
  return s.x;
}

function shotRadius(xg: number): number {
  return 0.35 + Math.min(2.8, xg * 8);
}

export function PitchShotMap({
  shots,
  homeTeamId,
  awayTeamId,
  homeTeamName,
  awayTeamName,
  homeXg,
  awayXg,
}: PitchShotMapProps) {
  return (
    <div className="pitch-shotmap">
      <div className="pitch-shotmap__xg">
        <span className="pitch-shotmap__xg-home" title="Sum of model xG for home shots in this match">
          <strong>{homeTeamName}</strong> xG <em>{homeXg.toFixed(2)}</em>
        </span>
        <span className="pitch-shotmap__xg-sep">·</span>
        <span className="pitch-shotmap__xg-away" title="Sum of model xG for away shots in this match">
          <strong>{awayTeamName}</strong> xG <em>{awayXg.toFixed(2)}</em>
        </span>
      </div>

      <div className="pitch-shotmap__note muted">
        <strong>{homeTeamName}</strong> attacks right → left - <strong>{awayTeamName}</strong> attacks
        left → right
      </div>

      <div className="pitch pitch--svg">
        <svg viewBox={`0 0 ${PW} ${PH}`} className="pitch__svg" preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id="pitchGrass" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1e5c34" />
              <stop offset="100%" stopColor="#163d28" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width={PW} height={PH} fill="url(#pitchGrass)" />

          {/* Touchlines & ends */}
          <rect
            x="0.25"
            y="0.25"
            width={PW - 0.5}
            height={PH - 0.5}
            fill="none"
            stroke="rgba(255,255,255,0.85)"
            strokeWidth="0.5"
          />

          {/* Halfway line */}
          <line x1={PW / 2} y1="0" x2={PW / 2} y2={PH} stroke="rgba(255,255,255,0.75)" strokeWidth="0.45" />

          {/* Center circle */}
          <circle
            cx={PW / 2}
            cy={CY}
            r={CR}
            fill="none"
            stroke="rgba(255,255,255,0.75)"
            strokeWidth="0.45"
          />
          <circle cx={PW / 2} cy={CY} r="0.5" fill="rgba(255,255,255,0.85)" />

          {/* Left penalty area */}
          <rect
            x="0"
            y={PA_Y0}
            width={PA_DEPTH}
            height={PA_Y1 - PA_Y0}
            fill="none"
            stroke="rgba(255,255,255,0.75)"
            strokeWidth="0.4"
          />
          {/* Right penalty area */}
          <rect
            x={PW - PA_DEPTH}
            y={PA_Y0}
            width={PA_DEPTH}
            height={PA_Y1 - PA_Y0}
            fill="none"
            stroke="rgba(255,255,255,0.75)"
            strokeWidth="0.4"
          />

          {/* Six-yard boxes */}
          <rect
            x="0"
            y={SA_Y0}
            width={SA_DEPTH}
            height={SA_Y1 - SA_Y0}
            fill="none"
            stroke="rgba(255,255,255,0.65)"
            strokeWidth="0.35"
          />
          <rect
            x={PW - SA_DEPTH}
            y={SA_Y0}
            width={SA_DEPTH}
            height={SA_Y1 - SA_Y0}
            fill="none"
            stroke="rgba(255,255,255,0.65)"
            strokeWidth="0.35"
          />

          {/* Goals (mouth) */}
          <line x1="0" y1={GY_LO} x2="0" y2={GY_HI} stroke="rgba(255,255,255,0.95)" strokeWidth="1.2" />
          <line
            x1={PW}
            y1={GY_LO}
            x2={PW}
            y2={GY_HI}
            stroke="rgba(255,255,255,0.95)"
            strokeWidth="1.2"
          />

          {/* Penalty spots */}
          <circle cx={PM} cy={CY} r="0.45" fill="rgba(255,255,255,0.9)" />
          <circle cx={PW - PM} cy={CY} r="0.45" fill="rgba(255,255,255,0.9)" />

          {/* Shots */}
          {shots.map((s, i) => {
            if (s.teamId == null) return null;
            const isHome = s.teamId === homeTeamId;
            const isAway = s.teamId === awayTeamId;
            if (!isHome && !isAway) return null;
            const team = isHome ? homeTeamName : awayTeamName;
            const cls = isHome ? "shot shot--home" : "shot shot--away";
            const r = shotRadius(s.xg);
            const cx = plotX(s, isHome);
            const title = [
              team,
              `${s.minute.toFixed(0)}′`,
              `xG ${s.xg.toFixed(3)}`,
              s.goal ? "GOAL" : "",
            ]
              .filter(Boolean)
              .join(" · ");
            return (
              <circle
                key={`${s.matchId}-${i}-${s.minute}`}
                className={cls}
                cx={cx}
                cy={s.y}
                r={r}
                data-goal={s.goal ? "1" : "0"}
              >
                <title>{title}</title>
              </circle>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

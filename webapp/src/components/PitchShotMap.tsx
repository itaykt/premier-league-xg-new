import type { ShotRow } from "../types";

type Props = {
  shots: ShotRow[];
};

export function PitchShotMap({ shots }: Props) {
  return (
    <div className="pitch">
      {shots.map((s, i) => {
        const x = s.x;
        const y = s.y;
        const left = `${(x / 120) * 100}%`;
        const top = `${(y / 80) * 100}%`;
        const size = 4 + Math.min(18, s.xg * 40);
        const cls = s.goal ? "shot-dot goal" : "shot-dot miss";
        return (
          <div
            key={`${s.matchId}-${i}`}
            className={cls}
            style={{
              left,
              top,
              width: size,
              height: size,
              marginLeft: -size / 2,
              marginTop: -size / 2,
            }}
            title={`xG ${s.xg.toFixed(2)} · ${s.minute.toFixed(0)}'`}
          />
        );
      })}
    </div>
  );
}

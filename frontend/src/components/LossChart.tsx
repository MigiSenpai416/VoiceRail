import { useMemo } from "react";

type Row = Record<string, number | null>;

/** Mel loss per epoch (train scaled to the validation range), as a small SVG. */
export function LossChart({ metrics }: { metrics: Row[] }) {
  const { train, val, min, max, epochs } = useMemo(() => {
    const train: Array<[number, number]> = [];
    const val: Array<[number, number]> = [];
    for (const m of metrics) {
      const epoch = Number(m.epoch ?? 0);
      if (typeof m.mel === "number") train.push([epoch, m.mel / 45]);
      if (typeof m.val_mel === "number") val.push([epoch, m.val_mel]);
    }
    const ys = [...train, ...val].map(([, y]) => y);
    const xs = [...train, ...val].map(([x]) => x);
    return {
      train,
      val,
      min: ys.length ? Math.min(...ys) : 0,
      max: ys.length ? Math.max(...ys) : 1,
      epochs: xs.length ? Math.max(...xs) : 1,
    };
  }, [metrics]);

  if (!train.length && !val.length) {
    return (
      <div className="flex h-[180px] items-center justify-center rounded-xl border border-dashed border-border text-[12.5px] text-faint">
        The loss chart fills in as epochs finish
      </div>
    );
  }

  const W = 600;
  const H = 180;
  const pad = { l: 40, r: 12, t: 12, b: 24 };
  const spanY = max - min || 1;
  const x = (e: number) => pad.l + ((e - 1) / Math.max(1, epochs - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - min) / spanY) * (H - pad.t - pad.b);
  const path = (pts: Array<[number, number]>) =>
    pts.map(([e, v], i) => `${i ? "L" : "M"}${x(e).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const ticks = [min, (min + max) / 2, max];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-[180px] w-full" role="img" aria-label="Loss per epoch">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#242a37" strokeDasharray="3 4" />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="#5d6579">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {[1, epochs].map((e) => (
          <text key={e} x={x(e)} y={H - 6} textAnchor={e === 1 ? "start" : "end"} fontSize="10" fill="#5d6579">
            epoch {e}
          </text>
        ))}
        {train.length > 1 && <path d={path(train)} fill="none" stroke="#38bdf8" strokeWidth="1.8" />}
        {val.length > 1 && <path d={path(val)} fill="none" stroke="#a78bfa" strokeWidth="2" />}
      </svg>
      <div className="mt-1 flex gap-4 text-[11.5px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-sky" /> train
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-accent-soft" /> validation (lower is better)
        </span>
      </div>
    </div>
  );
}

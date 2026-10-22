import { cn } from "@/lib/utils";

export function ProgressBar({
  value,
  active,
  className,
}: {
  value: number; // 0..1
  active?: boolean;
  className?: string;
}) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-surface-3", className)}>
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out",
          active ? "vr-shimmer" : "bg-gradient-to-r from-accent to-sky",
        )}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

export function StatusPill({ state }: { state: string }) {
  const styles: Record<string, string> = {
    running: "bg-accent/15 text-accent-soft border-accent/30",
    finished: "bg-ok/15 text-ok border-ok/30",
    stopped: "bg-warn/15 text-warn border-warn/30",
    failed: "bg-danger/15 text-danger border-danger/30",
    crashed: "bg-danger/15 text-danger border-danger/30",
    idle: "bg-surface-3 text-muted border-border",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11.5px] font-semibold capitalize",
        styles[state] ?? styles.idle,
      )}
    >
      {state === "running" && <span className="size-1.5 animate-pulse rounded-full bg-accent-soft" />}
      {state}
    </span>
  );
}

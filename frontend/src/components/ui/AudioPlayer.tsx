import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Download, Pause, Play } from "lucide-react";
import { cn } from "@/lib/utils";

function clock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Waveform player for a URL (served by the API) or a local File. */
export function AudioPlayer({
  src,
  title,
  downloadName,
  compact,
  className,
}: {
  src: string | File | null;
  title?: string;
  downloadName?: string;
  compact?: boolean;
  className?: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const wave = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!src) {
      setUrl(null);
      return;
    }
    if (typeof src === "string") {
      setUrl(src);
      return;
    }
    const object = URL.createObjectURL(src);
    setUrl(object);
    return () => URL.revokeObjectURL(object);
  }, [src]);

  useEffect(() => {
    if (!container.current || !url) return;
    const ws = WaveSurfer.create({
      container: container.current,
      url,
      height: compact ? 40 : 56,
      waveColor: "#3b4356",
      progressColor: "#a78bfa",
      cursorColor: "#38bdf8",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
    });
    wave.current = ws;
    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("timeupdate", (t) => setTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    return () => {
      ws.destroy();
      wave.current = null;
      setPlaying(false);
      setTime(0);
      setDuration(0);
    };
  }, [url, compact]);

  if (!url) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-xl border border-dashed border-border bg-surface-2/40 text-[12.5px] text-faint",
          compact ? "h-[72px]" : "h-[96px]",
          className,
        )}
      >
        {title ? `${title} appears here` : "Nothing yet"}
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-border bg-surface-2 px-3 py-2.5", className)}>
      {title && (
        <div className="mb-1.5 flex items-center justify-between text-[12px]">
          <span className="font-medium text-muted">{title}</span>
          <a
            href={typeof src === "string" ? `${src}&download=1` : url}
            download={downloadName}
            className="inline-flex items-center gap-1 text-muted hover:text-text"
          >
            <Download className="size-3.5" /> Download
          </a>
        </div>
      )}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => wave.current?.playPause()}
          aria-label={playing ? "Pause" : "Play"}
          className="flex size-9 shrink-0 items-center justify-center rounded-full bg-accent text-white shadow hover:brightness-110"
        >
          {playing ? <Pause className="size-4" /> : <Play className="ml-0.5 size-4" />}
        </button>
        <div ref={container} className="min-w-0 flex-1" />
        <span className="w-[5.5rem] shrink-0 text-right font-mono text-[11.5px] text-muted">
          {clock(time)} / {clock(duration)}
        </span>
      </div>
    </div>
  );
}

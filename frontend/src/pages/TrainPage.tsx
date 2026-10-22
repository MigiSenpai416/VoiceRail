import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AudioLines, Mic, Music, Play, RotateCcw, Square, Volume2 } from "lucide-react";
import { useApp } from "@/AppContext";
import { api, type SessionDetail, type SessionStatus } from "@/lib/api";
import { cn, formatMinutes } from "@/lib/utils";
import { LossChart } from "@/components/LossChart";
import { Alert } from "@/components/ui/Alert";
import { AudioPlayer } from "@/components/ui/AudioPlayer";
import { Button } from "@/components/ui/Button";
import { Card, CardBody, Section } from "@/components/ui/Card";
import { ChoiceCards } from "@/components/ui/ChoiceCards";
import { Disclosure } from "@/components/ui/Collapsible";
import { Dropzone } from "@/components/ui/Dropzone";
import { Field, Hint } from "@/components/ui/Field";
import { Input, NumberInput } from "@/components/ui/Input";
import { ProgressBar, StatusPill } from "@/components/ui/Progress";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";

type Cleaning = {
  isolate_vocals: boolean;
  vocal_model: string;
  remove_backing_vocals: boolean;
  lead_model: string;
  dereverb: boolean;
  dereverb_model: string;
  denoise: boolean;
  denoise_model: string;
};
type Segmenting = { target_lufs: number; silence_db: number; max_seconds: number; min_seconds: number };
type TrainConfig = {
  sample_rate: number;
  epochs: number;
  batch_size: number;
  pretrained: string;
  precision: string;
  save_every: number;
  keep_snapshots: boolean;
  early_stopping: boolean;
  patience: number;
  learning_rate: number;
  lr_decay: number;
  c_mel: number;
  c_kl: number;
  gradient_checkpointing: boolean;
};

const PRESET_ICONS: Record<string, ReactNode> = {
  songs: <Music className="size-4" />,
  clean: <Mic className="size-4" />,
  noisy: <Volume2 className="size-4" />,
};

export function TrainPage() {
  const { options, refreshModels } = useApp();
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [folder, setFolder] = useState("");
  const [preset, setPreset] = useState("songs");
  const [cleaning, setCleaning] = useState<Cleaning | null>(null);
  const [segmenting, setSegmenting] = useState<Segmenting | null>(null);
  const [train, setTrain] = useState<TrainConfig | null>(null);
  const [fresh, setFresh] = useState(false);
  const [sessions, setSessions] = useState<SessionStatus[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seenState = useRef<string | null>(null);

  const reset = useCallback(() => {
    if (!options) return;
    setPreset("songs");
    setCleaning({ ...(options.train.cleaning as Cleaning), ...options.train.presets[0].cleaning });
    setSegmenting(options.train.segmenting as Segmenting);
    setTrain(options.train.config as unknown as TrainConfig);
    setFresh(false);
    setFolder("");
  }, [options]);
  useEffect(reset, [reset]);

  const loadSessions = useCallback(async () => {
    const list = await api.get<SessionStatus[]>("/api/train/sessions");
    setSessions(list);
    return list;
  }, []);

  // First load: attach to the running (or latest) session.
  useEffect(() => {
    loadSessions()
      .then((list) => {
        const latest = list.find((s) => s.latest) ?? list[0];
        if (latest && !selected) {
          setSelected(latest.name);
          setName((n) => n || latest.name);
        }
      })
      .catch((e: Error) => setError(e.message));
  }, [loadSessions]); // eslint-disable-line react-hooks/exhaustive-deps

  // Poll the selected session while the page is open.
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    let stopped = false;
    const tick = async () => {
      try {
        const d = await api.get<SessionDetail>(`/api/train/sessions/${encodeURIComponent(selected)}`);
        if (stopped) return;
        setDetail(d);
        const signature = `${d.state}|${d.updated}`;
        if (seenState.current && seenState.current.split("|")[0] === "running" && d.state === "finished") {
          refreshModels();
        }
        seenState.current = signature;
      } catch (e) {
        if (!stopped) setError((e as Error).message);
      }
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [selected, refreshModels]);

  const applyPreset = (key: string) => {
    if (!options || !cleaning) return;
    const p = options.train.presets.find((x) => x.key === key);
    if (!p) return;
    setPreset(key);
    setCleaning({ ...cleaning, ...p.cleaning });
  };

  const selectSession = (sessionName: string) => {
    setSelected(sessionName);
    setName(sessionName);
    seenState.current = null;
  };

  const launch = async (stages: "full" | "prepare") => {
    if (!cleaning || !segmenting || !train) return;
    setError(null);
    setBusy(stages);
    const form = new FormData();
    form.append("name", name);
    form.append("settings", JSON.stringify({ cleaning, segmenting, train, folder, fresh, stages }));
    for (const f of files) form.append("files", f);
    try {
      const d = await api.form<SessionDetail>("/api/train", form);
      setFiles([]);
      setDetail(d);
      setSelected(d.name);
      setName(d.name);
      await loadSessions();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const act = async (action: "stop" | "publish") => {
    if (!selected) return;
    setError(null);
    setBusy(action);
    try {
      await api.json(`/api/train/sessions/${encodeURIComponent(selected)}/${action}`, "POST");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const running = detail?.state === "running";
  const sepModels = (list: { label: string; note: string }[]) =>
    list.map((m) => ({ value: m.label, label: m.label, description: m.note || undefined }));
  const sessionOptions = useMemo(
    () => sessions.map((s) => ({ value: s.name, label: s.name, description: s.state === "idle" ? undefined : s.state })),
    [sessions],
  );

  if (!options || !cleaning || !segmenting || !train) return null;

  return (
    <div className="space-y-4">
      <Card>
        <CardBody className="space-y-5">
          <Field label="Voice name" help="Also the name of the training session. Pick an earlier session under Options to resume it.">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. MyVoice" />
          </Field>
          <Dropzone
            files={files}
            onChange={setFiles}
            multiple
            accept={options.convert.audio_extensions.join(",")}
            title="Drop recordings of the voice here"
            subtitle="10–30 minutes of one person is ideal; 5 minutes works. mp3, wav, flac, m4a, videos…"
          />
          <Field label="What kind of recordings?" help="Picks the clean-up steps. Fine-tune them under Options › Clean-up.">
            <ChoiceCards
              value={preset}
              onChange={applyPreset}
              choices={options.train.presets.map((p) => ({
                value: p.key,
                label: p.label,
                description: p.description,
                icon: PRESET_ICONS[p.key],
              }))}
            />
          </Field>
        </CardBody>
      </Card>

      <Disclosure
        title="Options"
        summary={`${train.sample_rate / 1000}k · ${train.epochs || "auto"} epochs · batch ${train.batch_size || "auto"}`}
        action={
          <Button size="sm" variant="ghost" onClick={reset}>
            <RotateCcw className="size-3.5" /> Reset
          </Button>
        }
      >
        <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
          <Section title="Session">
            <Field label="Training session" help="Pick an earlier session to monitor or resume it; its name becomes the voice name.">
              <Select value={selected} onChange={selectSession} options={sessionOptions} placeholder="No sessions yet" />
            </Field>
            <Field label="Folder on this computer" help="Recordings in this folder are used in addition to uploaded files.">
              <Input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="C:\Users\me\voice-recordings" />
            </Field>
            {detail && detail.sources.count > 0 && (
              <Hint>
                {detail.sources.count} recording(s) stored in this session ({detail.sources.mb.toFixed(0)} MB).
              </Hint>
            )}
            <Switch checked={fresh} onChange={setFresh} label="Start over" help="Discard this session's previous checkpoints and train from the base model again." />
            <div className="flex flex-wrap gap-2">
              <Button size="sm" onClick={() => launch("prepare")} loading={busy === "prepare"} disabled={running || !name}>
                Prepare dataset only
              </Button>
              <Button size="sm" onClick={() => act("publish")} loading={busy === "publish"} disabled={running || !selected}>
                Save current model now
              </Button>
            </div>
          </Section>

          <Section title="Clean-up">
            <Switch checked={cleaning.isolate_vocals} onChange={(v) => setCleaning({ ...cleaning, isolate_vocals: v })} label="Remove music / background" />
            {cleaning.isolate_vocals && (
              <Select value={cleaning.vocal_model} onChange={(v) => setCleaning({ ...cleaning, vocal_model: v })} options={sepModels(options.convert.vocal_models)} />
            )}
            <Switch checked={cleaning.remove_backing_vocals} onChange={(v) => setCleaning({ ...cleaning, remove_backing_vocals: v })} label="Remove backing vocals / harmonies" />
            {cleaning.remove_backing_vocals && (
              <Select value={cleaning.lead_model} onChange={(v) => setCleaning({ ...cleaning, lead_model: v })} options={sepModels(options.convert.lead_models)} />
            )}
            <Switch checked={cleaning.dereverb} onChange={(v) => setCleaning({ ...cleaning, dereverb: v })} label="Remove reverb / echo" />
            {cleaning.dereverb && (
              <Select value={cleaning.dereverb_model} onChange={(v) => setCleaning({ ...cleaning, dereverb_model: v })} options={sepModels(options.convert.dereverb_models)} />
            )}
            <Switch checked={cleaning.denoise} onChange={(v) => setCleaning({ ...cleaning, denoise: v })} label="Remove noise" />
            {cleaning.denoise && (
              <Select value={cleaning.denoise_model} onChange={(v) => setCleaning({ ...cleaning, denoise_model: v })} options={sepModels(options.convert.denoise_models)} />
            )}
          </Section>

          <Section title="Training">
            <Field label="Sample rate" help="40k is the best-supported choice; 48k for hi-fi singing (more VRAM), 32k for very small GPUs.">
              <ChoiceCards
                value={String(train.sample_rate)}
                onChange={(v) => setTrain({ ...train, sample_rate: Number(v) })}
                choices={options.train.sample_rates.map((sr) => ({ value: String(sr), label: `${sr / 1000} kHz` }))}
                className="[&>button]:min-h-0 [&>button]:justify-center [&>button]:py-2"
              />
            </Field>
            <Field label="Base model" help="The voice is fine-tuned from this. TITAN is cleaner but a larger download.">
              <Select value={train.pretrained} onChange={(v) => setTrain({ ...train, pretrained: v })} options={options.train.pretrained.map((p) => ({ value: p, label: p }))} />
            </Field>
            <Slider
              label="Epochs"
              help="More = closer fit of the voice until it overfits (early stopping keeps the best epoch); fewer = faster, rougher. 0 = automatic (~7000 steps)."
              value={train.epochs}
              onChange={(v) => setTrain({ ...train, epochs: v })}
              min={0}
              max={1000}
              step={10}
              format={(v) => (v ? String(v) : "auto")}
            />
            <Slider
              label="Batch size"
              help="Larger = faster, smoother training but more VRAM; smaller = fits small GPUs, noisier updates. 0 = automatic from the GPU's memory."
              value={train.batch_size}
              onChange={(v) => setTrain({ ...train, batch_size: v })}
              min={0}
              max={32}
              format={(v) => (v ? String(v) : "auto")}
            />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Learning rate" help="Higher = learns faster but can get unstable or overfit; lower = slower, safer. 1e-4 is RVC's standard.">
                <NumberInput value={train.learning_rate} onChange={(v) => setTrain({ ...train, learning_rate: v })} min={0.00001} max={0.001} step={0.00001} />
              </Field>
              <Field label="LR decay / epoch" help="Closer to 1 = keeps learning at full speed longer; lower = settles down sooner.">
                <NumberInput value={train.lr_decay} onChange={(v) => setTrain({ ...train, lr_decay: v })} min={0.99} max={1} step={0.000005} />
              </Field>
            </div>
            <Switch
              checked={train.early_stopping}
              onChange={(v) => setTrain({ ...train, early_stopping: v })}
              label="Stop when it stops improving"
              help="Scores held-out clips through the real conversion path and keeps the best epoch."
            />
            <Slider
              label="Patience"
              help="Epochs without improvement before stopping. Higher = rides out plateaus, longer runs; lower = stops sooner."
              value={train.patience}
              onChange={(v) => setTrain({ ...train, patience: v })}
              min={10}
              max={200}
              step={5}
              format={(v) => `${v} epochs`}
            />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Precision" help="auto = bf16 on RTX 30xx and newer, else fp32. fp32 = most stable, slower; fp16 = fast on older GPUs, can overflow.">
                <Select value={train.precision} onChange={(v) => setTrain({ ...train, precision: v })} options={["auto", "fp32", "bf16", "fp16"].map((p) => ({ value: p, label: p }))} />
              </Field>
              <Field label="Save every" help="Lower = more checkpoints and preview clips (more disk, slightly slower); higher = fewer.">
                <NumberInput value={train.save_every} onChange={(v) => setTrain({ ...train, save_every: Math.max(1, Math.round(v)) })} min={1} max={50} step={1} />
              </Field>
            </div>
            <Slider
              label="Spectrogram loss weight"
              help="Higher = closer, cleaner timbre, can sound flatter; lower = more freedom and more artifacts. 45 is standard."
              value={train.c_mel}
              onChange={(v) => setTrain({ ...train, c_mel: v })}
              min={1}
              max={100}
            />
            <Slider
              label="KL loss weight"
              help="Higher = smoother, more generic voice; lower = sharper detail but less stable. 1 is standard."
              value={train.c_kl}
              onChange={(v) => setTrain({ ...train, c_kl: v })}
              min={0}
              max={5}
              step={0.1}
            />
            <Switch
              checked={train.gradient_checkpointing}
              onChange={(v) => setTrain({ ...train, gradient_checkpointing: v })}
              label="Low VRAM mode"
              help="Gradient checkpointing: fits about twice the batch in the same memory, ~30% slower."
            />
            <Switch checked={train.keep_snapshots} onChange={(v) => setTrain({ ...train, keep_snapshots: v })} label="Keep a model file at every save" />
          </Section>

          <Section title="Dataset slicing">
            <Slider
              label="Longest segment"
              help="Longer = more context per step but more VRAM per item; shorter = more, smaller segments."
              value={segmenting.max_seconds}
              onChange={(v) => setSegmenting({ ...segmenting, max_seconds: v })}
              min={2}
              max={10}
              step={0.1}
              format={(v) => `${v.toFixed(1)} s`}
            />
            <Slider
              label="Shortest segment"
              help="Shorter clips are dropped. Higher = cleaner but less data; lower = keeps short words and breaths."
              value={segmenting.min_seconds}
              onChange={(v) => setSegmenting({ ...segmenting, min_seconds: v })}
              min={0.3}
              max={3}
              step={0.1}
              format={(v) => `${v.toFixed(1)} s`}
            />
            <Slider
              label="Silence threshold"
              help="Level treated as silence when slicing. Higher = cuts more aggressively and drops quiet tails; lower = keeps breaths and room tone."
              value={segmenting.silence_db}
              onChange={(v) => setSegmenting({ ...segmenting, silence_db: v })}
              min={-60}
              max={-20}
              format={(v) => `${v} dB`}
            />
            <Slider
              label="Target loudness"
              help="Every clip is normalised to this. Higher = louder with less headroom; lower = quieter, safer."
              value={segmenting.target_lufs}
              onChange={(v) => setSegmenting({ ...segmenting, target_lufs: v })}
              min={-30}
              max={-12}
              format={(v) => `${v} LUFS`}
            />
          </Section>
        </div>
      </Disclosure>

      <div className="space-y-3">
        <div className="flex gap-2">
          <Button variant="primary" size="lg" className="flex-1" onClick={() => launch("full")} loading={busy === "full"} disabled={running || !name}>
            {!running && <Play className="size-4" />}
            {running ? "Training…" : detail && detail.state !== "idle" ? "Resume training" : "Start training"}
          </Button>
          {running && (
            <Button variant="danger" size="lg" onClick={() => act("stop")} loading={busy === "stop"}>
              <Square className="size-4" /> Stop
            </Button>
          )}
        </div>
        {error && (
          <Alert kind="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </div>

      <StatusCard detail={detail} />

      {detail && (detail.samples.preview || detail.samples.reference) && (
        <div className="grid gap-3 md:grid-cols-2">
          <AudioPlayer
            src={detail.samples.preview}
            title={detail.samples.preview_epoch ? `Preview after epoch ${detail.samples.preview_epoch}` : "Latest preview"}
            compact
          />
          <AudioPlayer src={detail.samples.reference} title="Original clip" compact />
        </div>
      )}

      {detail && detail.state !== "idle" && (
        <Disclosure title="Details" summary="loss chart, dataset report, log">
          <div className="space-y-5">
            <LossChart metrics={detail.metrics} />
            {detail.samples.cleaned && <AudioPlayer src={detail.samples.cleaned} title="A cleaned training clip" compact />}
            {detail.report && <ReportTable report={detail.report} />}
            <div>
              <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-faint">Log</div>
              <pre className="max-h-64 overflow-auto rounded-xl border border-border bg-bg/70 p-3 font-mono text-[11.5px] leading-relaxed text-muted">
                {detail.log || "…"}
              </pre>
            </div>
          </div>
        </Disclosure>
      )}
    </div>
  );
}

function StatusCard({ detail }: { detail: SessionDetail | null }) {
  if (!detail || detail.state === "idle") {
    return (
      <Card>
        <CardBody className="flex items-center gap-4 text-[13.5px] text-muted">
          <span className="flex size-10 items-center justify-center rounded-full bg-surface-3 text-faint">
            <AudioLines className="size-5" />
          </span>
          {detail ? (
            <span>
              <b className="text-text">{detail.name}</b> has not been trained yet.
            </span>
          ) : (
            <span>Name the voice, add recordings and press Start. Progress shows here and keeps running if you close this page.</span>
          )}
        </CardBody>
      </Card>
    );
  }
  const overall = detail.state === "finished" ? 1 : Math.min(1, (detail.stage_index + detail.progress) / Math.max(1, detail.stage_count));
  const elapsed = detail.started
    ? formatMinutes(((detail.state === "running" ? Date.now() / 1000 : detail.updated ?? Date.now() / 1000) - detail.started))
    : null;
  const m = detail.last_metrics;
  return (
    <Card>
      <CardBody className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[15px] font-semibold">{detail.name}</span>
          <StatusPill state={detail.state} />
          {elapsed && <span className="text-[12px] text-muted">{elapsed}</span>}
          <span className="ml-auto text-[12.5px] text-muted">{detail.message}</span>
        </div>
        <ProgressBar value={overall} active={detail.state === "running"} />
        {m && (
          <div className="text-[12.5px] text-muted">
            epoch {m.epoch} · step {m.step}
            {typeof m.val_mel === "number" && ` · validation ${m.val_mel.toFixed(3)}`}
            {m.best_epoch ? ` · best epoch ${m.best_epoch}` : ""}
          </div>
        )}
        <div className="flex flex-wrap gap-1.5">
          {detail.stages.map((stage, i) => {
            const done = i < detail.stage_index || detail.state === "finished";
            const current = i === detail.stage_index && detail.state === "running";
            return (
              <span
                key={stage}
                className={cn(
                  "rounded-full border px-2.5 py-0.5 text-[11.5px]",
                  done && "border-ok/40 text-ok",
                  current && "border-accent/60 bg-accent/10 text-accent-soft",
                  !done && !current && "border-border text-faint",
                )}
              >
                {STAGE_LABELS[stage] ?? stage}
              </span>
            );
          })}
        </div>
        {detail.error && (detail.state === "failed" || detail.state === "crashed") && (
          <pre className="whitespace-pre-wrap rounded-lg border border-danger/40 bg-danger/10 p-3 font-mono text-[11.5px] text-text">{detail.error}</pre>
        )}
        {detail.state === "finished" && detail.result?.model_dir != null && (
          <Alert kind="success">The voice is ready. Find it in the Convert page.</Alert>
        )}
      </CardBody>
    </Card>
  );
}

const STAGE_LABELS: Record<string, string> = {
  prepare: "Clean & slice",
  features: "Extract features",
  train: "Train",
  index: "Build index",
  publish: "Publish",
};

function ReportTable({ report }: { report: NonNullable<SessionDetail["report"]> }) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-faint">Dataset</div>
      <p className="text-[13px]">
        <b>{(report.total_seconds / 60).toFixed(1)} min</b> of clean voice in {report.total_segments} segments at{" "}
        {report.sample_rate / 1000} kHz
      </p>
      {report.warnings.map((w) => (
        <Alert key={w} kind="warning" className="mt-2">
          {w}
        </Alert>
      ))}
      <div className="mt-2 overflow-hidden rounded-xl border border-border">
        <table className="w-full text-[12.5px]">
          <thead className="bg-surface-2 text-left text-[11px] uppercase tracking-wider text-faint">
            <tr>
              <th className="px-3 py-2 font-semibold">File</th>
              <th className="px-3 py-2 font-semibold">Length</th>
              <th className="px-3 py-2 font-semibold">Kept</th>
              <th className="px-3 py-2 font-semibold">Notes</th>
            </tr>
          </thead>
          <tbody>
            {report.files.map((f) => (
              <tr key={f.name} className="border-t border-border">
                <td className="max-w-[220px] truncate px-3 py-1.5">{f.name}</td>
                <td className="px-3 py-1.5 text-muted">{(f.seconds / 60).toFixed(1)} min</td>
                <td className="px-3 py-1.5 text-muted">{(f.kept_seconds / 60).toFixed(1)} min</td>
                <td className="px-3 py-1.5 text-muted">
                  {f.error ??
                    [...f.warnings, ...Object.entries(f.dropped).map(([k, n]) => `dropped ${n} ${k}`)].join("; ") ??
                    "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

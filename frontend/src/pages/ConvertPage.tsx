import { useEffect, useMemo, useState } from "react";
import { Mic, Music, RefreshCw, RotateCcw, Sparkles } from "lucide-react";
import { useApp } from "@/AppContext";
import { api, followTask, type ConversionSettings, type ConvertRequestOptions, type ConvertResult, type Task } from "@/lib/api";
import { describePitch } from "@/lib/utils";
import { Alert } from "@/components/ui/Alert";
import { AudioPlayer } from "@/components/ui/AudioPlayer";
import { Button } from "@/components/ui/Button";
import { Card, CardBody, Section } from "@/components/ui/Card";
import { ChoiceCards } from "@/components/ui/ChoiceCards";
import { Disclosure } from "@/components/ui/Collapsible";
import { Dropzone } from "@/components/ui/Dropzone";
import { Field, Hint } from "@/components/ui/Field";
import { ProgressBar } from "@/components/ui/Progress";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";

const BACKING = [
  { value: "keep", label: "Keep the original backing vocals" },
  { value: "convert", label: "Convert all vocals together" },
  { value: "remove", label: "Remove backing vocals" },
];

export function ConvertPage() {
  const { options, models, refreshModels } = useApp();
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [settings, setSettings] = useState<ConversionSettings | null>(null);
  const [request, setRequest] = useState<ConvertRequestOptions | null>(null);
  const [task, setTask] = useState<Task<ConvertResult> | null>(null);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    if (!options) return;
    setSettings({ ...options.convert.settings });
    setRequest({ ...options.convert.request, auto_pitch: true });
  };
  useEffect(reset, [options]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!model && models.length) setModel(models[0].name);
    if (model && !models.some((m) => m.name === model)) setModel(models[0]?.name ?? null);
  }, [models, model]);

  const info = useMemo(() => models.find((m) => m.name === model), [models, model]);
  const song = request?.mode === "song";
  const busy = task?.state === "running";

  const run = async () => {
    if (!file || !model || !settings || !request) return;
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("audio", file);
    form.append("settings", JSON.stringify({ ...request, model, settings }));
    try {
      const started = await api.form<Task<ConvertResult>>("/api/convert", form);
      setTask(started);
      const done = await followTask<ConvertResult>(started, setTask);
      setResult(done);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setTask((t) => (t && t.state === "running" ? { ...t, state: "failed" } : t));
    }
  };

  if (!options || !settings || !request) return null;
  const sepModels = (list: { label: string; note: string }[]) =>
    list.map((m) => ({ value: m.label, label: m.label, description: m.note || undefined }));

  return (
    <div className="space-y-4">
      <Card>
        <CardBody className="space-y-5">
          <Dropzone
            files={file ? [file] : []}
            onChange={(files) => setFile(files[0] ?? null)}
            accept={options.convert.audio_extensions.join(",")}
            title="Drop a song or a recording here"
            subtitle="mp3, wav, flac, m4a, video files… up to 4 GB"
          />
          {file && <AudioPlayer src={file} compact />}

          <ChoiceCards
            value={request.mode}
            onChange={(mode) => setRequest({ ...request, mode: mode as "song" | "voice" })}
            choices={[
              {
                value: "song",
                label: "Song",
                description: "Vocals with music. The vocals are separated, converted and mixed back.",
                icon: <Music className="size-4" />,
              },
              {
                value: "voice",
                label: "Voice only",
                description: "Speech or a cappella. Converted directly.",
                icon: <Mic className="size-4" />,
              },
            ]}
          />

          <Field
            label="Voice"
            help="Voices you trained or added in the Library."
            right={
              <button type="button" onClick={refreshModels} className="text-muted hover:text-text" aria-label="Refresh voices">
                <RefreshCw className="size-3.5" />
              </button>
            }
          >
            <Select
              value={model}
              onChange={setModel}
              placeholder={models.length ? "Choose a voice" : "No voices yet — train or add one in the Library"}
              options={models.map((m) => ({ value: m.name, label: m.name, description: m.summary }))}
            />
            {info && (
              <Hint>
                {info.summary} · typical pitch {describePitch(info.f0_median)}
                {!info.has_index && " · no index, timbre retrieval is off"}
              </Hint>
            )}
          </Field>
        </CardBody>
      </Card>

      <Disclosure
        title="Options"
        summary={
          request.auto_pitch
            ? "auto pitch · defaults"
            : `pitch ${settings.pitch_shift >= 0 ? "+" : ""}${settings.pitch_shift}`
        }
        action={
          <Button size="sm" variant="ghost" onClick={reset}>
            <RotateCcw className="size-3.5" /> Reset
          </Button>
        }
      >
        <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
          <Section title="Pitch">
            <Switch
              checked={request.auto_pitch}
              onChange={(v) => setRequest({ ...request, auto_pitch: v })}
              label="Auto-match the voice's range"
              help="Adds the shift that fits the voice's natural pitch: whole octaves for songs (to stay in key), any amount for speech. Off = manual shift only."
            />
            <Slider
              label="Pitch shift"
              help="+12 = one octave up (male → female), −12 = one down. Added on top of the automatic match."
              value={settings.pitch_shift}
              onChange={(v) => setSettings({ ...settings, pitch_shift: v })}
              min={-24}
              max={24}
              format={(v) => `${v > 0 ? "+" : ""}${v} st`}
            />
            <Slider
              label="Autotune"
              help="Higher = snaps the melody harder onto exact notes (robotic at 1); 0 = natural pitch."
              value={settings.autotune_strength}
              onChange={(v) => setSettings({ ...settings, autotune_strength: v })}
              min={0}
              max={1}
              step={0.05}
            />
            <Slider
              label="Pitch detection threshold"
              help="Higher = quiet or breathy parts count as unvoiced (fewer stray notes); lower = catches soft notes but may add pitch noise."
              value={settings.pitch_threshold}
              onChange={(v) => setSettings({ ...settings, pitch_threshold: v })}
              min={0.01}
              max={0.2}
              step={0.01}
            />
            <Switch
              checked={settings.refine_pitch}
              onChange={(v) => setSettings({ ...settings, refine_pitch: v })}
              label="Clean up pitch tracking"
              help="Fixes octave jumps and tiny glitches in the detected melody. Turn off only if a real leap gets smoothed away."
            />
          </Section>

          <Section title="Voice">
            <Slider
              label="Timbre retrieval"
              help="Higher = more of the trained voice's timbre, can slur words; lower = clearer pronunciation, less character. Needs an index file."
              value={settings.index_rate}
              onChange={(v) => setSettings({ ...settings, index_rate: v })}
              min={0}
              max={1}
              step={0.05}
            />
            <Slider
              label="Consonant protection"
              help="Lower = cleaner breaths and 's' sounds but less of the voice on them; 0.5 = off."
              value={settings.protect}
              onChange={(v) => setSettings({ ...settings, protect: v })}
              min={0}
              max={0.5}
              step={0.01}
            />
            <Slider
              label="Volume envelope"
              help="Lower = follows the input's loudness (more dynamics); higher = keeps the voice's own levels (more even)."
              value={settings.rms_mix_rate}
              onChange={(v) => setSettings({ ...settings, rms_mix_rate: v })}
              min={0}
              max={1}
              step={0.05}
            />
          </Section>

          <Section title="Separation & clean-up">
            {song ? (
              <>
                <Field label="Backing vocals">
                  <Select
                    value={request.backing_vocals}
                    onChange={(v) => setRequest({ ...request, backing_vocals: v as ConvertRequestOptions["backing_vocals"] })}
                    options={BACKING}
                  />
                </Field>
                <Field label="Vocal separation model">
                  <Select value={request.vocal_model} onChange={(v) => setRequest({ ...request, vocal_model: v })} options={sepModels(options.convert.vocal_models)} />
                </Field>
                {request.backing_vocals !== "convert" && (
                  <Field label="Lead / backing separation model">
                    <Select value={request.lead_model} onChange={(v) => setRequest({ ...request, lead_model: v })} options={sepModels(options.convert.lead_models)} />
                  </Field>
                )}
              </>
            ) : (
              <>
                <Switch
                  checked={request.denoise}
                  onChange={(v) => setRequest({ ...request, denoise: v })}
                  label="Remove background noise"
                  help="For phone, video or street recordings."
                />
                {request.denoise && (
                  <Field label="Denoise model">
                    <Select value={request.denoise_model} onChange={(v) => setRequest({ ...request, denoise_model: v })} options={sepModels(options.convert.denoise_models)} />
                  </Field>
                )}
              </>
            )}
            <Switch
              checked={request.dereverb}
              onChange={(v) => setRequest({ ...request, dereverb: v })}
              label="Remove reverb before converting"
              help="Dry vocals convert much more cleanly."
            />
            {request.dereverb && (
              <Field label="De-reverb model">
                <Select value={request.dereverb_model} onChange={(v) => setRequest({ ...request, dereverb_model: v })} options={sepModels(options.convert.dereverb_models)} />
              </Field>
            )}
          </Section>

          <Section title="Mix & output">
            <Slider
              label="Vocal reverb"
              help="Higher = more room around the vocal after de-reverb; 0 = completely dry."
              value={request.reverb}
              onChange={(v) => setRequest({ ...request, reverb: v })}
              min={0}
              max={1}
              step={0.05}
            />
            {song && (
              <>
                <Slider label="Vocals" value={request.vocal_gain_db} onChange={(v) => setRequest({ ...request, vocal_gain_db: v })} min={-12} max={12} step={0.5} format={(v) => `${v > 0 ? "+" : ""}${v} dB`} />
                <Slider label="Music" value={request.instrumental_gain_db} onChange={(v) => setRequest({ ...request, instrumental_gain_db: v })} min={-12} max={12} step={0.5} format={(v) => `${v > 0 ? "+" : ""}${v} dB`} />
                {request.backing_vocals === "keep" && (
                  <Slider label="Backing vocals" value={request.backing_gain_db} onChange={(v) => setRequest({ ...request, backing_gain_db: v })} min={-12} max={12} step={0.5} format={(v) => `${v > 0 ? "+" : ""}${v} dB`} />
                )}
                <Switch
                  checked={request.shift_instrumental}
                  onChange={(v) => setRequest({ ...request, shift_instrumental: v })}
                  label="Shift the music by the same amount"
                  help="For non-octave shifts, so the song stays in tune with the vocals."
                />
              </>
            )}
            <Field label="Output format">
              <Select
                value={request.output_format}
                onChange={(v) => setRequest({ ...request, output_format: v })}
                options={options.convert.formats.map((f) => ({ value: f, label: f.toUpperCase() }))}
              />
            </Field>
          </Section>
        </div>
      </Disclosure>

      <div className="space-y-3">
        <Button variant="primary" size="lg" className="w-full" onClick={run} loading={busy} disabled={!file || !model}>
          {!busy && <Sparkles className="size-4" />}
          {busy ? task?.message || "Working…" : "Convert"}
        </Button>
        {busy && task && <ProgressBar value={task.progress} active />}
        {error && (
          <Alert kind="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </div>

      {result && (
        <Card>
          <CardBody className="space-y-4">
            <AudioPlayer src={result.output} title="Result" downloadName={result.output_name} />
            {(result.pitch_shift !== 0 || result.notes.length > 0) && (
              <Hint>
                {result.pitch_shift !== 0 && `Pitch shift used: ${result.pitch_shift > 0 ? "+" : ""}${result.pitch_shift} semitones. `}
                {result.notes.join(" ")}
              </Hint>
            )}
            {(result.instrumental || result.source_vocals || result.backing) && (
              <Disclosure title="Stems">
                <div className="grid gap-3 md:grid-cols-2">
                  <AudioPlayer src={result.converted_vocals} title="Converted vocals" compact />
                  {result.instrumental && <AudioPlayer src={result.instrumental} title="Music" compact />}
                  {result.source_vocals && <AudioPlayer src={result.source_vocals} title="Original vocals" compact />}
                  {result.backing && <AudioPlayer src={result.backing} title="Backing vocals" compact />}
                </div>
              </Disclosure>
            )}
          </CardBody>
        </Card>
      )}
    </div>
  );
}

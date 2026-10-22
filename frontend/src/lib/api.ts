// Thin client for the VoiceRail HTTP API (see src/voicerail/web/server.py).

export interface DeviceInfo {
  device: string;
  name: string;
  vram_gb: number | null;
  torch_version: string;
  accelerator: string;
}

export interface SystemInfo {
  version: string;
  device: DeviceInfo;
  home: string;
}

export interface SeparationModel {
  label: string;
  filename: string;
  note: string;
}

export interface ConversionSettings {
  pitch_shift: number;
  index_rate: number;
  protect: number;
  rms_mix_rate: number;
  refine_pitch: boolean;
  autotune_strength: number;
  speaker_id: number;
  pitch_threshold: number;
}

export interface ConvertRequestOptions {
  mode: "song" | "voice";
  auto_pitch: boolean;
  vocal_model: string;
  backing_vocals: "keep" | "convert" | "remove";
  lead_model: string;
  dereverb: boolean;
  dereverb_model: string;
  denoise: boolean;
  denoise_model: string;
  vocal_gain_db: number;
  instrumental_gain_db: number;
  backing_gain_db: number;
  reverb: number;
  shift_instrumental: boolean;
  output_format: string;
}

export interface TrainPreset {
  key: string;
  label: string;
  description: string;
  cleaning: Record<string, boolean>;
}

export interface Options {
  convert: {
    settings: ConversionSettings;
    request: ConvertRequestOptions;
    vocal_models: SeparationModel[];
    lead_models: SeparationModel[];
    dereverb_models: SeparationModel[];
    denoise_models: SeparationModel[];
    formats: string[];
    audio_extensions: string[];
  };
  train: {
    config: Record<string, unknown>;
    cleaning: Record<string, unknown>;
    segmenting: Record<string, number>;
    presets: TrainPreset[];
    sample_rates: number[];
    pretrained: string[];
    default_pretrained: string;
    stages: Record<string, string>;
  };
}

export interface ModelInfo {
  name: string;
  sample_rate: number;
  version: string;
  pitch_guided: boolean;
  epochs: number | null;
  has_index: boolean;
  size_mb: number;
  source: string;
  created: number;
  f0_median: number | null;
  notes: string;
  summary: string;
}

export interface Task<T = unknown> {
  id: string;
  kind: string;
  state: "running" | "finished" | "failed";
  progress: number;
  message: string;
  result: T | null;
  error: string | null;
  updated: number;
}

export interface ConvertResult {
  output: string;
  output_name: string;
  converted_vocals: string;
  instrumental: string | null;
  backing: string | null;
  source_vocals: string | null;
  pitch_shift: number;
  notes: string[];
}

export interface SessionStatus {
  name: string;
  state: "idle" | "running" | "finished" | "stopped" | "failed" | "crashed";
  stages: string[];
  stage: string | null;
  stage_index: number;
  stage_count: number;
  progress: number;
  message: string;
  error: string | null;
  started: number | null;
  updated: number | null;
  result: Record<string, unknown> | null;
  last_metrics: Record<string, number | null> | null;
  latest?: boolean;
}

export interface FileReport {
  name: string;
  seconds: number;
  kept_seconds: number;
  segments: number;
  dropped: Record<string, number>;
  warnings: string[];
  error: string | null;
}

export interface DatasetReport {
  sample_rate: number;
  files: FileReport[];
  total_segments: number;
  total_seconds: number;
  warnings: string[];
}

export interface SessionDetail extends SessionStatus {
  metrics: Array<Record<string, number | null>>;
  report: DatasetReport | null;
  samples: {
    preview: string | null;
    preview_epoch: number | null;
    reference: string | null;
    cleaned: string | null;
  };
  log: string;
  sources: { count: number; mb: number };
  job: Record<string, unknown> | null;
}

export interface AssetInfo {
  key: string;
  mb: number;
  ready: boolean;
}

export interface StorageInfo {
  label: string;
  path: string;
  gb: number;
}

export class ApiError extends Error {}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // keep statusText
    }
    throw new ApiError(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(url: string) => fetch(url).then((r) => handle<T>(r)),
  json: <T>(url: string, method: string, body?: unknown) =>
    fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => handle<T>(r)),
  form: <T>(url: string, form: FormData) =>
    fetch(url, { method: "POST", body: form }).then((r) => handle<T>(r)),
};

/** Follow a background task until it finishes; resolves with its result. */
export function followTask<T>(task: Task, onUpdate?: (task: Task<T>) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/tasks/${task.id}/events`);
    source.onmessage = (event) => {
      const current = JSON.parse(event.data) as Task<T>;
      onUpdate?.(current);
      if (current.state === "finished") {
        source.close();
        resolve(current.result as T);
      } else if (current.state === "failed") {
        source.close();
        reject(new ApiError(current.error ?? "The task failed."));
      }
    };
    source.onerror = () => {
      source.close();
      // Fall back to polling in case the stream dropped.
      const poll = async () => {
        try {
          const current = await api.get<Task<T>>(`/api/tasks/${task.id}`);
          onUpdate?.(current);
          if (current.state === "finished") resolve(current.result as T);
          else if (current.state === "failed") reject(new ApiError(current.error ?? "The task failed."));
          else setTimeout(poll, 1000);
        } catch (error) {
          reject(error);
        }
      };
      poll();
    };
  });
}

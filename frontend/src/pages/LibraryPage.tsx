import { useCallback, useEffect, useState } from "react";
import { Check, Download, Pencil, Plus, Trash2, X } from "lucide-react";
import { useApp } from "@/AppContext";
import { api, followTask, type AssetInfo, type ModelInfo, type SessionStatus, type StorageInfo, type Task } from "@/lib/api";
import { describePitch, formatBytes } from "@/lib/utils";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { Card, CardBody, CardTitle, Section } from "@/components/ui/Card";
import { Disclosure } from "@/components/ui/Collapsible";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { Dropzone } from "@/components/ui/Dropzone";
import { Field } from "@/components/ui/Field";
import { Input } from "@/components/ui/Input";
import { ProgressBar } from "@/components/ui/Progress";
import { Select } from "@/components/ui/Select";

export function LibraryPage() {
  const { models, refreshModels, system } = useApp();
  const [notice, setNotice] = useState<{ kind: "success" | "error"; text: string } | null>(null);

  return (
    <div className="space-y-4">
      {notice && (
        <Alert kind={notice.kind} onClose={() => setNotice(null)}>
          {notice.text}
        </Alert>
      )}
      <VoicesCard models={models} refresh={refreshModels} notify={setNotice} />
      <AddVoiceCard refresh={refreshModels} notify={setNotice} />
      <Disclosure title="Base models, storage & system" summary={system ? `${system.device.name} · ${system.home}` : undefined}>
        <SystemPanel notify={setNotice} />
      </Disclosure>
    </div>
  );
}

type Notify = (n: { kind: "success" | "error"; text: string } | null) => void;

function VoicesCard({ models, refresh, notify }: { models: ModelInfo[]; refresh: () => Promise<void>; notify: Notify }) {
  const [renaming, setRenaming] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState<Record<string, Array<{ name: string; url: string; mb: number }>>>({});

  const rename = async (old: string) => {
    if (!newName.trim()) return;
    setBusy(true);
    try {
      const { name } = await api.json<{ name: string }>(`/api/models/${encodeURIComponent(old)}`, "PATCH", { name: newName });
      notify({ kind: "success", text: `Renamed to ${name}.` });
      setRenaming(null);
      await refresh();
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.json(`/api/models/${encodeURIComponent(deleting)}`, "DELETE");
      notify({ kind: "success", text: `Deleted ${deleting}.` });
      setDeleting(null);
      await refresh();
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };
  const showFiles = async (name: string) => {
    if (files[name]) {
      setFiles((current) => {
        const next = { ...current };
        delete next[name];
        return next;
      });
      return;
    }
    try {
      const list = await api.get<Array<{ name: string; url: string; mb: number }>>(`/api/models/${encodeURIComponent(name)}/files`);
      setFiles((f) => ({ ...f, [name]: list }));
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    }
  };

  return (
    <Card>
      <CardTitle>Voices</CardTitle>
      <CardBody className="pt-2">
        {models.length === 0 ? (
          <p className="py-6 text-center text-[13.5px] text-muted">No voices yet. Train one, or add model files below.</p>
        ) : (
          <ul className="divide-y divide-border">
            {models.map((m) => (
              <li key={m.name} className="py-3">
                <div className="flex items-center gap-3">
                  <div className="min-w-0 flex-1">
                    {renaming === m.name ? (
                      <div className="flex items-center gap-2">
                        <Input value={newName} onChange={(e) => setNewName(e.target.value)} className="h-9 max-w-xs" autoFocus onKeyDown={(e) => e.key === "Enter" && rename(m.name)} />
                        <Button size="sm" variant="primary" onClick={() => rename(m.name)} loading={busy}>
                          <Check className="size-3.5" />
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setRenaming(null)}>
                          <X className="size-3.5" />
                        </Button>
                      </div>
                    ) : (
                      <div className="truncate text-[14px] font-semibold">{m.name}</div>
                    )}
                    <div className="mt-0.5 text-[12.5px] text-muted">
                      {m.summary} · {describePitch(m.f0_median)} · {formatBytes(m.size_mb)} · {m.source}
                    </div>
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <Button size="sm" variant="ghost" onClick={() => showFiles(m.name)} title="Model files">
                      <Download className="size-3.5" /> Files
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setRenaming(m.name);
                        setNewName(m.name);
                      }}
                      title="Rename"
                    >
                      <Pencil className="size-3.5" />
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setDeleting(m.name)} title="Delete" className="hover:text-danger">
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                </div>
                {files[m.name] && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {files[m.name].map((f) => (
                      <a key={f.url} href={f.url} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-2.5 py-1.5 text-[12.5px] hover:border-accent/60">
                        <Download className="size-3.5 text-accent-soft" /> {f.name}
                        <span className="text-faint">{formatBytes(f.mb)}</span>
                      </a>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardBody>
      <ConfirmDialog
        open={deleting !== null}
        onOpenChange={(open) => !open && setDeleting(null)}
        title={`Delete ${deleting}?`}
        description="The model and its index are removed from the library. Training sessions are kept."
        confirmLabel="Delete"
        onConfirm={remove}
        loading={busy}
      />
    </Card>
  );
}

function AddVoiceCard({ refresh, notify }: { refresh: () => Promise<void>; notify: Notify }) {
  const [files, setFiles] = useState<File[]>([]);
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [task, setTask] = useState<Task<ModelInfo> | null>(null);

  const add = async () => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    form.append("url", url);
    form.append("name", name);
    try {
      const started = await api.form<Task<ModelInfo>>("/api/models/import", form);
      setTask(started);
      const info = await followTask<ModelInfo>(started, setTask);
      notify({ kind: "success", text: `Added ${info.name} (${info.summary}).` });
      setFiles([]);
      setUrl("");
      setName("");
      await refresh();
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    } finally {
      setTask(null);
    }
  };

  return (
    <Card>
      <CardTitle>Add a voice</CardTitle>
      <CardBody className="space-y-4 pt-2">
        <Dropzone
          files={files}
          onChange={setFiles}
          multiple
          accept=".pth,.index,.zip"
          title="Drop RVC model files here"
          subtitle=".pth with its .index, or a .zip from RVC WebUI / Applio"
        />
        <Field label="…or a download link" help="Hugging Face or any direct link to a .pth or .zip.">
          <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://huggingface.co/user/repo/resolve/main/model.zip" />
        </Field>
        <div className="flex gap-2">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (optional, defaults to the file name)" className="flex-1" />
          <Button variant="primary" onClick={add} loading={task !== null} disabled={!files.length && !url.trim()}>
            <Plus className="size-4" /> Add
          </Button>
        </div>
        {task && (
          <div className="space-y-1">
            <ProgressBar value={task.progress} active />
            <div className="text-[12px] text-muted">{task.message || "Working…"}</div>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function SystemPanel({ notify }: { notify: Notify }) {
  const { system } = useApp();
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [storage, setStorage] = useState<StorageInfo[]>([]);
  const [sessions, setSessions] = useState<SessionStatus[]>([]);
  const [session, setSession] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [task, setTask] = useState<Task | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const [a, s, t] = await Promise.all([
      api.get<AssetInfo[]>("/api/assets"),
      api.get<StorageInfo[]>("/api/storage"),
      api.get<SessionStatus[]>("/api/train/sessions"),
    ]);
    setAssets(a);
    setStorage(s);
    setSessions(t);
  }, []);
  useEffect(() => {
    load().catch((e: Error) => notify({ kind: "error", text: e.message }));
  }, [load, notify]);

  const download = async () => {
    try {
      const started = await api.json<Task>("/api/assets/download", "POST");
      setTask(started);
      await followTask(started, setTask);
      notify({ kind: "success", text: "Base models are ready." });
      await load();
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    } finally {
      setTask(null);
    }
  };
  const clearCache = async () => {
    setBusy(true);
    try {
      await api.json("/api/cache/clear", "POST");
      await load();
    } finally {
      setBusy(false);
    }
  };
  const deleteSession = async () => {
    if (!session) return;
    setBusy(true);
    try {
      await api.json(`/api/train/sessions/${encodeURIComponent(session)}`, "DELETE");
      notify({ kind: "success", text: `Deleted the training session ${session}.` });
      setSession(null);
      setConfirm(false);
      await load();
    } catch (e) {
      notify({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const ready = assets.filter((a) => a.ready).length;
  return (
    <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
      <Section title="System">
        {system && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
            <dt className="text-muted">Device</dt>
            <dd>
              {system.device.name} ({system.device.device})
              {system.device.vram_gb ? ` · ${system.device.vram_gb.toFixed(1)} GB VRAM` : ""}
            </dd>
            <dt className="text-muted">PyTorch</dt>
            <dd>
              {system.device.torch_version} · {system.device.accelerator}
            </dd>
            <dt className="text-muted">Data folder</dt>
            <dd className="break-all font-mono text-[12px]">{system.home}</dd>
          </dl>
        )}
        {system?.device.device === "cpu" && (
          <Alert kind="warning">No GPU detected. Conversion works but is slow; training on a CPU is impractical.</Alert>
        )}
      </Section>

      <Section title="Base models">
        <p className="text-[13px] text-muted">
          {ready} of {assets.length} files downloaded. They are fetched automatically when first needed; fetch the essentials now (about 0.6 GB) to avoid a wait later.
        </p>
        <Button size="sm" variant="primary" onClick={download} loading={task !== null}>
          Download essentials
        </Button>
        {task && (
          <div className="space-y-1">
            <ProgressBar value={task.progress} active />
            <div className="text-[12px] text-muted">{task.message}</div>
          </div>
        )}
        <ul className="max-h-48 overflow-auto rounded-xl border border-border text-[12px]">
          {assets.map((a) => (
            <li key={a.key} className="flex items-center justify-between border-b border-border px-3 py-1.5 last:border-0">
              <span className="truncate font-mono">{a.key}</span>
              <span className={a.ready ? "text-ok" : "text-faint"}>{a.ready ? "ready" : formatBytes(a.mb)}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Storage">
        <ul className="space-y-1 text-[13px]">
          {storage.map((s) => (
            <li key={s.label} className="flex justify-between gap-3">
              <span className="text-muted">{s.label}</span>
              <span className="font-mono text-[12.5px]">{s.gb.toFixed(2)} GB</span>
            </li>
          ))}
        </ul>
        <Button size="sm" onClick={clearCache} loading={busy}>
          Clear conversion cache
        </Button>
      </Section>

      <Section title="Delete a training session">
        <p className="text-[13px] text-muted">Removes its dataset, checkpoints and logs. Published voices are kept.</p>
        <Select value={session} onChange={setSession} options={sessions.map((s) => ({ value: s.name, label: s.name, description: s.state }))} placeholder="Choose a session" />
        <Button size="sm" variant="danger" onClick={() => setConfirm(true)} disabled={!session}>
          <Trash2 className="size-3.5" /> Delete session
        </Button>
        <ConfirmDialog
          open={confirm}
          onOpenChange={setConfirm}
          title={`Delete the session ${session}?`}
          description="Its dataset, checkpoints and logs are removed. The published voice, if any, stays in the library."
          confirmLabel="Delete"
          onConfirm={deleteSession}
          loading={busy}
        />
      </Section>
    </div>
  );
}

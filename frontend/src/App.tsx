import { useEffect, useState } from "react";
import { Cpu, Library, Mic, Sparkles, Zap } from "lucide-react";
import { AppProvider, useApp } from "./AppContext";
import { ConvertPage } from "./pages/ConvertPage";
import { TrainPage } from "./pages/TrainPage";
import { LibraryPage } from "./pages/LibraryPage";
import { Tip } from "./components/ui/Tooltip";
import { cn } from "./lib/utils";

type Page = "convert" | "train" | "library";

const PAGES: Array<{ key: Page; label: string; icon: typeof Sparkles; blurb: string }> = [
  { key: "convert", label: "Convert", icon: Sparkles, blurb: "Give a song or a recording a new voice." },
  { key: "train", label: "Train", icon: Mic, blurb: "Teach VoiceRail a voice from your recordings." },
  { key: "library", label: "Library", icon: Library, blurb: "Your voices, base models and storage." },
];

function pageFromHash(): Page {
  const key = window.location.hash.replace(/^#\/?/, "") as Page;
  return PAGES.some((p) => p.key === key) ? key : "convert";
}

function Logo() {
  return (
    <span className="flex size-8 items-center justify-center rounded-[9px] bg-gradient-to-br from-[#7c3aed] to-[#0ea5e9] shadow-[0_6px_18px_-8px_rgba(124,58,237,0.9)]">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M6 13v-2M9 16V8M12 18V6M15 15V9M18 13v-2"
          stroke="white"
          strokeWidth="2.2"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}

function DeviceBadge() {
  const { system } = useApp();
  if (!system) return null;
  const { device } = system;
  const gpu = device.device !== "cpu";
  const label = gpu
    ? `${device.name}${device.vram_gb ? ` · ${Math.round(device.vram_gb)} GB` : ""}`
    : "CPU only · slow";
  return (
    <Tip
      content={
        gpu
          ? `${device.accelerator} · PyTorch ${device.torch_version}`
          : "No usable GPU. Conversion works but is slow; training is impractical."
      }
    >
      <div
        className={cn(
          "flex items-center gap-2 rounded-lg border px-2.5 py-2 text-[12px]",
          gpu ? "border-accent/30 bg-accent/10 text-text" : "border-warn/30 bg-warn/10 text-warn",
        )}
      >
        {gpu ? <Zap className="size-3.5 text-accent-soft" /> : <Cpu className="size-3.5" />}
        <span className="truncate">{label}</span>
      </div>
    </Tip>
  );
}

function Shell() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const { system, error } = useApp();

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const current = PAGES.find((p) => p.key === page)!;

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[232px_1fr]">
      <aside className="flex items-center gap-2 border-b border-border px-4 py-3 lg:sticky lg:top-0 lg:h-screen lg:flex-col lg:items-stretch lg:gap-0 lg:border-b-0 lg:border-r lg:px-4 lg:py-6">
        <a href="#/convert" className="flex items-center gap-2.5 lg:mb-8 lg:px-2">
          <Logo />
          <span className="text-[15px] font-semibold tracking-tight">VoiceRail</span>
        </a>
        <nav className="ml-auto flex gap-1 lg:ml-0 lg:flex-col">
          {PAGES.map(({ key, label, icon: Icon }) => (
            <a
              key={key}
              href={`#/${key}`}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] font-medium transition-colors",
                page === key ? "bg-surface-3 text-text" : "text-muted hover:bg-surface-2 hover:text-text",
              )}
            >
              <Icon className={cn("size-4", page === key && "text-accent-soft")} />
              <span className="hidden sm:inline">{label}</span>
            </a>
          ))}
        </nav>
        <div className="hidden lg:mt-auto lg:block lg:space-y-2">
          <DeviceBadge />
          {system && <div className="px-2 text-[11px] text-faint">v{system.version}</div>}
        </div>
      </aside>
      <main className="mx-auto w-full max-w-[920px] px-5 py-7 sm:px-8 lg:py-10">
        <header className="mb-6">
          <h1 className="text-[26px] font-semibold tracking-tight">{current.label}</h1>
          <p className="mt-1 text-[14px] text-muted">{current.blurb}</p>
        </header>
        {error && (
          <div className="mb-4 rounded-xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm">
            Could not reach the VoiceRail server: {error}
          </div>
        )}
        {page === "convert" && <ConvertPage />}
        {page === "train" && <TrainPage />}
        {page === "library" && <LibraryPage />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}

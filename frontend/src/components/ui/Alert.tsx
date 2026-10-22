import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Alert({
  kind = "info",
  children,
  onClose,
  className,
}: {
  kind?: "info" | "success" | "error" | "warning";
  children: ReactNode;
  onClose?: () => void;
  className?: string;
}) {
  const styles = {
    info: "border-sky/30 bg-sky/10 text-text",
    success: "border-ok/30 bg-ok/10 text-text",
    error: "border-danger/40 bg-danger/10 text-text",
    warning: "border-warn/30 bg-warn/10 text-text",
  }[kind];
  const Icon = { info: Info, success: CheckCircle2, error: AlertTriangle, warning: AlertTriangle }[kind];
  const iconColor = { info: "text-sky", success: "text-ok", error: "text-danger", warning: "text-warn" }[kind];
  return (
    <div className={cn("flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-[13px] leading-snug", styles, className)}>
      <Icon className={cn("mt-0.5 size-4 shrink-0", iconColor)} />
      <div className="min-w-0 flex-1 break-words">{children}</div>
      {onClose && (
        <button type="button" onClick={onClose} className="rounded p-0.5 text-muted hover:text-text" aria-label="Dismiss">
          <X className="size-3.5" />
        </button>
      )}
    </div>
  );
}

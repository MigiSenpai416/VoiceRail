import * as DialogPrimitive from "@radix-ui/react-dialog";
import type { ReactNode } from "react";
import { Button } from "./Button";

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
  loading,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  loading?: boolean;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[2px]" />
        <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[min(92vw,420px)] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border-strong bg-surface p-5 shadow-2xl focus:outline-none">
          <DialogPrimitive.Title className="text-base font-semibold">{title}</DialogPrimitive.Title>
          {description && (
            <DialogPrimitive.Description className="mt-2 text-[13px] leading-relaxed text-muted">
              {description}
            </DialogPrimitive.Description>
          )}
          <div className="mt-5 flex justify-end gap-2">
            <DialogPrimitive.Close asChild>
              <Button variant="ghost">Cancel</Button>
            </DialogPrimitive.Close>
            <Button variant="danger" onClick={onConfirm} loading={loading}>
              {confirmLabel}
            </Button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

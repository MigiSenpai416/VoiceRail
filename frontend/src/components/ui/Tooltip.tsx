import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { Info } from "lucide-react";
import type { ReactNode } from "react";

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          sideOffset={6}
          collisionPadding={12}
          className="z-50 max-w-[300px] rounded-lg border border-border-strong bg-surface-3 px-3 py-2 text-[12.5px] leading-snug text-text shadow-xl animate-in fade-in"
        >
          {content}
          <TooltipPrimitive.Arrow className="fill-surface-3" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

/** The small ⓘ that follows a label when it has help text. */
export function InfoTip({ content }: { content: ReactNode }) {
  return (
    <Tip content={content}>
      <button
        type="button"
        tabIndex={-1}
        aria-label="More information"
        className="inline-flex align-middle text-faint hover:text-muted"
      >
        <Info className="size-3.5" />
      </button>
    </Tip>
  );
}

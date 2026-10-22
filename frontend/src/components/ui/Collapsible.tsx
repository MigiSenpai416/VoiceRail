import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";
import { ChevronRight } from "lucide-react";
import { useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A quiet disclosure row: closed by default, opens inline. */
export function Disclosure({
  title,
  summary,
  action,
  defaultOpen,
  children,
  className,
}: {
  title: ReactNode;
  summary?: ReactNode;
  action?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <CollapsiblePrimitive.Root
      open={open}
      onOpenChange={setOpen}
      className={cn("rounded-xl border border-border bg-surface-2/50", className)}
    >
      <div className="flex items-center gap-2 pr-2">
        <CollapsiblePrimitive.Trigger className="group flex min-w-0 flex-1 items-center gap-2 px-4 py-3 text-left">
          <ChevronRight
            className={cn("size-4 shrink-0 text-muted transition-transform", open && "rotate-90")}
          />
          <span className="text-[13.5px] font-semibold text-text">{title}</span>
          {summary && !open && <span className="ml-1 truncate text-[12px] text-muted">{summary}</span>}
        </CollapsiblePrimitive.Trigger>
        {action}
      </div>
      <CollapsiblePrimitive.Content className="border-t border-border px-4 pb-4 pt-4">
        {children}
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  );
}

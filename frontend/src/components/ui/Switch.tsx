import * as SwitchPrimitive from "@radix-ui/react-switch";
import type { ReactNode } from "react";
import { InfoTip } from "./Tooltip";
import { cn } from "@/lib/utils";

export function Switch({
  checked,
  onChange,
  label,
  help,
  description,
  className,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  help?: ReactNode;
  description?: ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("flex cursor-pointer items-start justify-between gap-3 py-1", className)}>
      <span className="min-w-0">
        <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-text">
          {label}
          {help && <InfoTip content={help} />}
        </span>
        {description && <span className="mt-0.5 block text-[12px] text-muted">{description}</span>}
      </span>
      <SwitchPrimitive.Root
        checked={checked}
        onCheckedChange={onChange}
        className="relative mt-0.5 h-5 w-9 shrink-0 rounded-full border border-border-strong bg-surface-3 transition-colors data-[state=checked]:border-accent data-[state=checked]:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      >
        <SwitchPrimitive.Thumb className="block size-4 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[18px]" />
      </SwitchPrimitive.Root>
    </label>
  );
}

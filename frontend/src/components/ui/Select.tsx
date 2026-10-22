import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
}

export function Select({
  value,
  onChange,
  options,
  placeholder,
  className,
  disabled,
}: {
  value: string | null;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <SelectPrimitive.Root value={value ?? undefined} onValueChange={onChange} disabled={disabled}>
      <SelectPrimitive.Trigger
        className={cn(
          "flex h-10 w-full items-center justify-between gap-2 rounded-xl border border-border bg-surface-2 px-3 text-left text-sm text-text transition-colors hover:border-border-strong focus:outline-none focus:ring-2 focus:ring-accent/25 data-[placeholder]:text-faint disabled:opacity-50",
          className,
        )}
      >
        <span className="truncate">
          <SelectPrimitive.Value placeholder={placeholder ?? "Choose…"} />
        </span>
        <SelectPrimitive.Icon>
          <ChevronDown className="size-4 shrink-0 text-muted" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={6}
          className="z-50 max-h-80 w-[var(--radix-select-trigger-width)] overflow-hidden rounded-xl border border-border-strong bg-surface-3 shadow-2xl"
        >
          <SelectPrimitive.Viewport className="p-1">
            {options.map((option) => (
              <SelectPrimitive.Item
                key={option.value}
                value={option.value}
                className="relative flex cursor-pointer select-none flex-col rounded-lg py-2 pl-8 pr-3 text-sm text-text outline-none data-[highlighted]:bg-accent/15 data-[state=checked]:text-accent-soft"
              >
                <SelectPrimitive.ItemIndicator className="absolute left-2.5 top-2.5">
                  <Check className="size-4" />
                </SelectPrimitive.ItemIndicator>
                <SelectPrimitive.ItemText>{option.label}</SelectPrimitive.ItemText>
                {option.description && (
                  <span className="text-[12px] text-muted">{option.description}</span>
                )}
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

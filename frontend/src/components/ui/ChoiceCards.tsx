import * as RadioGroup from "@radix-ui/react-radio-group";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface Choice {
  value: string;
  label: string;
  description?: string;
  icon?: ReactNode;
}

/** A row of selectable cards: the friendly alternative to a radio list. */
export function ChoiceCards({
  value,
  onChange,
  choices,
  columns,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  choices: Choice[];
  columns?: number;
  className?: string;
}) {
  return (
    <RadioGroup.Root
      value={value}
      onValueChange={onChange}
      className={cn("grid gap-2", className)}
      style={{ gridTemplateColumns: `repeat(${columns ?? choices.length}, minmax(0, 1fr))` }}
    >
      {choices.map((choice) => (
        <RadioGroup.Item
          key={choice.value}
          value={choice.value}
          className={cn(
            "group flex min-h-[64px] items-start gap-3 rounded-xl border border-border bg-surface-2 px-3.5 py-3 text-left transition-all",
            "hover:border-border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50",
            "data-[state=checked]:border-accent/70 data-[state=checked]:bg-accent/10 data-[state=checked]:shadow-[0_0_0_1px_rgba(139,92,246,0.4)]",
          )}
        >
          {choice.icon && (
            <span className="mt-0.5 text-muted group-data-[state=checked]:text-accent-soft">{choice.icon}</span>
          )}
          <span className="min-w-0">
            <span className="block text-[13.5px] font-semibold leading-tight text-text">{choice.label}</span>
            {choice.description && (
              <span className="mt-1 block text-[12px] leading-snug text-muted">{choice.description}</span>
            )}
          </span>
        </RadioGroup.Item>
      ))}
    </RadioGroup.Root>
  );
}

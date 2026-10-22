import * as SliderPrimitive from "@radix-ui/react-slider";
import { Label } from "./Field";
import type { ReactNode } from "react";

export function Slider({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
  marks,
}: {
  label: ReactNode;
  help?: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (value: number) => string;
  marks?: [string, string];
}) {
  const shown = format ? format(value) : String(value);
  return (
    <div className="min-w-0">
      <Label
        help={help}
        right={
          <input
            type="number"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => {
              const next = Number(e.target.value);
              if (!Number.isNaN(next)) onChange(Math.min(max, Math.max(min, next)));
            }}
            aria-label={`${typeof label === "string" ? label : "value"} as a number`}
            className="h-7 w-[4.5rem] rounded-md border border-border bg-surface-2 px-2 text-right font-mono text-[12px] text-text focus:border-accent/60 focus:outline-none"
          />
        }
      >
        {label}
      </Label>
      <SliderPrimitive.Root
        className="relative flex h-5 w-full touch-none select-none items-center"
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      >
        <SliderPrimitive.Track className="relative h-1.5 w-full grow rounded-full bg-surface-3">
          <SliderPrimitive.Range className="absolute h-full rounded-full bg-gradient-to-r from-accent to-sky" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          aria-label={typeof label === "string" ? label : "value"}
          className="block size-4 rounded-full border-2 border-accent bg-bg shadow transition-transform hover:scale-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
        />
      </SliderPrimitive.Root>
      <div className="mt-1 flex justify-between text-[11px] text-faint">
        <span>{marks?.[0] ?? min}</span>
        <span className="text-muted">{shown}</span>
        <span>{marks?.[1] ?? max}</span>
      </div>
    </div>
  );
}

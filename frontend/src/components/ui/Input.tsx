import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const inputClass =
  "h-10 w-full rounded-xl border border-border bg-surface-2 px-3 text-sm text-text placeholder:text-faint transition-colors focus:border-accent/60 focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:opacity-50";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...props }, ref) {
    return <input ref={ref} className={cn(inputClass, className)} {...props} />;
  },
);

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  className,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step ?? "any"}
      onChange={(e) => {
        const next = Number(e.target.value);
        if (!Number.isNaN(next)) onChange(next);
      }}
      className={cn(inputClass, "font-mono text-[13px]", className)}
    />
  );
}

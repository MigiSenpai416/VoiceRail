import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { InfoTip } from "./Tooltip";

export function Label({
  children,
  help,
  right,
  htmlFor,
  className,
}: {
  children: ReactNode;
  help?: ReactNode;
  right?: ReactNode;
  htmlFor?: string;
  className?: string;
}) {
  return (
    <div className={cn("mb-1.5 flex items-center justify-between gap-2", className)}>
      <label htmlFor={htmlFor} className="inline-flex items-center gap-1.5 text-[13px] font-medium text-text">
        {children}
        {help && <InfoTip content={help} />}
      </label>
      {right}
    </div>
  );
}

export function Field({
  label,
  help,
  right,
  children,
  className,
}: {
  label: ReactNode;
  help?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <Label help={help} right={right}>
        {label}
      </Label>
      {children}
    </div>
  );
}

export function Hint({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("mt-1.5 text-[12.5px] leading-snug text-muted", className)}>{children}</p>;
}

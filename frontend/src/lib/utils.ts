import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

export function formatMinutes(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  if (minutes >= 60) return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
  return `${minutes} min`;
}

export function formatBytes(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

export function describePitch(hz: number | null | undefined): string {
  if (!hz) return "unknown range";
  const range = hz < 145 ? "low / male" : hz < 200 ? "mid" : "high / female";
  return `~${Math.round(hz)} Hz (${range})`;
}

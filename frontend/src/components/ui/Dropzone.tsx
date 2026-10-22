import { useId, useRef, useState, type DragEvent } from "react";
import { FileAudio, Upload, X } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";

export function Dropzone({
  files,
  onChange,
  multiple,
  accept,
  title,
  subtitle,
  className,
}: {
  files: File[];
  onChange: (files: File[]) => void;
  multiple?: boolean;
  accept?: string;
  title: string;
  subtitle?: string;
  className?: string;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const add = (incoming: FileList | File[]) => {
    const list = Array.from(incoming);
    if (!list.length) return;
    onChange(multiple ? [...files, ...list] : [list[0]]);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    add(e.dataTransfer.files);
  };
  const remove = (index: number) => onChange(files.filter((_, i) => i !== index));

  return (
    <div className={className}>
      <label
        htmlFor={id}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-6 py-8 text-center transition-all",
          over
            ? "border-accent bg-accent/10"
            : "border-border-strong bg-surface-2/60 hover:border-accent/60 hover:bg-surface-2",
        )}
      >
        <span className="flex size-11 items-center justify-center rounded-full bg-accent/15 text-accent-soft">
          <Upload className="size-5" />
        </span>
        <span className="text-sm font-medium text-text">{title}</span>
        {subtitle && <span className="text-[12px] text-muted">{subtitle}</span>}
        <input
          id={id}
          ref={input}
          type="file"
          className="sr-only"
          multiple={multiple}
          accept={accept}
          onChange={(e) => {
            if (e.target.files) add(e.target.files);
            e.target.value = "";
          }}
        />
      </label>
      {files.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {files.map((file, index) => (
            <li
              key={`${file.name}-${index}`}
              className="flex items-center gap-3 rounded-lg border border-border bg-surface-2 px-3 py-2 text-[13px]"
            >
              <FileAudio className="size-4 shrink-0 text-accent-soft" />
              <span className="min-w-0 flex-1 truncate">{file.name}</span>
              <span className="text-[12px] text-muted">{formatBytes(file.size / 1024 ** 2)}</span>
              <button
                type="button"
                onClick={() => remove(index)}
                className="rounded-md p-1 text-muted hover:bg-surface-3 hover:text-text"
                aria-label={`Remove ${file.name}`}
              >
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

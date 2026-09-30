import { Star, Loader2, X } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function Banner({
  variant,
  children,
  action,
}: {
  variant: "success" | "info" | "warn" | "error";
  children: ReactNode;
  action?: ReactNode;
}) {
  const styles = {
    success: "bg-success-bg text-success-fg",
    info: "bg-info-bg text-info-fg",
    warn: "bg-warn-bg text-warn-fg",
    error: "bg-error-bg text-error-fg",
  }[variant];

  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3 text-sm",
        styles,
      )}
      role="status"
    >
      <span>{children}</span>
      {action}
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Banner
      variant="error"
      action={
        <button
          type="button"
          onClick={onRetry}
          className="rounded-md border border-error-fg/30 px-3 py-1 text-xs font-semibold uppercase tracking-wide transition-colors hover:bg-error-fg/10"
        >
          Tentar de novo
        </button>
      }
    >
      {message}
    </Banner>
  );
}

export function Loading({ label = "Carregando…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-soft">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-sm text-soft">
      {children}
    </div>
  );
}

export function Stars({
  value,
  size = 16,
  onChange,
}: {
  value: number;
  size?: number;
  onChange?: (v: number) => void;
}) {
  if (!onChange) {
    return (
      <span className="inline-flex items-center gap-0.5" aria-label={`Nota ${value}`}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Star
            key={i}
            style={{ width: size, height: size }}
            className={cn(
              i <= Math.round(value) ? "fill-star text-star" : "text-line",
            )}
            aria-hidden
          />
        ))}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((i) => (
        <button
          key={i}
          type="button"
          onClick={() => onChange(i)}
          aria-label={`${i} estrela${i > 1 ? "s" : ""}`}
          className="rounded transition-transform hover:scale-110 focus-visible:outline-2 focus-visible:outline-navy"
        >
          <Star
            style={{ width: size, height: size }}
            className={cn(i <= value ? "fill-star text-star" : "text-line")}
            aria-hidden
          />
        </button>
      ))}
    </span>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full bg-tag/10 px-2.5 py-0.5 text-[11px] font-medium text-tag">
      {children}
    </span>
  );
}

export function FilterChip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-navy px-3 py-1 text-xs font-medium text-background">
      {label}
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remover filtro ${label}`}
        className="rounded-full transition-opacity hover:opacity-70"
      >
        <X className="size-3" aria-hidden />
      </button>
    </span>
  );
}

export function PrimaryButton({
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={cn(
        "inline-flex items-center justify-center rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-background transition-colors hover:bg-brand/90 disabled:opacity-50",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function GhostButton({
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={cn(
        "inline-flex items-center justify-center rounded-lg border border-line bg-card px-3 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-muted",
        className,
      )}
    >
      {children}
    </button>
  );
}

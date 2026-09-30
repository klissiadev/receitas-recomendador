import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import {
  Banner,
  Empty,
  ErrorBanner,
  GhostButton,
  Loading,
  PrimaryButton,
  Stars,
  Tag,
} from "@/components/kit";
import { getRecipe, postRating, type RecipeDetail as Detail } from "@/api";

const NUTRI_LABELS: Array<[keyof Detail["nutrition"], string, string]> = [
  ["calories_kcal", "Calorias", "kcal"],
  ["fat_pdv", "Gordura", "% VD"],
  ["sat_fat_pdv", "Gordura sat.", "% VD"],
  ["sugar_pdv", "Açúcar", "% VD"],
  ["sodium_pdv", "Sódio", "% VD"],
  ["protein_pdv", "Proteína", "% VD"],
  ["carbs_pdv", "Carboidratos", "% VD"],
];

export function RecipeDetailView({
  recipeId,
  userId,
  onBack,
  onRated,
}: {
  recipeId: number;
  userId: number;
  onBack: () => void;
  onRated?: (recipeId: number, rating: number) => void;
}) {
  const [recipe, setRecipe] = useState<Detail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rating, setRating] = useState(0);
  const [hint, setHint] = useState(false);
  const [sending, setSending] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getRecipe(recipeId, userId)
      .then((r) => {
        setRecipe(r);
        setRating(0);
        setHint(false);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [recipeId, userId]);

  useEffect(load, [load]);

  async function submit() {
    if (rating === 0) {
      setHint(true);
      return;
    }
    setSending(true);
    try {
      await postRating(userId, recipeId, rating);
      toast.success("Avaliação enviada!");
      onRated?.(recipeId, rating);
      setRecipe((r) =>
        r
          ? { ...r, user_rating: { rating, date: new Date().toISOString().slice(0, 10) } }
          : r,
      );
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="space-y-5">
      <GhostButton onClick={onBack}>← Voltar</GhostButton>

      {loading && <Loading />}
      {!loading && error && <ErrorBanner message={error} onRetry={load} />}
      {!loading && !error && !recipe && <Empty>Receita não encontrada.</Empty>}

      {!loading && !error && recipe && (
        <article className="space-y-5">
          <header className="space-y-3">
            <h2 className="font-serif text-2xl leading-tight text-brand">{recipe.name}</h2>
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-2 rounded-full bg-badge px-3 py-1 text-sm font-semibold text-ink">
                <Stars value={recipe.avg_rating} />
                {recipe.avg_rating.toFixed(1)}
                <span className="font-normal text-soft">
                  ({recipe.n_ratings} avaliações)
                </span>
              </span>
            </div>
            <div className="flex max-h-21 flex-wrap gap-2 overflow-y-auto pr-1">
              {recipe.tags.map((t) => (
                <Tag key={t}>{t}</Tag>
              ))}
            </div>
            <p className="text-sm text-soft">
              {recipe.minutes} min · {recipe.steps.length} passos · {recipe.n_ingredients}{" "}
              ingredientes
            </p>
          </header>

          <p className="text-sm leading-relaxed text-ink">{recipe.description}</p>

          <section className="space-y-2">
            <h3 className="font-serif text-lg text-brand">Nutrição</h3>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {NUTRI_LABELS.map(([key, label, unit]) => (
                <div key={key} className="rounded-lg bg-nutri px-3 py-2">
                  <p className="text-[11px] uppercase tracking-wide text-soft">{label}</p>
                  <p className="text-sm font-semibold text-ink">
                    {recipe.nutrition[key]}
                    <span className="ml-1 text-[11px] font-normal text-soft">{unit}</span>
                  </p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-soft">VD = % do valor diário</p>
          </section>

          <section className="space-y-2">
            <h3 className="font-serif text-lg text-brand">Modo de preparo</h3>
            <ol className="space-y-2">
              {recipe.steps.map((step, i) => (
                <li key={i} className="flex gap-3 text-sm leading-relaxed text-ink">
                  <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-navy text-[11px] font-semibold text-background">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </section>

          <section className="space-y-3 border-t border-line pt-4">
            {recipe.user_rating ? (
              <Banner variant="success">
                Você já avaliou esta receita: nota {recipe.user_rating.rating}
              </Banner>
            ) : (
              <div className="flex flex-col items-start gap-4">
                <p className="text-sm font-medium text-ink">Sua avaliação</p>
                <Stars value={rating} size={28} onChange={(v) => { setRating(v); setHint(false); }} />
                {hint && <Banner variant="warn">Escolha uma nota antes de enviar.</Banner>}
                <PrimaryButton onClick={submit} disabled={sending} className="mt-2">
                  {sending ? "Enviando…" : "Enviar avaliação"}
                </PrimaryButton>
              </div>
            )}
          </section>
        </article>
      )}
    </div>
  );
}

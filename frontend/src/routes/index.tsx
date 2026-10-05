// index.tsx
import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";

import bg from "@/assets/veggies-bg.jpg";
import {
  Banner,
  Empty,
  ErrorBanner,
  FilterChip,
  GhostButton,
  Loading,
  PrimaryButton,
  Stars,
  Tag,
} from "@/components/kit";
import { RecipeDetailView } from "@/components/RecipeDetail";
import {
  createUser,
  getHistory,
  getKnnBase,
  getModels,
  getOnboarding,
  getRecommendations,
  getTags,
  getUsers,
  type HistoryItem,
  type Model,
  type RecipeSummary,
  type User,
} from "@/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Recomendação de Receitas" },
      {
        name: "description",
        content:
          "Descubra receitas recomendadas para o seu gosto, avalie pratos e acompanhe seu histórico.",
      },
      { property: "og:title", content: "Recomendação de Receitas" },
      {
        property: "og:description",
        content:
          "Descubra receitas recomendadas para o seu gosto, avalie pratos e acompanhe seu histórico.",
      },
    ],
  }),
  component: App,
});

const REQUIRED_RATINGS = 3;

const MODEL_INFO: Record<string, { icon: string; title: string; desc: string }> = {
  popularity: {
    icon: "",
    title: "Por popularidade",
    desc: "As receitas mais bem avaliadas por toda a comunidade, ideal para começar.",
  },
  item_based: {
    icon: "",
    title: "Porque você gostou de uma receita",
    desc: "Sugestões parecidas com os pratos que você já avaliou bem.",
  },
};

function App() {
  const cardRef = useRef<HTMLDivElement>(null);

  const [user, setUser] = useState<User | null>(null);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);
  const [tab, setTab] = useState<"rec" | "hist">("rec");
  const [openRecipe, setOpenRecipe] = useState<number | null>(null);

  const scrollTop = useCallback(() => {
    cardRef.current?.scrollTo({ top: 0 });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  useEffect(scrollTop, [openRecipe, scrollTop]);

  function pickUser(u: User) {
    setUser(u);
    setNeedsOnboarding(u.n_ratings < REQUIRED_RATINGS);
    setTab("rec");
    setOpenRecipe(null);
  }

  function reset() {
    setUser(null);
    setNeedsOnboarding(false);
    setOpenRecipe(null);
  }

  return (
    <main
      className="app-backdrop min-h-screen px-4 py-10"
      style={{ backgroundImage: `url(${bg})` }}
    >
      <div
        ref={cardRef}
        className="mx-auto max-w-2xl rounded-2xl border border-line bg-card/95 p-6 shadow-xl backdrop-blur-sm sm:p-8"
      >
        <h1 className="font-serif text-3xl leading-tight text-brand">
          Recomendação de Receitas
        </h1>

        {!user && <HomeScreen onPick={pickUser} />}

        {user && openRecipe !== null && (
          <div className="mt-6">
            <RecipeDetailView
              recipeId={openRecipe}
              userId={user.id}
              onBack={() => setOpenRecipe(null)}
            />
          </div>
        )}

        {user && openRecipe === null && needsOnboarding && (
          <OnboardingScreen
            user={user}
            onDone={() => setNeedsOnboarding(false)}
            onSwitch={reset}
          />
        )}

        {user && openRecipe === null && !needsOnboarding && (
          <div className="mt-6 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex gap-1 rounded-lg border border-line p-1">
                {(
                  [
                    ["rec", "Recomendações"],
                    ["hist", "Histórico"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setTab(id)}
                    className={
                      tab === id
                        ? "rounded-md bg-navy px-3 py-1.5 text-sm font-semibold text-background"
                        : "rounded-md px-3 py-1.5 text-sm font-medium text-soft hover:text-ink"
                    }
                  >
                    {label}
                  </button>
                ))}
              </div>
              <GhostButton onClick={reset}>Trocar usuário</GhostButton>
            </div>

            <p className="text-sm text-soft">
              Olá, <span className="font-semibold text-ink">{user.name}</span>.
            </p>

            {tab === "rec" ? (
              <RecommendationsTab userId={user.id} onOpen={setOpenRecipe} />
            ) : (
              <HistoryTab userId={user.id} onOpen={setOpenRecipe} />
            )}
          </div>
        )}
      </div>
    </main>
  );
}

/* ------------------------------------------------------------------ Início */

function HomeScreen({ onPick }: { onPick: (u: User) => void }) {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [warn, setWarn] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getUsers()
      .then(setUsers)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  async function create() {
    if (!name.trim()) return;
    try {
      const u = await createUser(name);
      onPick(u);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="mt-6 space-y-5">
      <p className="text-sm text-soft">
        Escolha um usuário de demonstração ou crie um novo para receber sugestões de
        receitas.
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        {[
          {
            icon: "",
            title: "Por popularidade",
            desc: "Veja as receitas mais bem avaliadas pela comunidade.",
          },
          {
            icon: "",
            title: "Feitas para você",
            desc: "Sugestões baseadas nos pratos que você já gostou.",
          },
          {
            icon: "",
            title: "Avalie e refine",
            desc: "Quanto mais você avalia, melhores ficam as recomendações.",
          },
        ].map((c) => (
          <div
            key={c.title}
            className="rounded-xl border border-line bg-card p-4 shadow-sm"
          >
            <span className="text-2xl" aria-hidden>
              {c.icon}
            </span>
            <p className="mt-2 font-serif text-sm font-semibold text-ink">{c.title}</p>
            <p className="mt-1 text-xs leading-relaxed text-soft">{c.desc}</p>
          </div>
        ))}
      </div>

      {loading && <Loading label="Carregando usuários…" />}
      {!loading && error && <ErrorBanner message={error} onRetry={load} />}

      {!loading && !error && (
        <>
          {users.length === 0 ? (
            <Empty>Nenhum usuário cadastrado ainda.</Empty>
          ) : (
            <label className="block space-y-1.5">
              <span className="text-sm font-medium text-ink">Selecionar usuário</span>
              <select
                value={selected}
                onChange={(e) => {
                  setSelected(e.target.value);
                  setWarn(false);
                }}
                className="w-full rounded-lg border border-line bg-card px-3 py-2 text-sm text-ink outline-none focus:border-navy"
              >
                <option value="">— selecione —</option>
                {users.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name} ({u.n_ratings} avaliações)
                  </option>
                ))}
              </select>
            </label>
          )}

          {warn && <Banner variant="info">Selecione um usuário para começar.</Banner>}

          <div className="flex flex-wrap items-center gap-3">
            <PrimaryButton
              onClick={() => {
                const u = users.find((x) => String(x.id) === selected);
                if (!u) {
                  setWarn(true);
                  return;
                }
                onPick(u);
              }}
            >
              Começar
            </PrimaryButton>
            <GhostButton onClick={() => setCreating((c) => !c)}>+ novo usuário</GhostButton>
          </div>

          {creating && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line p-3">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Nome do novo usuário"
                className="min-w-48 flex-1 rounded-lg border border-line bg-card px-3 py-2 text-sm outline-none focus:border-navy"
              />
              <PrimaryButton onClick={create} disabled={!name.trim()}>
                Criar
              </PrimaryButton>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* --------------------------------------------------------- Recomendações */

function RecipeRow({
  recipe,
  onOpen,
  right,
  check,
}: {
  recipe: RecipeSummary;
  onOpen: () => void;
  right?: React.ReactNode;
  check?: boolean;
}) {
  return (
    <li className="flex items-start justify-between gap-3 rounded-lg border border-line bg-card p-3">
      <div className="min-w-0 space-y-1.5">
        <p className="font-serif text-base leading-snug text-ink">
          {check && <span className="mr-1">✅</span>}
          {recipe.name}
        </p>
        <p className="text-xs text-soft">{recipe.minutes} min</p>
        <div className="flex flex-wrap gap-1.5">
          {recipe.tags.slice(0, 4).map((t) => (
            <Tag key={t}>{t}</Tag>
          ))}
        </div>
        {right}
      </div>
      <PrimaryButton className="shrink-0 px-3 py-1.5" onClick={onOpen}>
        Ver
      </PrimaryButton>
    </li>
  );
}

function RecommendationsTab({
  userId,
  onOpen,
}: {
  userId: number;
  onOpen: (id: number) => void;
}) {
  const [tags, setTags] = useState<string[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("user_based");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [items, setItems] = useState<RecipeSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [baseName, setBaseName] = useState<string | null>(null);

  useEffect(() => {
    getTags().then(setTags).catch(() => setTags([]));
    getModels().then(setModels).catch(() => setModels([]));
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getRecommendations(userId, model, 10, selectedTags)
      .then(setItems)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [userId, model, selectedTags]);

  useEffect(load, [load]);

  useEffect(() => {
    if (model !== "item_based") return setBaseName(null);
    getKnnBase(userId)
      .then((b) => setBaseName(b?.name ?? null))
      .catch(() => setBaseName(null));
  }, [userId, model]);

  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <h3 className="font-serif text-lg text-ink">Como você quer descobrir receitas?</h3>
        <div className="grid gap-3 sm:grid-cols-2">
          {models.map((m) => {
            const active = model === m.id;
            const meta = MODEL_INFO[m.id] ?? {
              icon: "",
              title: m.label,
              desc: "Sugestões personalizadas para você.",
            };
            const desc =
              m.id === "item_based" && active && baseName
                ? `Sugestões parecidas com "${baseName}", a última receita que você avaliou bem.`
                : meta.desc;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => setModel(m.id)}
                aria-pressed={active}
                className={
                  active
                    ? "rounded-xl border-2 border-navy bg-navy/5 p-4 text-left shadow-sm ring-2 ring-navy/20 transition-all"
                    : "rounded-xl border border-line bg-card p-4 text-left transition-all hover:border-navy/40 hover:shadow-sm"
                }
              >
                <span className="text-2xl" aria-hidden>
                  {meta.icon}
                </span>
                <span className="mt-2 block font-serif text-base font-semibold text-ink">
                  {meta.title}
                </span>
                <span className="mt-1 block text-xs leading-relaxed text-soft">
                  {desc}
                </span>
                {active && (
                  <span className="mt-2 inline-block rounded-full bg-navy px-2.5 py-0.5 text-[11px] font-semibold text-background">
                    Selecionado
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </section>

      <section className="space-y-2">
        <h3 className="font-serif text-lg text-ink">Filtrar por tags</h3>
        <p className="text-xs text-soft">
          Toque nas tags para refinar as sugestões. Toque de novo para remover.
        </p>
        <div className="flex max-h-21 flex-wrap gap-2 overflow-y-auto pr-1">
          {tags.map((t) => {
            const active = selectedTags.includes(t);
            return (
              <button
                key={t}
                type="button"
                onClick={() =>
                  setSelectedTags((s) =>
                    active ? s.filter((x) => x !== t) : [...s, t],
                  )
                }
                aria-pressed={active}
                className={
                  active
                    ? "rounded-full bg-navy px-3 py-1.5 text-xs font-semibold text-background shadow-sm transition-all"
                    : "rounded-full border border-line bg-card px-3 py-1.5 text-xs font-medium text-ink transition-all hover:border-navy/50 hover:text-navy"
                }
              >
                {t}
              </button>
            );
          })}
        </div>
        {selectedTags.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs font-medium text-soft">Filtros ativos:</span>
            {selectedTags.map((t) => (
              <FilterChip
                key={t}
                label={t}
                onRemove={() => setSelectedTags((s) => s.filter((x) => x !== t))}
              />
            ))}
            <button
              type="button"
              onClick={() => setSelectedTags([])}
              className="text-xs font-medium text-brand underline underline-offset-2 hover:text-brand/80"
            >
              Limpar tudo
            </button>
          </div>
        )}
      </section>

      {loading && <Loading label="Buscando recomendações…" />}
      {!loading && error && <ErrorBanner message={error} onRetry={load} />}
      {!loading && !error && items.length === 0 && (
        <Empty>Nenhuma receita encontrada com esses filtros.</Empty>
      )}
      {!loading && !error && items.length > 0 && (
        <ul className="scroll-soft max-h-[26rem] space-y-2 overflow-y-auto pr-1">
          {items.map((r) => (
            <RecipeRow key={r.id} recipe={r} onOpen={() => onOpen(r.id)} />
          ))}
        </ul>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- Histórico */

function HistoryTab({ userId, onOpen }: { userId: number; onOpen: (id: number) => void }) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getHistory(userId)
      .then(setItems)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [userId]);

  useEffect(load, [load]);

  if (loading) return <Loading label="Carregando histórico…" />;
  if (error) return <ErrorBanner message={error} onRetry={load} />;
  if (items.length === 0) return <Empty>Você ainda não avaliou nenhuma receita.</Empty>;

  return (
    <ul className="scroll-soft max-h-[26rem] space-y-2 overflow-y-auto pr-1">
      {items.map((h) => (
        <RecipeRow
          key={h.recipe.id}
          recipe={h.recipe}
          onOpen={() => onOpen(h.recipe.id)}
          right={
            <p className="flex items-center gap-2 text-xs text-soft">
              <Stars value={h.rating} size={14} /> nota {h.rating} ·{" "}
              {new Date(h.date).toLocaleDateString("pt-BR")}
            </p>
          }
        />
      ))}
    </ul>
  );
}

/* ------------------------------------------------------------ Boas-vindas */

function OnboardingScreen({
  user,
  onDone,
  onSwitch,
}: {
  user: User;
  onDone: () => void;
  onSwitch: () => void;
}) {
  const [items, setItems] = useState<RecipeSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rated, setRated] = useState<number[]>([]);
  const [detail, setDetail] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getOnboarding(12)
      .then(setItems)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const count = Math.min(rated.length + user.n_ratings, REQUIRED_RATINGS);
  const complete = rated.length + user.n_ratings >= REQUIRED_RATINGS;

  if (detail !== null) {
    return (
      <div className="mt-6">
        <RecipeDetailView
          recipeId={detail}
          userId={user.id}
          onBack={() => setDetail(null)}
          onRated={(id) => setRated((r) => (r.includes(id) ? r : [...r, id]))}
        />
      </div>
    );
  }

  return (
    <div className="mt-6 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-serif text-xl text-ink">Bem-vindo, {user.name}!</h2>
        <GhostButton onClick={onSwitch}>Trocar usuário</GhostButton>
      </div>
      <p className="text-sm text-soft">
        Avalie pelo menos {REQUIRED_RATINGS} receitas populares para liberar suas
        recomendações.
      </p>

      <Banner variant={complete ? "success" : "info"}>
        Progresso: {count}/{REQUIRED_RATINGS}
      </Banner>

      {complete && (
        <Banner
          variant="success"
          action={<PrimaryButton onClick={onDone}>Ir para recomendações</PrimaryButton>}
        >
          Pronto! Já temos avaliações suficientes para recomendar receitas para você.
        </Banner>
      )}

      {loading && <Loading label="Carregando receitas populares…" />}
      {!loading && error && <ErrorBanner message={error} onRetry={load} />}
      {!loading && !error && items.length === 0 && (
        <Empty>Nenhuma receita disponível no momento.</Empty>
      )}
      {!loading && !error && items.length > 0 && (
        <ul className="scroll-soft max-h-[24rem] space-y-2 overflow-y-auto pr-1">
          {items.map((r) => (
            <RecipeRow
              key={r.id}
              recipe={r}
              check={rated.includes(r.id)}
              onOpen={() => {
                setDetail(r.id);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * api.ts — Camada única de acesso a dados.
 * Fala com o backend FastAPI (main.py) via fetch. As assinaturas das funções
 * exportadas são as mesmas da versão mockada, então os componentes não mudam.
 */

export const API_BASE_URL = "http://localhost:8000";

export type User = { id: number; name: string; n_ratings: number };

export type Nutrition = {
  calories_kcal: number;
  fat_pdv: number;
  sugar_pdv: number;
  sodium_pdv: number;
  protein_pdv: number;
  sat_fat_pdv: number;
  carbs_pdv: number;
};

export type RecipeSummary = {
  id: number;
  name: string;
  minutes: number;
  tags: string[];
  avg_rating: number;
  n_ratings: number;
};

export type UserRating = { rating: number; date: string };

export type RecipeDetail = RecipeSummary & {
  description: string;
  n_ingredients: number;
  steps: string[];
  nutrition: Nutrition;
  user_rating: UserRating | null;
};

export type HistoryItem = {
  recipe: RecipeSummary;
  rating: number;
  date: string;
};

export type Model = { id: string; label: string };

// ---------------------------------------------------------------------------
// Helpers de HTTP
// ---------------------------------------------------------------------------

type Params = Record<string, string | number | boolean | null | undefined | string[]>;

function buildUrl(path: string, params?: Params): string {
  const url = new URL(path, API_BASE_URL);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null) continue;
      if (Array.isArray(value)) {
        // FastAPI espera parâmetros repetidos: ?tags=a&tags=b
        value.forEach((v) => url.searchParams.append(key, v));
      } else {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

/** Extrai a mensagem de erro do FastAPI ({"detail": "..."} ou lista de validação). */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d?.msg ?? String(d)).join("; ");
    }
  } catch {
    // corpo não era JSON
  }
  return `Erro ${res.status} ao falar com o servidor.`;
}

async function request<T>(
  path: string,
  options: { method?: "GET" | "POST"; params?: Params; body?: unknown } = {},
): Promise<T> {
  const { method = "GET", params, body } = options;

  // Com exactOptionalPropertyTypes, não dá para passar `body: undefined`;
  // as chaves só entram no objeto quando têm valor.
  const init: RequestInit = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }

  let res: Response;
  try {
    res = await fetch(buildUrl(path, params), init);
  } catch {
    throw new Error("Não foi possível falar com o servidor.");
  }

  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

export async function getUsers(): Promise<User[]> {
  return request<User[]>("/users");
}

export async function createUser(name: string): Promise<User> {
  return request<User>("/users", { method: "POST", body: { name: name.trim() } });
}

/** Receitas populares usadas no cadastro de um usuário novo. */
export async function getOnboarding(k = 12): Promise<RecipeSummary[]> {
  return request<RecipeSummary[]>("/onboarding", { params: { k } });
}

export async function getRecommendations(
  userId: number,
  model = "popularity",
  k = 10,
  tags: string[] = [],
): Promise<RecipeSummary[]> {
  return request<RecipeSummary[]>("/recommendations", {
    params: { user_id: userId, model, k, tags },
  });
}

export async function getKnnBase(
  userId: number,
): Promise<{ id: number; name: string } | null> {
  return request<{ id: number; name: string } | null>("/recommendations/base", {
    params: { user_id: userId },
  });
}

export async function getHistory(userId: number): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/history/${userId}`);
}

export async function getRecipe(
  recipeId: number,
  userId?: number | null,
): Promise<RecipeDetail> {
  return request<RecipeDetail>(`/recipes/${recipeId}`, {
    params: { user_id: userId ?? undefined },
  });
}

export async function postRating(
  userId: number,
  recipeId: number,
  rating: number,
): Promise<{ ok: true }> {
  return request<{ ok: true }>("/ratings", {
    method: "POST",
    body: { user_id: userId, recipe_id: recipeId, rating },
  });
}

export async function getTags(): Promise<string[]> {
  return request<string[]>("/tags");
}

export async function getModels(): Promise<Model[]> {
  return request<Model[]>("/models");
}
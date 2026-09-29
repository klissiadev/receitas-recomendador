/**
 * Camada única de acesso a dados.
 * Hoje retorna dados MOCKADOS. Para ligar num backend real, basta trocar
 * o corpo de cada função por um fetch usando API_BASE_URL.
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

const MODELS: Model[] = [
  { id: "popularity", label: "Por popularidade" },
  { id: "knn", label: "Porque você gostou de uma receita" },
];

const TAGS = [
  "30-minutes-or-less",
  "60-minutes-or-less",
  "vegetarian",
  "desserts",
  "main-dish",
  "healthy",
  "low-carb",
  "easy",
  "comfort-food",
  "side-dishes",
  "breakfast",
  "soups-stews",
];

function nutrition(
  calories_kcal: number,
  fat_pdv: number,
  sugar_pdv: number,
  sodium_pdv: number,
  protein_pdv: number,
  sat_fat_pdv: number,
  carbs_pdv: number,
): Nutrition {
  return {
    calories_kcal,
    fat_pdv,
    sugar_pdv,
    sodium_pdv,
    protein_pdv,
    sat_fat_pdv,
    carbs_pdv,
  };
}

type Seed = RecipeDetail & { popularity: number };

const RECIPES: Seed[] = [
  {
    id: 137739,
    name: "Arriba Baked Winter Squash Mexican Style",
    minutes: 55,
    tags: ["60-minutes-or-less", "side-dishes", "vegetarian", "healthy"],
    avg_rating: 4.6,
    n_ratings: 218,
    description:
      "Autumn is my favorite time of year to cook! This recipe can be prepared either spicy or sweet, your choice! Two of my posted Mexican-inspired seasonings, Mexican Seasoning Mix and Mexican Hot Chocolate Seasoning, are used.",
    n_ingredients: 7,
    steps: [
      "Make a choice and proceed with recipe",
      "Depending on size of squash, cut into half or fourths",
      "Remove seeds and membrane",
      "Season with salt, pepper and the spice mix of your choice",
      "Bake at 350 degrees for 40 to 50 minutes, until fork tender",
      "Serve warm, drizzled with butter",
    ],
    nutrition: nutrition(51.5, 0, 13, 0, 2, 0, 4),
    user_rating: null,
    popularity: 98,
  },
  {
    id: 31490,
    name: "Creamy Tomato Basil Soup",
    minutes: 40,
    tags: ["60-minutes-or-less", "soups-stews", "vegetarian", "comfort-food"],
    avg_rating: 4.8,
    n_ratings: 943,
    description:
      "A silky, restaurant-style tomato soup finished with fresh basil and a swirl of cream. Perfect with grilled cheese on a cold evening.",
    n_ingredients: 9,
    steps: [
      "Sweat the onion and garlic in butter until translucent",
      "Add canned San Marzano tomatoes and vegetable stock",
      "Simmer uncovered for 20 minutes",
      "Blend until completely smooth",
      "Stir in heavy cream and torn basil leaves",
      "Season with salt, pepper and a pinch of sugar",
    ],
    nutrition: nutrition(214.3, 21, 18, 26, 8, 34, 5),
    user_rating: null,
    popularity: 96,
  },
  {
    id: 44061,
    name: "Best Banana Bread",
    minutes: 75,
    tags: ["desserts", "breakfast", "easy", "comfort-food"],
    avg_rating: 4.9,
    n_ratings: 1512,
    description:
      "The classic, never-fails banana bread: deeply moist, heavy on the ripe bananas and just sweet enough. Freezes beautifully.",
    n_ingredients: 8,
    steps: [
      "Preheat the oven to 350 degrees and butter a loaf pan",
      "Mash four very ripe bananas in a large bowl",
      "Whisk in melted butter, sugar, egg and vanilla",
      "Fold in flour, baking soda and salt until just combined",
      "Pour into the pan and bake for 55 to 65 minutes",
      "Cool 10 minutes in the pan before turning out",
    ],
    nutrition: nutrition(312.7, 18, 96, 9, 10, 32, 17),
    user_rating: null,
    popularity: 99,
  },
  {
    id: 88512,
    name: "Garlic Butter Shrimp Scampi",
    minutes: 25,
    tags: ["30-minutes-or-less", "main-dish", "easy"],
    avg_rating: 4.7,
    n_ratings: 664,
    description:
      "Plump shrimp seared in garlic butter, deglazed with white wine and brightened with lemon and parsley. Dinner in under half an hour.",
    n_ingredients: 8,
    steps: [
      "Pat the shrimp dry and season with salt",
      "Melt butter with olive oil and sliced garlic over medium heat",
      "Add shrimp in a single layer and cook 90 seconds per side",
      "Remove shrimp and deglaze the pan with white wine",
      "Reduce by half, then whisk in lemon juice and cold butter",
      "Return shrimp to the pan and toss with parsley",
    ],
    nutrition: nutrition(386.1, 42, 3, 38, 61, 55, 2),
    user_rating: null,
    popularity: 92,
  },
  {
    id: 20983,
    name: "Roasted Brussels Sprouts With Balsamic",
    minutes: 35,
    tags: ["60-minutes-or-less", "side-dishes", "vegetarian", "healthy", "low-carb"],
    avg_rating: 4.5,
    n_ratings: 387,
    description:
      "Crispy, caramelized sprouts roasted at high heat and tossed with a sticky balsamic reduction. Converts the skeptics every time.",
    n_ingredients: 5,
    steps: [
      "Heat the oven to 425 degrees",
      "Halve the sprouts and toss with olive oil, salt and pepper",
      "Roast cut side down for 25 minutes until deeply browned",
      "Reduce balsamic vinegar with honey until syrupy",
      "Toss the hot sprouts in the glaze and serve immediately",
    ],
    nutrition: nutrition(128.4, 11, 14, 7, 12, 6, 4),
    user_rating: null,
    popularity: 88,
  },
  {
    id: 67221,
    name: "Slow Cooker Chicken Tortilla Soup",
    minutes: 240,
    tags: ["soups-stews", "main-dish", "comfort-food"],
    avg_rating: 4.6,
    n_ratings: 751,
    description:
      "Everything goes in the slow cooker in the morning and you come home to a smoky, chile-spiked soup. Top with avocado and crisp tortilla strips.",
    n_ingredients: 12,
    steps: [
      "Layer chicken breasts, beans, corn and tomatoes in the slow cooker",
      "Add chicken stock, cumin, chili powder and chipotle in adobo",
      "Cook on low for 6 hours or high for 4 hours",
      "Shred the chicken with two forks and return it to the pot",
      "Finish with lime juice and chopped cilantro",
      "Serve with tortilla strips, avocado and cotija",
    ],
    nutrition: nutrition(295.8, 14, 11, 44, 68, 12, 9),
    user_rating: null,
    popularity: 90,
  },
  {
    id: 15422,
    name: "Fluffy Buttermilk Pancakes",
    minutes: 20,
    tags: ["30-minutes-or-less", "breakfast", "easy", "comfort-food"],
    avg_rating: 4.8,
    n_ratings: 1104,
    description:
      "Tall, tender pancakes with a tangy buttermilk crumb. The trick is not overmixing the batter and letting it rest ten minutes.",
    n_ingredients: 8,
    steps: [
      "Whisk flour, sugar, baking powder, baking soda and salt",
      "Beat buttermilk, eggs and melted butter in a second bowl",
      "Combine wet and dry, leaving plenty of lumps",
      "Rest the batter for 10 minutes",
      "Cook on a buttered griddle until bubbles break on top",
      "Flip once and cook a further minute",
    ],
    nutrition: nutrition(248.9, 17, 22, 31, 16, 29, 11),
    user_rating: null,
    popularity: 94,
  },
  {
    id: 73310,
    name: "Zucchini Noodles With Pesto",
    minutes: 15,
    tags: ["30-minutes-or-less", "vegetarian", "healthy", "low-carb", "easy"],
    avg_rating: 4.3,
    n_ratings: 212,
    description:
      "Raw spiralized zucchini tossed with a bright basil and pine nut pesto. Light, fast and endlessly adaptable.",
    n_ingredients: 6,
    steps: [
      "Spiralize the zucchini and salt lightly for 10 minutes",
      "Blitz basil, pine nuts, garlic, parmesan and olive oil",
      "Pat the noodles dry with a towel",
      "Toss with the pesto and cherry tomatoes",
      "Finish with lemon zest and extra parmesan",
    ],
    nutrition: nutrition(198.2, 27, 6, 9, 14, 18, 3),
    user_rating: null,
    popularity: 80,
  },
  {
    id: 90117,
    name: "Classic Beef Chili Con Carne",
    minutes: 90,
    tags: ["main-dish", "comfort-food", "soups-stews"],
    avg_rating: 4.7,
    n_ratings: 829,
    description:
      "A deeply savory chili built on browned chuck, toasted dried chiles and a long, lazy simmer. Better on the second day.",
    n_ingredients: 13,
    steps: [
      "Brown cubed chuck in batches and set aside",
      "Soften onions, peppers and garlic in the rendered fat",
      "Add toasted ground ancho and cumin and cook one minute",
      "Return the beef with tomatoes, stock and beans",
      "Simmer partly covered for 75 minutes",
      "Adjust with salt, vinegar and a square of dark chocolate",
    ],
    nutrition: nutrition(452.6, 38, 12, 51, 84, 47, 14),
    user_rating: null,
    popularity: 91,
  },
  {
    id: 51203,
    name: "Lemon Blueberry Ricotta Cake",
    minutes: 65,
    tags: ["60-minutes-or-less", "desserts", "vegetarian"],
    avg_rating: 4.6,
    n_ratings: 341,
    description:
      "Ricotta keeps this simple loaf cake impossibly tender while lemon zest and blueberries keep it from feeling heavy.",
    n_ingredients: 10,
    steps: [
      "Cream butter, sugar and lemon zest until pale",
      "Beat in eggs one at a time, then the ricotta",
      "Fold in flour, baking powder and salt",
      "Toss blueberries in flour and fold them through",
      "Bake at 340 degrees for 50 to 60 minutes",
      "Glaze with lemon juice and icing sugar once cool",
    ],
    nutrition: nutrition(341.4, 26, 88, 13, 17, 41, 19),
    user_rating: null,
    popularity: 84,
  },
  {
    id: 12876,
    name: "Crispy Smashed Potatoes With Herbs",
    minutes: 50,
    tags: ["60-minutes-or-less", "side-dishes", "vegetarian", "easy"],
    avg_rating: 4.8,
    n_ratings: 522,
    description:
      "Boiled baby potatoes flattened and roasted until the edges shatter. Rosemary, garlic and flaky salt do the rest.",
    n_ingredients: 6,
    steps: [
      "Boil baby potatoes in salted water until fork tender",
      "Drain and dry them thoroughly",
      "Smash each one flat on an oiled baking sheet",
      "Drizzle generously with olive oil and scatter rosemary",
      "Roast at 450 degrees for 25 minutes until golden",
      "Finish with flaky salt and grated garlic",
    ],
    nutrition: nutrition(226.7, 19, 3, 22, 9, 8, 12),
    user_rating: null,
    popularity: 89,
  },
  {
    id: 60934,
    name: "Chickpea And Spinach Curry",
    minutes: 30,
    tags: ["30-minutes-or-less", "main-dish", "vegetarian", "healthy"],
    avg_rating: 4.5,
    n_ratings: 476,
    description:
      "A weeknight curry from the pantry: chickpeas simmered in coconut milk with ginger, turmeric and a mountain of spinach.",
    n_ingredients: 11,
    steps: [
      "Fry onion, ginger and garlic until soft",
      "Add curry powder, turmeric and tomato paste",
      "Stir in chickpeas and coconut milk",
      "Simmer for 12 minutes to thicken",
      "Wilt in the spinach handful by handful",
      "Finish with lime juice and serve over rice",
    ],
    nutrition: nutrition(367.9, 31, 9, 24, 33, 52, 16),
    user_rating: null,
    popularity: 86,
  },
];

// ---------------------------------------------------------------------------
// Estado em memória (simula o banco)
// ---------------------------------------------------------------------------

let users: User[] = [
  { id: 1, name: "Marina Alves", n_ratings: 14 },
  { id: 2, name: "Bruno Carvalho", n_ratings: 9 },
  { id: 3, name: "Helena Dias", n_ratings: 21 },
];

let nextUserId = 4;

type RatingRow = { user_id: number; recipe_id: number; rating: number; date: string };

let ratings: RatingRow[] = [
  { user_id: 1, recipe_id: 44061, rating: 5, date: "2026-08-14" },
  { user_id: 1, recipe_id: 31490, rating: 4, date: "2026-08-02" },
  { user_id: 1, recipe_id: 12876, rating: 5, date: "2026-07-21" },
  { user_id: 2, recipe_id: 90117, rating: 5, date: "2026-09-01" },
  { user_id: 2, recipe_id: 15422, rating: 3, date: "2026-06-30" },
  { user_id: 3, recipe_id: 73310, rating: 4, date: "2026-09-10" },
  { user_id: 3, recipe_id: 60934, rating: 5, date: "2026-08-28" },
  { user_id: 3, recipe_id: 20983, rating: 4, date: "2026-08-05" },
];

const delay = (ms = 420) => new Promise((r) => setTimeout(r, ms));

/** Falha simulada só para exercitar os estados de erro (desligada). */
const FAILURE_RATE = 0;
async function simulate<T>(value: () => T, ms?: number): Promise<T> {
  await delay(ms);
  if (Math.random() < FAILURE_RATE) {
    throw new Error("Não foi possível falar com o servidor.");
  }
  return value();
}

function toSummary(r: Seed): RecipeSummary {
  return {
    id: r.id,
    name: r.name,
    minutes: r.minutes,
    tags: r.tags,
    avg_rating: r.avg_rating,
    n_ratings: r.n_ratings,
  };
}

function scoreFor(model: string, r: Seed, userId: number): number {
  if (model === "knn") {
    const liked = ratings.filter((x) => x.user_id === userId && x.rating >= 4);
    const likedTags = new Set(
      liked.flatMap((x) => RECIPES.find((s) => s.id === x.recipe_id)?.tags ?? []),
    );
    const overlap = r.tags.filter((t) => likedTags.has(t)).length;
    return overlap * 20 + r.avg_rating * 8;
  }
  if (model === "svd") {
    return r.avg_rating * 14 + (r.id % 17);
  }
  return r.popularity;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

export async function getUsers(): Promise<User[]> {
  return simulate(() => users.map((u) => ({ ...u })));
}

export async function createUser(name: string): Promise<User> {
  return simulate(() => {
    const user: User = { id: nextUserId++, name: name.trim(), n_ratings: 0 };
    users = [...users, user];
    return { ...user };
  });
}

/** Receitas populares usadas no cadastro de um usuário novo. */
export async function getOnboarding(k = 12): Promise<RecipeSummary[]> {
  return simulate(() =>
    [...RECIPES]
      .sort((a, b) => b.popularity - a.popularity)
      .slice(0, k)
      .map(toSummary),
  );
}

export async function getRecommendations(
  userId: number,
  model = "popularity",
  k = 10,
  tags: string[] = [],
): Promise<RecipeSummary[]> {
  return simulate(() => {
    const seen = new Set(
      ratings.filter((r) => r.user_id === userId).map((r) => r.recipe_id),
    );
    return RECIPES.filter((r) => !seen.has(r.id))
      .filter((r) => tags.every((t) => r.tags.includes(t)))
      .sort((a, b) => scoreFor(model, b, userId) - scoreFor(model, a, userId))
      .slice(0, k)
      .map(toSummary);
  });
}

export async function getHistory(userId: number): Promise<HistoryItem[]> {
  return simulate(() =>
    ratings
      .filter((r) => r.user_id === userId)
      .sort((a, b) => b.date.localeCompare(a.date))
      .flatMap((r) => {
        const recipe = RECIPES.find((s) => s.id === r.recipe_id);
        return recipe
          ? [{ recipe: toSummary(recipe), rating: r.rating, date: r.date }]
          : [];
      }),
  );
}

export async function getRecipe(
  recipeId: number,
  userId?: number | null,
): Promise<RecipeDetail> {
  return simulate(() => {
    const recipe = RECIPES.find((r) => r.id === recipeId);
    if (!recipe) throw new Error("Receita não encontrada.");
    const own = userId
      ? ratings.find((r) => r.user_id === userId && r.recipe_id === recipeId)
      : undefined;
    const { popularity: _popularity, ...rest } = recipe;
    return {
      ...rest,
      tags: [...recipe.tags],
      steps: [...recipe.steps],
      nutrition: { ...recipe.nutrition },
      user_rating: own ? { rating: own.rating, date: own.date } : null,
    };
  });
}

export async function postRating(
  userId: number,
  recipeId: number,
  rating: number,
): Promise<{ ok: true }> {
  return simulate(() => {
    const date = new Date().toISOString().slice(0, 10);
    ratings = [
      ...ratings.filter((r) => !(r.user_id === userId && r.recipe_id === recipeId)),
      { user_id: userId, recipe_id: recipeId, rating, date },
    ];
    users = users.map((u) =>
      u.id === userId ? { ...u, n_ratings: u.n_ratings + 1 } : u,
    );
    return { ok: true } as const;
  });
}

export async function getTags(): Promise<string[]> {
  return simulate(() => [...TAGS]);
}

export async function getModels(): Promise<Model[]> {
  return simulate(() => MODELS.map((m) => ({ ...m })));
}

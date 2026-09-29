from dataclasses import dataclass, asdict


@dataclass
class Recipe:
    id: int
    name: str
    description: str
    tags: list
    nutrition: list
    steps: list
    predicted_rating: float | None = None   # preenchido nas recomendações
    user_rating: int | None = None          # preenchido no histórico
    source: str | None = None               # 'personalized' | 'popular'

    def to_dict(self) -> dict:
        return asdict(self)
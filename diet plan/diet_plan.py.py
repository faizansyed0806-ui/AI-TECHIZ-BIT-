# diet_plan.py
"""
Diet plan generator.

Features:
- Compute daily calorie target from TDEE + goal (loss/gain/maintenance).
- Apply macro splits (percent -> grams).
- Generate a 7-day meal plan from a small internal meal database.
- Support simple exclusions: vegetarian, dairy_free.
- Export as JSON or print to console.

How to use:
    python diet_plan.py
    or import functions and call generate_weekly_plan(...)

Author: (you)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random
import json
import math

# ---------- Constants ----------
KCAL_PER_G_PROTEIN = 4
KCAL_PER_G_CARBS = 4
KCAL_PER_G_FAT = 9

# Macro templates (percentages)
MACRO_TEMPLATES = {
    # From your spec:
    # Maintenance: 40% Carbs / 30% Protein / 30% Fat
    # Weight Loss: 40% Protein / 30% Carbs / 30% Fat
    # Muscle Gain: 40% Carbs / 30% Protein / 30% Fat
    "maintenance": {"carbs": 0.40, "protein": 0.30, "fat": 0.30},
    "weight_loss": {"protein": 0.40, "carbs": 0.30, "fat": 0.30},
    "muscle_gain": {"carbs": 0.40, "protein": 0.30, "fat": 0.30},
}

# Calorie adjustments by goal
CALORIE_ADJUSTMENTS = {
    "weight_loss": -500,    # conservative default
    "muscle_gain": +300,    # conservative default
    "maintenance": 0,
}

# ---------- Data structures ----------
@dataclass
class Meal:
    id: str
    name: str
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    tags: List[str]  # e.g., ['vegetarian', 'dairy_free']

    def macros_cal(self) -> Dict[str, float]:
        """Return macro calories breakdown from grams."""
        return {
            "protein_kcal": self.protein_g * KCAL_PER_G_PROTEIN,
            "carbs_kcal": self.carbs_g * KCAL_PER_G_CARBS,
            "fat_kcal": self.fat_g * KCAL_PER_G_FAT,
        }

# ---------- Simple meals database ----------
# NOTE: This is a small illustrative DB. Replace or expand with real foods/recipes.
MEALS_DB: List[Meal] = [
    Meal("m1", "Oatmeal with banana & peanut butter", 420, 14, 55, 15, ["vegetarian"]),
    Meal("m2", "Greek yogurt + mixed berries + granola", 380, 20, 45, 10, ["vegetarian"]),
    Meal("m3", "Grilled chicken breast + quinoa + veg", 520, 45, 40, 12, ["dairy_free"]),
    Meal("m4", "Tuna salad (olive oil) + whole grain bread", 480, 38, 36, 16, ["dairy_free"]),
    Meal("m5", "Lentil curry + brown rice", 550, 22, 78, 10, ["vegetarian", "dairy_free"]),
    Meal("m6", "Egg omelette + spinach + toast", 360, 24, 28, 16, []),
    Meal("m7", "Protein shake (whey) + banana", 300, 28, 35, 4, []),
    Meal("m8", "Veggie stir-fry with tofu + rice", 500, 20, 70, 15, ["vegetarian"]),
    Meal("m9", "Salmon + sweet potato + greens", 610, 46, 44, 22, ["dairy_free"]),
    Meal("m10", "Chickpea salad with olive oil dressing", 420, 15, 48, 18, ["vegetarian", "dairy_free"]),
    # add more meals for variety...
]

# ---------- Utility functions ----------
def round_to_int(x: float) -> int:
    """Round using standard rules but returns int."""
    # Explicit digit-by-digit safe rounding:
    return int(math.floor(x + 0.5))


def adjust_calories(tdee: int, goal: str) -> int:
    """Return adjusted daily calories based on goal."""
    adj = CALORIE_ADJUSTMENTS.get(goal, 0)
    target = tdee + adj
    return max(1200, round_to_int(target))  # protective floor


def macros_from_percentages(total_cals: int, macro_percents: Dict[str, float]) -> Dict[str, int]:
    """
    Convert macro percentages into grams per day.
    Returns dict with keys: protein_g, carbs_g, fat_g
    """
    protein_kcal = total_cals * macro_percents.get("protein", 0)
    carbs_kcal = total_cals * macro_percents.get("carbs", 0)
    fat_kcal = total_cals * macro_percents.get("fat", 0)

    protein_g = protein_kcal / KCAL_PER_G_PROTEIN
    carbs_g = carbs_kcal / KCAL_PER_G_CARBS
    fat_g = fat_kcal / KCAL_PER_G_FAT

    return {
        "protein_g": round_to_int(protein_g),
        "carbs_g": round_to_int(carbs_g),
        "fat_g": round_to_int(fat_g),
    }


def filter_meals(meals: List[Meal], exclusions: Optional[List[str]] = None) -> List[Meal]:
    """Filter meals by exclusion tags (e.g., 'vegetarian' required or 'dairy_free')."""
    if not exclusions:
        return meals[:]
    def ok(meal: Meal):
        # Exclusions here mean we require meals to have these tags if the user selected them.
        # For example: if 'vegetarian' in exclusions -> we only keep meals that have 'vegetarian' tag.
        for ex in exclusions:
            if ex == "vegetarian":
                if "vegetarian" not in meal.tags:
                    return False
            if ex == "dairy_free":
                if "dairy_free" not in meal.tags:
                    return False
        return True
    return [m for m in meals if ok(m)]


def pick_meals_for_day(
    meals: List[Meal],
    target_calories: int,
    allowed_variance: float = 0.12
) -> List[Meal]:
    """
    Select a combination of meals that sums reasonably close to target_calories.
    Approach: greedy + randomization:
      - Start with meals sorted by calories
      - Attempt to add until close to target
      - If overshoot, backtrack a bit
    """
    low = target_calories * (1 - allowed_variance)
    high = target_calories * (1 + allowed_variance)

    # simple deterministic shuffle for variety but reproducible if needed
    shuffled = sorted(meals, key=lambda m: m.calories)

    # Try different seeds to find combination
    best_combo = None
    best_diff = float("inf")

    # We'll try combos with 2-4 meals per day
    for n_meals in (3, 4, 2):  # typical breakfast/lunch/dinner (+ snack)
        # Use deterministic selection: choose evenly spaced items to cover calories
        for start in range(0, max(1, len(shuffled) - n_meals + 1)):
            combo = shuffled[start:start + n_meals]
            total = sum(m.calories for m in combo)
            diff = abs(total - target_calories)
            if low <= total <= high:
                return combo
            if diff < best_diff:
                best_diff = diff
                best_combo = combo

    # fallback: try random attempts (limited)
    for _ in range(200):
        n = random.choice([2,3,4])
        combo = random.sample(meals, min(n, len(meals)))
        total = sum(m.calories for m in combo)
        diff = abs(total - target_calories)
        if low <= total <= high:
            return combo
        if diff < best_diff:
            best_diff = diff
            best_combo = combo

    return best_combo or (meals[:1] if meals else [])


def day_macros_from_meals(meals: List[Meal]) -> Dict[str, int]:
    """Sum macros from a list of Meal objects."""
    protein = sum(m.protein_g for m in meals)
    carbs = sum(m.carbs_g for m in meals)
    fat = sum(m.fat_g for m in meals)
    kcal = sum(m.calories for m in meals)
    return {
        "calories": round_to_int(kcal),
        "protein_g": round_to_int(protein),
        "carbs_g": round_to_int(carbs),
        "fat_g": round_to_int(fat),
    }


# ---------- Main generation function ----------
def generate_weekly_plan(
    tdee: int,
    goal: str = "maintenance",
    exclusions: Optional[List[str]] = None,
    meals_db: Optional[List[Meal]] = None
) -> Dict[str, Dict]:
    """
    Generate a 7-day meal plan.
    Returns a dict with keys: 'daily_targets', 'days' (list of 7 day plans)
    """
    meals_db = meals_db or MEALS_DB
    usable_meals = filter_meals(meals_db, exclusions)

    calorie_target = adjust_calories(tdee, goal)
    macro_percents = MACRO_TEMPLATES.get(goal, MACRO_TEMPLATES["maintenance"])
    macro_targets = macros_from_percentages(calorie_target, macro_percents)

    days = []
    # distribute daily calories across meals (we'll pick combos per day)
    for day_idx in range(7):
        # simple approach: daily target same every day; could vary if desired
        combo = pick_meals_for_day(usable_meals, calorie_target)
        if not combo:
            day_entry = {
                "meals": [],
                "macros": {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
            }
        else:
            macro_sum = day_macros_from_meals(combo)
            day_entry = {
                "meals": [asdict(m) for m in combo],
                "macros": macro_sum
            }
        days.append(day_entry)

    return {
        "input": {
            "tdee": tdee,
            "goal": goal,
            "calorie_target": calorie_target,
            "macro_percentages": macro_percents,
            "macro_targets_g_per_day": macro_targets,
            "exclusions": exclusions or []
        },
        "days": days
    }


# ---------- CLI demo ----------
def pretty_print_plan(plan: Dict):
    print("=== DIET PLAN ===\n")
    inp = plan["input"]
    print(f"TDEE (input): {inp['tdee']} kcal")
    print(f"Goal: {inp['goal']}")
    print(f"Daily calorie target: {inp['calorie_target']} kcal")
    mp = inp["macro_targets_g_per_day"]
    print(f"Macro targets (g/day): Protein {mp['protein_g']} g | Carbs {mp['carbs_g']} g | Fat {mp['fat_g']} g")
    print(f"Exclusions: {', '.join(inp['exclusions']) if inp['exclusions'] else 'None'}")
    print("\n7-day plan summary:\n")
    for i, day in enumerate(plan["days"], start=1):
        macros = day["macros"]
        print(f"Day {i}: {macros['calories']} kcal  |  P {macros['protein_g']}g  C {macros['carbs_g']}g  F {macros['fat_g']}g")
        for m in day["meals"]:
            print(f"  - {m['name']} ({m['calories']} kcal)")
        print("")


if __name__ == "__main__":
    # Example usage
    # Suppose TDEE = 2400 kcal, user wants weight loss, vegetarian
    example_tdee = 2400
    plan = generate_weekly_plan(example_tdee, goal="weight_loss", exclusions=["vegetarian"])
    pretty_print_plan(plan)

    # Export example as JSON if desired
    with open("7_day_plan_example.json", "w") as f:
        json.dump(plan, f, indent=2)
    print("Saved 7_day_plan_example.json")

# tdee.py
# Calorie Maintenance (TDEE) Calculator

def calculate_bmr(weight, height, age, sex):
    """
    Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation.
    weight: in kilograms
    height: in centimeters
    age: in years
    sex: 'male' or 'female'
    """
    if sex.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif sex.lower() == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Sex must be 'male' or 'female'")
    return bmr


def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure (TDEE).
    activity_level: one of ['sedentary', 'light', 'moderate', 'active', 'extreme']
    """
    multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'extreme': 1.9
    }
    if activity_level not in multipliers:
        raise ValueError("Invalid activity level. Choose from: sedentary, light, moderate, active, extreme")
    return bmr * multipliers[activity_level]


if __name__ == "__main__":
    # Example usage
    weight = 70      # kg
    height = 175     # cm
    age = 25         # years
    sex = 'male'
    activity_level = 'moderate'

    bmr = calculate_bmr(weight, height, age, sex)
    tdee = calculate_tdee(bmr, activity_level)

    print(f"BMR: {bmr:.2f} kcal/day")
    print(f"TDEE (Maintenance Calories): {tdee:.2f} kcal/day")
# Defining MDP

# Possible states
states = ["RU8p", "TU10p", "RU10p", "RD10p", "RU8a", "RD8a", "TU10a", "RU10a", "RD10a", "TD10a", "11aFin"]

# Possible actions from each state.
actions = {
    "RU8p": ["P", "R", "S"],
    "TU10p": ["P", "R"],
    "RU10p": ["R", "P", "S"],
    "RD10p": ["R", "P"],
    "RU8a": ["P", "R", "S"],
    "RD8a": ["R", "P"],
    "TU10a": ["P", "R", "S"],
    "RU10a": ["P", "R", "S"],
    "RD10a": ["P", "R", "S"],
    "TD10a": ["P", "R", "S"],
    "11aFin": []
}
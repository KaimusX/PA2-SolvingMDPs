# part2 PA2
# authors: Luis Franco and Diego de Leon
# IDs: 80677301(LUIS) and 80754838(DIEGO)

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

# Transitions. syntax: [(probability, next state, reward), ...]
P = {
    "RU8p": {
        "P": [(1.0, "TU10p", 2)],
        "R": [(1.0, "RU10p", 0)],
        "S": [(1.0, "RD10p", -1)]
    },
    "TU10p": {
        "P": [(1.0, "RU10a", 2)],
        "R": [(1.0, "RU8a", 0)]
    },
    "RU10p": {
        "R": [(1.0, "RU8a", 0)],
        "P": [(0.5, "RU8a", 2), (0.5, "RU10a", 2)],
        "S": [(1.0, "RD8a", -1)]
    },
    "RD10p": {
        "R": [(1.0, "RD8a", 0)],
        "P": [(0.5, "RD8a", 2), (0.5, "RD10a", 2)]
    },
    "RU8a": {
        "P": [(1.0, "TU10a", 2)],
        "R": [(1.0, "RU10a", 0)],
        "S": [(1.0, "RD10a", -1)]
    },
    "RD8a": {
        "R": [(1.0, "RD10a", 0)],
        "P": [(1.0, "TD10a", 2)]
    },
    "TU10a": {
        "P": [(1.0, "11aFin", -1)],
        "R": [(1.0, "11aFin", -1)],
        "S": [(1.0, "11aFin", -1)]
    },
    "RU10a": {
        "P": [(1.0, "11aFin", 0)],
        "R": [(1.0, "11aFin", 0)],
        "S": [(1.0, "11aFin", 0)]
    },
    "RD10a": {
        "P": [(1.0, "11aFin", 4)],
        "R": [(1.0, "11aFin", 4)],
        "S": [(1.0, "11aFin", 4)]
    },
    "TD10a": {
        "P": [(1.0, "11aFin", 3)],
        "R": [(1.0, "11aFin", 3)],
        "S": [(1.0, "11aFin", 3)]
    },
    "11aFin": {}
}

# threshold
threshold = 0.001

# learning rate
alpha = 0.1

#discount factor
lambdaDR = 0.99

# epsilon greedy 80% best, 20% random
epsilon = 0.2

start_state = "RU8p"

# init Q(s,a) values to 0
Q = {}
for s in states:
    Q[s] = {}
    for a in actions[s]:
        Q[s][a] = 0.0
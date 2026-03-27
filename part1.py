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

# Transitions. syntax: [(probablitty, next state, reward)...]
P = {
    #level 0 states
    "RU8p": {
        "P": [(1.0, "TU10p", 2)],
        "R": [(1.0, "RU10p", 0)],
        "S": [(1.0, "RD10p", -1)]
    },
    # level 1 states
    "TU10p": {
        "P": [(1.0, "RU10a", 2)],
        "R": [(1.0, "RU8a", 0)]
    },
    "RU10p": {
        "R": [(1.0, "RU8a", 0)],
        "P": [(0.5, "RU8a",2), (0.5, "RU10a", 2)],
        "S": [(1.0, "RD8a", -1)]
    },
    "RD10p": {
        "R": [(1.0, "RD8a", 0)],
        "P": [(0.5, "RD8a", 2), (0.5, "RD10a", 2)]
    },
    # level 2 states
    "RU8a": {
        "P":[(1.0,"TU10a",2)]
        "R":[(1.0,"RU10a",0)]
        "S":[(1.0,"RD10a",-1)]
    "RD8a":{
        "R":[(1,"RD10a",0)]
        "P":[(1,"TD10a",2)]
    
    # level 3 states
    "TU10a":{
        "any": [(1.0,"11aFin",-1)]

    "RU10a":{
        "any":[1.0,"11aFin",0]
        
    "RD10a":{
        "any":[(1.0,"11aFin",4)]

    "TD10a":{
        "any":[(1.0,"11aFin",3)]
    # level 4 state, terminal
    "11aFin": {}
}

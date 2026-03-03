import random

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# Dictionary containing 100 BTC BIP-39 seed words
word_dict = {
    0: "oxygen", 1: "castle", 2: "gravity", 3: "dolphin", 4: "forest",
    5: "guitar", 6: "helmet", 7: "island", 8: "jungle", 9: "kingdom",
    10: "lemon", 11: "magnet", 12: "nephew", 13: "orbit", 14: "pencil",
    15: "quantum", 16: "rocket", 17: "satoshi", 18: "tunnel", 19: "unfair",
    20: "velvet", 21: "whisper", 22: "yellow", 23: "zebra", 24: "artist",
    25: "balance", 26: "camera", 27: "desert", 28: "energy", 29: "fabric",
    30: "galaxy", 31: "harbor", 32: "ignore", 33: "jacket", 34: "kitten",
    35: "ladder", 36: "manual", 37: "nature", 38: "object", 39: "planet",
    40: "quiet", 41: "rescue", 42: "shadow", 43: "talent", 44: "update",
    45: "vacuum", 46: "window", 47: "youth", 48: "anchor", 49: "battle",
    50: "circle", 51: "damage", 52: "escape", 53: "future", 54: "globe",
    55: "hammer", 56: "impact", 57: "jewel", 58: "knife", 59: "launch",
    60: "memory", 61: "normal", 62: "ocean", 63: "palace", 64: "random",
    65: "school", 66: "travel", 67: "unique", 68: "victory", 69: "wealth",
    70: "zone", 71: "brick", 72: "credit", 73: "drama", 74: "elite",
    75: "flame", 76: "giant", 77: "honor", 78: "index", 79: "judge",
    80: "legend", 81: "motion", 82: "novel", 83: "option", 84: "pride",
    85: "rare", 86: "signal", 87: "theme", 88: "urban", 89: "value",
    90: "wolf", 91: "bonus", 92: "climb", 93: "dream", 94: "event",
    95: "focus", 96: "green", 97: "host", 98: "iron", 99: "logic"
}

selected_word = ""
string_chars = []
board_string = ""


# ============================================================
# wordPicker()
# ------------------------------------------------------------
# 1. Generate random index from dictionary range
# 2. Store selected word in global variable
# 3. Break word into list of characters with indexes
# ============================================================

def wordPicker():
    global selected_word, string_chars
    
    rand_index = random.randint(0, len(word_dict) - 1)
    selected_word = word_dict[rand_index]
    string_chars = list(selected_word)
    
    return selected_word


# ============================================================
# playString()
# ------------------------------------------------------------
# 1. Create board with underscores for each letter
# 2. Separate each underscore by space
# 3. Save and return board as string
# ============================================================

def playString():
    global board_string
    
    board_string = "_ " * len(string_chars)
    return board_string.strip()


# ============================================================
# updateBoard(playerGuess, current_board)
# ------------------------------------------------------------
# 1. Compare guessed character with each character in word
# 2. Replace matching underscore with correct letter
# 3. Return updated board and number of matches
# ============================================================

def updateBoard(playerGuess, current_board):
    board_list = current_board.split(" ")
    matches = 0
    
    for index, char in enumerate(string_chars):
        if playerGuess == char and board_list[index] == "_":
            board_list[index] = char
            matches += 1
    
    return " ".join(board_list), matches
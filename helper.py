import random

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# Dictionary containing 100 BTC BIP-39 seed words
word_dict = {
    0: "abandon", 1: "ability", 2: "able", 3: "about", 4: "above",
    5: "absent", 6: "absorb", 7: "abstract", 8: "absurd", 9: "abuse",
    10: "access", 11: "accident", 12: "account", 13: "accuse", 14: "achieve",
    15: "acid", 16: "acoustic", 17: "acquire", 18: "across", 19: "act",
    20: "action", 21: "actor", 22: "actress", 23: "actual", 24: "adapt",
    25: "add", 26: "addict", 27: "address", 28: "adjust", 29: "admit",
    30: "adult", 31: "advance", 32: "advice", 33: "aerobic", 34: "affair",
    35: "afford", 36: "afraid", 37: "again", 38: "age", 39: "agent",
    40: "agree", 41: "ahead", 42: "aim", 43: "air", 44: "airport",
    45: "aisle", 46: "alarm", 47: "album", 48: "alcohol", 49: "alert",
    50: "alien", 51: "all", 52: "alley", 53: "allow", 54: "almost",
    55: "alone", 56: "alpha", 57: "already", 58: "also", 59: "alter",
    60: "always", 61: "amateur", 62: "amazing", 63: "among", 64: "amount",
    65: "amused", 66: "analyst", 67: "anchor", 68: "ancient", 69: "anger",
    70: "angle", 71: "angry", 72: "animal", 73: "ankle", 74: "announce",
    75: "annual", 76: "another", 77: "answer", 78: "antenna", 79: "antique",
    80: "anxiety", 81: "any", 82: "apart", 83: "apology", 84: "appear",
    85: "apple", 86: "approve", 87: "april", 88: "arch", 89: "arctic",
    90: "area", 91: "arena", 92: "argue", 93: "arm", 94: "armed",
    95: "armor", 96: "army", 97: "around", 98: "arrange", 99: "arrest"
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
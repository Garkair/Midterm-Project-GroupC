import helper

# ============================================================
# MAIN GAME LOOP
# ============================================================

def main():
    
    print("==== Welcome to BTC Seed Word Guess Game ====")
    
    # Select random word
    helper.wordPicker()
    
    # Choose number of lives
    lives = int(input("Choose difficulty (3, 5, 10 lives): "))
    
    # Generate board
    board = helper.playString()
    
    guessed_letters = []
    total_guessed = 0
    
    # Continue until win or lose
    while lives > 0:
        
        print("\nLives:", lives)
        print("Word:", board)
        print("Guessed Letters:", guessed_letters)
        
        playerGuess = input("Enter a letter: ").lower()
        
        if playerGuess in guessed_letters:
            print("Already guessed that letter!")
            continue
        
        guessed_letters.append(playerGuess)
        
        # Update board
        updated_board, matches = helper.updateBoard(playerGuess, board)
        
        if matches > 0:
            print("Correct guess!")
          #  lives += 1  # Add life if correct
            total_guessed += matches
            board = updated_board
        else:
            print("Wrong guess!")
            lives -= 1
        
        # Win condition
        if total_guessed == len(helper.string_chars):
            print("\n🎉 You Win!")
            print("The word was:", helper.selected_word)
            break
    
    # Lose condition
    if lives == 0:
        print("\n💀 You Lost!")
        print("The word was:", helper.selected_word)
    
    replay()


# ============================================================
# Replay Function
# ------------------------------------------------------------
# Ask user if they want to play again
# If yes → restart main()
# If no → exit program
# ============================================================

def replay():
    choice = input("\nPlay again? (yes/no): ").lower()
    if choice == "yes":
        main()
    else:
        print("Thanks for playing!")
        exit()


# Run game
if __name__ == "__main__":
    main()


    # add a warning that you are inputting the same letter.
HANGMAN GAME

Project Overview

This is a Python Hangman game, using a randomly selected set of Bitcoin (BIP-39) seed words. The game challenges players to guess a hidden word, letter by letter, while managing a limited number of lives. Correct guesses maintain player lives, and the game continues until the player either wins or loses.

Features

Randomly selects a word from a pool of 100 BTC seed words.

Displays a playing board with underscores for unguessed letters.

Tracks letters guessed by the player.

Lives system (3, 5, or 10) with +1 life awarded for correct guesses.

Replay option to start a new game without restarting the program.

Modular design:

main.py – controls gameplay.

helper.py – contains reusable functions like wordPicker() and updateBoard().

Project Structure
btc_word_game/
│
├── main.py          # Main game controller
├── helper.py        # Reusable helper functions
├── README.md        # Project overview and instructions
└── __pycache__/     # Python auto-generated cache files
Gameplay Instructions

Run main.py.

Select the number of lives (3, 5, or 10).

Guess letters one at a time until the word is fully revealed or lives run out.

Correct guesses add a life; incorrect guesses reduce lives.

At the end, you can choose to play again with a new random word.

Pseudocode Collaboration

Each team member is contributing a pseudocode file that outlines their approach to implementing the game. These pseudocode files will be:

Compared across the team.

Tested with different AI-assisted implementations.

Used to select and integrate the best features into the final version of the game.

Technical Details

Python 3.x required.

Uses standard Python libraries (random).

Modular design supports easy updates and feature expansion.

Future Enhancements

Add ASCII art for the hangman/lives display.

Implement difficulty levels that adjust word length and lives.

Expand the word pool to the full 2048 BIP-39 seed list.

Add a GUI version using tkinter or PyQt.

Credits

Developed by:

Tim van Egmond

Noah Myers

Juan D. Silvera

With collaboration and AI-assisted development for feature optimization.
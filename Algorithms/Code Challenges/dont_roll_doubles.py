#Don't roll doubles 
"""
John is playing a dice game. The rules are as follows.

    Roll two dice.
    Add the numbers on the dice together.
    Add the total to your overall score.
    Repeat this for three rounds.

But if you roll DOUBLES, your score is instantly wiped to 0 and your game ends immediately!

Create a function that takes in a list of tuples as input, and return John's score after his game has ended.


Example: 
dice_game([(1, 2), (3, 4), (5, 6)]) ➞ 21

dice_game([(1, 1), (5, 6), (6, 4)]) ➞ 0

dice_game([(4, 5), (4, 5), (4, 5)]) ➞ 27
"""
def double_canceller(outcomes):
    #Input is a list containing tuples of outcomes
    total = []
    for i in range(len(outcomes)):
        if sum(outcomes[i]) == outcomes[i][0]*2:
            print("Game terminated! Resulting sum is 0...")
            break
        else:
            total.append(sum(outcomes[i]))
    print("Total: {}".format(sum(total)))


double_canceller([(1, 2), (3, 3), (6, 4)])

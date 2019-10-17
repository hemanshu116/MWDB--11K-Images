"""
TO-DO:
1. SIFT needs to be fixed.
2. LDA needs to be implemented.
3. SVD needs to be implemented.
4. NMF needs to be implemented.
"""

from Main.tasks.Task1 import startTask1
from Main.tasks.Task2 import startTask2
from Main.tasks.Task3 import startTask3
from Main.tasks.Task4 import startTask4
from Main.tasks.Task5 import startTask5
from Main.tasks.Task6 import startTask6
from Main.tasks.Task7 import startTask7
from Main.tasks.Task8 import startTask8

runAgain = True
print('Run again?', runAgain)

while runAgain:
    print("Select one of the below")
    print("1. Task 1")
    print("2. Task 2")
    print("3. Task 3")
    print("4. Task 4")
    print("5. Task 5")
    print("6. Task 6")
    print("7. Task 7")
    print("8. Task 8")
    print("Any other number to exit")
    userInput = input()
    if int(userInput) == 1:
        startTask1()
    elif int(userInput) == 2:
        startTask2()
    elif int(userInput) == 3:
        startTask3()
    elif int(userInput) == 4:
        startTask4()
    elif int(userInput) == 5:
        startTask5()
    elif int(userInput) == 6:
        startTask6()
    elif int(userInput) == 7:
        startTask7()
    elif int(userInput) == 8:
        startTask8()
    else:
        runAgain = False

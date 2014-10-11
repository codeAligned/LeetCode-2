__author__ = 'leon'

import random
# random.choice([somelist])

class Ghost(object):
    # your code goes here
    def __init__(self):
        self.colorlist = ["white", "yellow", "purple", "red"]
        self.color = self.colorlist[random.randint(0, 3)]


# Best Practice
class Ghost2(object):
    def __init__(self):
        self.color = random.choice(["white", "yellow", "purple", "red"])


print Ghost().color
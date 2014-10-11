__author__ = 'leon'

# Topic: Default value

class Ball(object):
    # your code goes here
    def __init__(self, ball_type="regular"):
        self.ball_type = ball_type


print Ball("super").ball_type

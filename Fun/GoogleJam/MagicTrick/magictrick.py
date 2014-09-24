__author__ = 'leon'

def magictrick(f):
    input = f.read().splitlines()
    output = open('output.out', 'w')
    lines = len(input)
    totalcases = input[0]
    curr = 1
    currcase = 1
    while curr < lines:
        # First round
        # print input[curr]
        row1 = int(input[curr])
        curr += row1
        tmp = input[curr].split()
        #print tmp

        # jump to next round
        curr += 5 - row1

        # Second round
        #print input[curr]
        row2 = int(input[curr])
        curr += row2
        possible = []
        for s in input[curr].split():
            if s in tmp:
                possible.append(s)

        # output
        if len(possible) == 1:  # good
            #print 'Case #'+str(currcase)+':' , possible[0]
            output.writelines('Case #' + str(currcase) + ': ' + possible[0] + '\n')
        elif len(possible) == 0:  # Volunteer cheated!
            #print 'Case #'+str(currcase)+':' , 'Volunteer cheated!'
            output.writelines('Case #' + str(currcase) + ': ' + 'Volunteer cheated!' + '\n')
        elif len(possible) > 1:  # Bad magician!
            #print 'Case #'+str(currcase)+':' , 'Bad magician!'
            output.writelines('Case #' + str(currcase) + ': ' + 'Bad magician!' + '\n')

        currcase += 1;
        curr += 5 - row2


f = open('testcase.in')
magictrick(f)
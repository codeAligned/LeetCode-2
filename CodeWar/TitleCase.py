__author__ = 'leon'

# x.title(), x.capitalize(), x.lower()

def title_case(title, minor_words=""):
    title = title.lower()
    wordlist = title.split()
    minor_words = minor_words.lower()
    minorlist = minor_words.split()
    if len(wordlist) == 0:
        return title

    wordlist[0] = wordlist[0].title()
    for i in range(1, len(wordlist)):
        if wordlist[i] not in minorlist:
            wordlist[i] = wordlist[i].title()
        else:
            wordlist[i] = wordlist[i].lower()

    return ' '.join(wordlist)

# Best Practices
def title_case2(title, minor_words=''):
    title = title.capitalize().split()
    minor_words = minor_words.lower().split()
    return ' '.join([(word if word in minor_words else word.capitalize()) for word in title])

print title_case2('a clash of KINGS', 'a an the of')
print title_case2('THE WIND IN THE WILLOWS', 'The In')
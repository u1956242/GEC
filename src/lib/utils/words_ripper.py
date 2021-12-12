import string

# Return a list of pairs of the different ways to split the word in two words
# Example: split('pen') -> [('', 'pen'), ('p', 'en'), ('pe', 'n'), ('pen', '')]
def split(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

# Return a list of string deleting each letter of the word
# Example: delete('pen') -> ['en', 'pn', 'pe']
def delete(word):
    return [l + r[1:] for l, r in split(word) if r]

# Return a list of string swapping each letter pair of the word
# Example: swap('pen') -> ['epn', 'pne']
def swap(word):
    return [l + r[1] + r[0] + r[2:] for l, r in split(word) if len(r) > 1]

# Return a list of string replacing each letter of the word to each letter of the ascii
# Example: replace('pen') -> ['aen', 'ben', 'cen', 'den', ..., 'pex', 'pey', 'pez']
def replace(word):
    letters = string.ascii_lowercase
    return [l + c + r[1:] for l, r in split(word) if r for c in letters]

# Return a list of string inserting each letter of the ascii before each letter of the word
# Example: insert('pen' -> ['apen', 'bpen', 'cpen', 'dpen', ..., 'penx', 'peny', 'penz']
def insert(word):
    letters = string.ascii_lowercase
    return [l + c + r for l, r in split(word) for c in letters]

# Return a set of string of all the different ways to rip a word
def ripper_level_one(word, extended = False):
    if extended:
        return set(delete(word) + swap(word) + replace(word) + insert(word))
    else:
        return set(delete(word) + swap(word))
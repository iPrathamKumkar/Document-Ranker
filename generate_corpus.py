import sys

file_name = sys.argv[1]
with open(file_name, "r") as corpus:
    new_corpus = corpus.read().replace("\n", "\n\n")
    print(new_corpus)

    f = open("corpus.txt", "w+")
    f.write(new_corpus)

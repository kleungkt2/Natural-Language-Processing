# author: Leung Ko Tsun
# student_id: 20516287
def q1():
    print('q1: {:}'.format(''))
    from nltk.corpus import gutenberg as gb
    import nltk
    file_id = 'austen-sense.txt'
    word_list = gb.words(file_id)
    # YOUR CODE
    # 1. Print the number of word tokens in the corpus.
    print("1. Print the number of word tokens in the corpus:")
    print(len(word_list))
    # 2. Print the size of the vocabulary (number of unique word tokens).
    print()
    print("2. Print the size of the vocabulary (number of unique word tokens).")
    print(len(set(word_list)))
    print()
    print("3. Print the tokenized words of the first sentence in the corpus.")
    # 3. Print the tokenized words of the first sentence in the corpus.
    first_sents = gb.sents(file_id)[0]
    s = " ".join(first_sents)
    print(nltk.word_tokenize(s))
def q2():
    print('q2: {:}'.format(''))
    import nltk
    from nltk.corpus import brown
    # Your Code
    romance_word_list = brown.words(categories='romance')
    # 1. Print the top 10 most common words in the romance category.
    print("1. Print the top 10 most common words in the romance category:")
    # 2. Print the word frequencies
    from collections import Counter
    words_given = ['ring','activities','love','sports','church']
    counter =  Counter(romance_word_list)
    most_common = counter.most_common(10)
    word_list = []
    for word,freq in most_common:
        word_list.append(word)
    print(word_list)
    print()
    print("2. Print the word frequencies:")
    for word in words_given:
        print('frequency of ' + word + ' in romance: ' + str(counter[word]))
    hobbies_word_list = brown.words(categories='hobbies')
    counter = Counter(hobbies_word_list)
    print()
    for word in words_given:
        print('frequency of ' + word + ' in hobbies: ' + str(counter[word])) 
    
    
    
    
    

def q3():
    print('q3: {:}'.format(''))
    from nltk.corpus import wordnet as wn
    # Your Code
    # 1. Print all synonymous words of the word ‘dictionary’.
    print('1. All synonymous words of the word \'dictionary\':')
    print(wn.synset('dictionary.n.01').lemma_names())
    print()
    # 2. Print all hyponyms of the word ‘dictionary’.
    print("2. Hyponyms of the word \'dictionary\':")
    dictionary = wn.synset('dictionary.n.01')
    hyp = dictionary.hyponyms()
    print(hyp)
    print()
    # 3. Calculate similarities.
    sim = []
    right_whale = wn.synset('right_whale.n.01')
    novel = wn.synset('novel.n.01')
    minke_whale = wn.synset('minke_whale.n.01')
    tortoise = wn.synset('tortoise.n.01')
    
    pair1 = right_whale.path_similarity(novel)
    pair2 = right_whale.path_similarity(minke_whale)
    pair3 = right_whale.path_similarity(tortoise)
    print("3.Calculate similarities:")
    print("Similarity of \'right_whale\' and \'minke_whale\': " + str(pair2))
    print("Similarity of \'right_whale\' and \'tortoise\': " + str(pair3))
    print("Similarity of \'right_whale\' and \'novel\': " + str(pair1))

if __name__ == '__main__':
    
    q1()

    print()
    
    q2()

    print()
    
    q3()


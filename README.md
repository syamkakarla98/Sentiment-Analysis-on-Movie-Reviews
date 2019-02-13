
# <font color="tomato">IMDB Sentiment Analysis</font>

* Sentiment Analysis means finding the mood of the public about things like movies, politicians, stocks, or even current events. We will analyse the sentiment of the movie reviews corpus we saw earlier.

## 1. Let’s import our libraries:


```python
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

* We will be using the **<font color="red">Naive Bayes classifier**<font color="red"> for this example. The Naive Bayes is a fairly simple machine learning algorithm, that works mainly with probabilities. 



* Before we start, there is something that had me stumped for a long time. I saw it in all the examples, but it didn’t make sense. But the Naive Bayes classifier, especially in the **<font color="red">Nltk library**</font>, expects the input to be in this format: Every word must be followed by true. So for example, if you have these words:
    
```python
"Hello World"
```
    you need to pass it in as:

```python
{'Hello': True,  'World': True}
```


```python
# This is how the Naive Bayes classifier expects the input
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
```

Let's see what will return <font color="red">stopwords.words("english")</font>


```python
for i in stopwords.words("english"):
    print(i+',',end=' ')
```

    i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't, 

Let’s see how this works:



```python
create_word_features(["the", "quick", "brown", "quick", "a", "fox"])
```




    {'quick': True, 'brown': True, 'fox': True}



We call our function with the string “the quick brown quick a fox”.

You can see that a) The stop words are removed  b) Repeat words are removed  c) There is a True with each word.

Again, this is just the format the Naive Bayes classifier in nltk expects.

Okay, let’s start with the code. Remember, the sentiment analysis code is just a machine learning algorithm that has been trained to identify positive/negative reviews.


```python
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))
                        
```

We create an empty list called neg_reviews. Next, we loop over all the files in the neg folder.

We get all the words in that file.

Then we use the function we wrote earlier to create word features in the format nltk expects. Here is a sample of the output:


```python
print(neg_reviews[0],len(neg_reviews))    
```

    ({'plot': True, ':': True, 'two': True, 'teen': True, 'couples': True, 'go': True, 'church': True, 'party': True, ',': True, 'drink': True, 'drive': True, '.': True, 'get': True, 'accident': True, 'one': True, 'guys': True, 'dies': True, 'girlfriend': True, 'continues': True, 'see': True, 'life': True, 'nightmares': True, "'": True, 'deal': True, '?': True, 'watch': True, 'movie': True, '"': True, 'sorta': True, 'find': True, 'critique': True, 'mind': True, '-': True, 'fuck': True, 'generation': True, 'touches': True, 'cool': True, 'idea': True, 'presents': True, 'bad': True, 'package': True, 'makes': True, 'review': True, 'even': True, 'harder': True, 'write': True, 'since': True, 'generally': True, 'applaud': True, 'films': True, 'attempt': True, 'break': True, 'mold': True, 'mess': True, 'head': True, '(': True, 'lost': True, 'highway': True, '&': True, 'memento': True, ')': True, 'good': True, 'ways': True, 'making': True, 'types': True, 'folks': True, 'snag': True, 'correctly': True, 'seem': True, 'taken': True, 'pretty': True, 'neat': True, 'concept': True, 'executed': True, 'terribly': True, 'problems': True, 'well': True, 'main': True, 'problem': True, 'simply': True, 'jumbled': True, 'starts': True, 'normal': True, 'downshifts': True, 'fantasy': True, 'world': True, 'audience': True, 'member': True, 'going': True, 'dreams': True, 'characters': True, 'coming': True, 'back': True, 'dead': True, 'others': True, 'look': True, 'like': True, 'strange': True, 'apparitions': True, 'disappearances': True, 'looooot': True, 'chase': True, 'scenes': True, 'tons': True, 'weird': True, 'things': True, 'happen': True, 'explained': True, 'personally': True, 'trying': True, 'unravel': True, 'film': True, 'every': True, 'give': True, 'clue': True, 'kind': True, 'fed': True, 'biggest': True, 'obviously': True, 'got': True, 'big': True, 'secret': True, 'hide': True, 'seems': True, 'want': True, 'completely': True, 'final': True, 'five': True, 'minutes': True, 'make': True, 'entertaining': True, 'thrilling': True, 'engaging': True, 'meantime': True, 'really': True, 'sad': True, 'part': True, 'arrow': True, 'dig': True, 'flicks': True, 'actually': True, 'figured': True, 'half': True, 'way': True, 'point': True, 'strangeness': True, 'start': True, 'little': True, 'bit': True, 'sense': True, 'still': True, 'guess': True, 'bottom': True, 'line': True, 'movies': True, 'always': True, 'sure': True, 'given': True, 'password': True, 'enter': True, 'understanding': True, 'mean': True, 'showing': True, 'melissa': True, 'sagemiller': True, 'running': True, 'away': True, 'visions': True, '20': True, 'throughout': True, 'plain': True, 'lazy': True, '!': True, 'okay': True, 'people': True, 'chasing': True, 'know': True, 'need': True, 'giving': True, 'us': True, 'different': True, 'offering': True, 'insight': True, 'apparently': True, 'studio': True, 'took': True, 'director': True, 'chopped': True, 'shows': True, 'might': True, 'decent': True, 'somewhere': True, 'suits': True, 'decided': True, 'turning': True, 'music': True, 'video': True, 'edge': True, 'would': True, 'actors': True, 'although': True, 'wes': True, 'bentley': True, 'seemed': True, 'playing': True, 'exact': True, 'character': True, 'american': True, 'beauty': True, 'new': True, 'neighborhood': True, 'kudos': True, 'holds': True, 'entire': True, 'feeling': True, 'unraveling': True, 'overall': True, 'stick': True, 'entertain': True, 'confusing': True, 'rarely': True, 'excites': True, 'feels': True, 'redundant': True, 'runtime': True, 'despite': True, 'ending': True, 'explanation': True, 'craziness': True, 'came': True, 'oh': True, 'horror': True, 'slasher': True, 'flick': True, 'packaged': True, 'someone': True, 'assuming': True, 'genre': True, 'hot': True, 'kids': True, 'also': True, 'wrapped': True, 'production': True, 'years': True, 'ago': True, 'sitting': True, 'shelves': True, 'ever': True, 'whatever': True, 'skip': True, 'joblo': True, 'nightmare': True, 'elm': True, 'street': True, '3': True, '7': True, '/': True, '10': True, 'blair': True, 'witch': True, '2': True, 'crow': True, '9': True, 'salvation': True, '4': True, 'stir': True, 'echoes': True, '8': True}, 'negative') 1000
    

So there are a 1000 negative reviews.

Let’s do the same for the positive reviews. The code is exactly the same:




```python
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))
    
print(pos_reviews[0])    
print(len(pos_reviews))
```

    ({'films': True, 'adapted': True, 'comic': True, 'books': True, 'plenty': True, 'success': True, ',': True, 'whether': True, "'": True, 'superheroes': True, '(': True, 'batman': True, 'superman': True, 'spawn': True, ')': True, 'geared': True, 'toward': True, 'kids': True, 'casper': True, 'arthouse': True, 'crowd': True, 'ghost': True, 'world': True, 'never': True, 'really': True, 'book': True, 'like': True, 'hell': True, '.': True, 'starters': True, 'created': True, 'alan': True, 'moore': True, 'eddie': True, 'campbell': True, 'brought': True, 'medium': True, 'whole': True, 'new': True, 'level': True, 'mid': True, '80s': True, '12': True, '-': True, 'part': True, 'series': True, 'called': True, 'watchmen': True, 'say': True, 'thoroughly': True, 'researched': True, 'subject': True, 'jack': True, 'ripper': True, 'would': True, 'saying': True, 'michael': True, 'jackson': True, 'starting': True, 'look': True, 'little': True, 'odd': True, '"': True, 'graphic': True, 'novel': True, '500': True, 'pages': True, 'long': True, 'includes': True, 'nearly': True, '30': True, 'consist': True, 'nothing': True, 'footnotes': True, 'words': True, 'dismiss': True, 'film': True, 'source': True, 'get': True, 'past': True, 'thing': True, 'might': True, 'find': True, 'another': True, 'stumbling': True, 'block': True, 'directors': True, 'albert': True, 'allen': True, 'hughes': True, 'getting': True, 'brothers': True, 'direct': True, 'seems': True, 'almost': True, 'ludicrous': True, 'casting': True, 'carrot': True, 'top': True, 'well': True, 'anything': True, 'riddle': True, ':': True, 'better': True, 'set': True, 'ghetto': True, 'features': True, 'violent': True, 'street': True, 'crime': True, 'mad': True, 'geniuses': True, 'behind': True, 'menace': True, 'ii': True, 'society': True, '?': True, 'question': True, 'course': True, 'whitechapel': True, '1888': True, 'london': True, 'east': True, 'end': True, 'filthy': True, 'sooty': True, 'place': True, 'whores': True, 'unfortunates': True, 'nervous': True, 'mysterious': True, 'psychopath': True, 'carving': True, 'profession': True, 'surgical': True, 'precision': True, 'first': True, 'stiff': True, 'turns': True, 'copper': True, 'peter': True, 'godley': True, 'robbie': True, 'coltrane': True, 'enough': True, 'calls': True, 'inspector': True, 'frederick': True, 'abberline': True, 'johnny': True, 'depp': True, 'blow': True, 'crack': True, 'case': True, 'widower': True, 'prophetic': True, 'dreams': True, 'unsuccessfully': True, 'tries': True, 'quell': True, 'copious': True, 'amounts': True, 'absinthe': True, 'opium': True, 'upon': True, 'arriving': True, 'befriends': True, 'unfortunate': True, 'named': True, 'mary': True, 'kelly': True, 'heather': True, 'graham': True, 'proceeds': True, 'investigate': True, 'horribly': True, 'gruesome': True, 'crimes': True, 'even': True, 'police': True, 'surgeon': True, 'stomach': True, 'think': True, 'anyone': True, 'needs': True, 'briefed': True, 'go': True, 'particulars': True, 'unique': True, 'interesting': True, 'theory': True, 'identity': True, 'killer': True, 'reasons': True, 'chooses': True, 'slay': True, 'bother': True, 'cloaking': True, 'screenwriters': True, 'terry': True, 'hayes': True, 'vertical': True, 'limit': True, 'rafael': True, 'yglesias': True, 'les': True, 'mis': True, 'rables': True, 'good': True, 'job': True, 'keeping': True, 'hidden': True, 'viewers': True, 'funny': True, 'watch': True, 'locals': True, 'blindly': True, 'point': True, 'finger': True, 'blame': True, 'jews': True, 'indians': True, 'englishman': True, 'could': True, 'capable': True, 'committing': True, 'ghastly': True, 'acts': True, 'ending': True, 'whistling': True, 'stonecutters': True, 'song': True, 'simpsons': True, 'days': True, 'holds': True, 'back': True, 'electric': True, 'car': True, '/': True, 'made': True, 'steve': True, 'guttenberg': True, 'star': True, 'worry': True, 'make': True, 'sense': True, 'see': True, 'onto': True, 'appearance': True, 'certainly': True, 'dark': True, 'bleak': True, 'surprising': True, 'much': True, 'looks': True, 'tim': True, 'burton': True, 'planet': True, 'apes': True, 'times': True, 'sleepy': True, 'hollow': True, '2': True, 'print': True, 'saw': True, 'completely': True, 'finished': True, 'color': True, 'music': True, 'finalized': True, 'comments': True, 'marilyn': True, 'manson': True, 'cinematographer': True, 'deming': True, 'word': True, 'ably': True, 'captures': True, 'dreariness': True, 'victorian': True, 'era': True, 'helped': True, 'flashy': True, 'killing': True, 'scenes': True, 'remind': True, 'crazy': True, 'flashbacks': True, 'twin': True, 'peaks': True, 'though': True, 'violence': True, 'pales': True, 'comparison': True, 'black': True, 'white': True, 'oscar': True, 'winner': True, 'martin': True, 'childs': True, 'shakespeare': True, 'love': True, 'production': True, 'design': True, 'original': True, 'prague': True, 'surroundings': True, 'one': True, 'creepy': True, 'acting': True, 'solid': True, 'dreamy': True, 'turning': True, 'typically': True, 'strong': True, 'performance': True, 'deftly': True, 'handling': True, 'british': True, 'accent': True, 'ians': True, 'holm': True, 'joe': True, 'gould': True, 'secret': True, 'richardson': True, '102': True, 'dalmatians': True, 'log': True, 'great': True, 'supporting': True, 'roles': True, 'big': True, 'surprise': True, 'cringed': True, 'time': True, 'opened': True, 'mouth': True, 'imagining': True, 'attempt': True, 'irish': True, 'actually': True, 'half': True, 'bad': True, 'however': True, '00': True, 'r': True, 'gore': True, 'sexuality': True, 'language': True, 'drug': True, 'content': True}, 'positive')
    1000
    

So we have a 1000 negative and 1000 positive reviews, for a total of 2000. We will now create our test and train samples, this time manually:


```python
train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
print(len(train_set),  len(test_set))
```

    1500 500
    

We end up with 1500 training samples and 500 test.

Let’s create our Naive Bayes Classifier, and train it with our training set.


```python
classifier = NaiveBayesClassifier.train(train_set)
```

And let’s use our test set to find the accuracy:


```python
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)
```

    72.39999999999999
    

Ac accuracy of 72.39999999999999%. Could you improve it? How?

For now, I want to show you how to classify a review as negative or positive. But before that, a warning.

The problem with sentiment analysis, as with any machine learning approach, is that your algorithm is only as good as your data. <font color="red">If your data is crap, your algorithm will be crap</font>.

Not only that, the <font color="red">algorithm depends on the type of input you train it with</font>. So if you train your data with long movie reviews, it will not work with Twitter data, which is much shorter.

This particular dataset is, imo, a bit short. Also, the reviews are very informal, using a lot of swear words etc. Which is why I found it not very accurate when comparing it to Imdb reviews, where swearing is discouraged and reviews are (slightly) more formal.

Anyway, I was looking for negative and positive reviews. Our algorithm is more accurate when the review contains stronger words (horrible instead of bad). For the bad reviews, I found this gem of a movie. A real masterpiece:


```python
review_1 = '''It would be impossible to sum up all the stuff that sucks about this film, 
so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume 
(funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear;
an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly 
and caaarreee-fulll-yyy prrooo-noun-ceeed;
a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.
The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. 
Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
'''
print(review_1 )
```

    It would be impossible to sum up all the stuff that sucks about this film, 
    so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume 
    (funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear;
    an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly 
    and caaarreee-fulll-yyy prrooo-noun-ceeed;
    a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.
    The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. 
    Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
    
    

* We need to word_tokenize the text, call our function it, and then use the classify() function to let our algorithm decide if this is a positive or negative review.


```python
words = word_tokenize(review_1)
words = create_word_features(words)
classifier.classify(words)
```




    'negative'



* That was correct, but only because the review was really scathing.

* For the positive review, I chose one of my favourite movies, Spirited Away, a very beautiful movie:


```python
review_spirit = '''Spirited Away' is the first Miyazaki I have seen, but from this stupendous film.
I can tell he is a master storyteller. A hallmark of a good storyteller is making the audience empathise or
pull them into the shoes of the central character. Miyazaki does this brilliantly in 'Spirited Away'. 
During the first fifteen minutes we have no idea what is going on. Neither does the main character Chihiro. 
We discover the world as Chihiro does and it's truly amazing to watch. But Miyazaki doesn't seem to treat this world as
something amazing. The world is filmed just like our workaday world would. The inhabitants of the world go about their daily
business as usual as full with apathy as us normal folks. Places and buildings are not greeted by towering establishing shots
and majestic music. The fact that this place is amazing doesn't seem to concern Miyazaki.
What do however, are the characters. Miyazaki lingers upon the characters as if they were actors. 
He infixes his animated actors with such subtleties that I have never seen, even from animation giants Pixar. 
Twenty minutes into this film and I completely forgot these were animated characters; 
I started to care for them like they were living and breathing. 
Miyazaki treats the modest achievements of Chihiro with unashamed bombast. 
The uplifting scene where she cleanses the River God is accompanied by stirring music and is as exciting as 
watching gladiatorial combatants fight. Of course, by giving the audience developed characters to care about, 
the action and conflicts will always be more exciting, terrifying and uplifting than normal, generic action scenes. 
'''
print(review_spirit)
```

    Spirited Away' is the first Miyazaki I have seen, but from this stupendous film.
    I can tell he is a master storyteller. A hallmark of a good storyteller is making the audience empathise or
    pull them into the shoes of the central character. Miyazaki does this brilliantly in 'Spirited Away'. 
    During the first fifteen minutes we have no idea what is going on. Neither does the main character Chihiro. 
    We discover the world as Chihiro does and it's truly amazing to watch. But Miyazaki doesn't seem to treat this world as
    something amazing. The world is filmed just like our workaday world would. The inhabitants of the world go about their daily
    business as usual as full with apathy as us normal folks. Places and buildings are not greeted by towering establishing shots
    and majestic music. The fact that this place is amazing doesn't seem to concern Miyazaki.
    What do however, are the characters. Miyazaki lingers upon the characters as if they were actors. 
    He infixes his animated actors with such subtleties that I have never seen, even from animation giants Pixar. 
    Twenty minutes into this film and I completely forgot these were animated characters; 
    I started to care for them like they were living and breathing. 
    Miyazaki treats the modest achievements of Chihiro with unashamed bombast. 
    The uplifting scene where she cleanses the River God is accompanied by stirring music and is as exciting as 
    watching gladiatorial combatants fight. Of course, by giving the audience developed characters to care about, 
    the action and conflicts will always be more exciting, terrifying and uplifting than normal, generic action scenes. 
    
    


```python
words = word_tokenize(review_spirit)
words = create_word_features(words)
classifier.classify(words)
```




    'positive'




```python
reviews_list=["It's a nice movie","It's not a nice movie","the Movie is more violent",
             "In the movie, the hreo doesn't performed well","OK, the movie is well","OK nice","it's a disaster"
            ]

for review in reviews_list:
    words = word_tokenize(review)
    words = create_word_features(words)
    print(review,'      ('+classifier.classify(words)+')')

```

    It's a nice movie       (positive)
    It's not a nice movie       (positive)
    the Movie is more violent       (positive)
    In the movie, the hreo doesn't performed well       (positive)
    OK, the movie is well       (positive)
    OK nice       (positive)
    it's a disaster       (negative)
    

* Some of them are correct and few of them aren't, but I’d like to repeat, the  classifier isn’t very accurate overall, I suspect because the original sample is very small and not  very representative for Imdb reviews. But it’s good enough for learning.

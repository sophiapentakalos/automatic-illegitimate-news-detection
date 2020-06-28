'''
Raw Score = 0.1579 * (PDW) + 0.0496 * ASL

PDW = Percentage of Difficult Words (words not on Dale-Chall list)

ASL = Average Sentence Length in words

If (PDW) is greater than 5%, then:

Adjusted Score = Raw Score + 3.6365, otherwise Adjusted Score = Raw Score

Adjusted Score = ReadingD Grade of a reader who can comprehend your text at 4th grade or above.

'''
import nltk
from nltk import tokenize
from nltk.tag import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

wordList = open("DaleChallEasyWordList.txt", "r")
easy_words = set(wordList.read().splitlines())
def calculateScore(fileName):
    fakeTxt = open('/Users/Sophia/Documents/SeniorResearch/fakeNewsDataset/legit/' +fileName, "r")
    sampleSentences = fakeTxt.read()
    sampleSentences = str(sampleSentences).replace("\n", ". ")
    sampleSentences = str(sampleSentences).replace('\"', "")
    sampleSentences = tokenize.sent_tokenize(sampleSentences)
    for sentence in set(sampleSentences):
        if len(sentence) == 1:
            sampleSentences.remove(sentence)
    #print(sampleSentences)

    diffWordCount = 0
    sentLength = 0
    sentCount = 0
    hardWords = []
    for sentence in sampleSentences:
        tagged = pos_tag(sentence.split())
        propernouns = [word for word, pos in tagged if pos == 'NNP']
        sentCount += 1
        wordsInSent = sentence.split(" ")
        sentLength += len(wordsInSent)
        for word in wordsInSent:
            if word.lower() not in easy_words and word.isalpha() and word not in propernouns:
                diffWordCount += 1
                hardWords.append(word)

    #print(hardWords)
    pdw = diffWordCount / sentLength
    #print(pdw)
    asl = sentLength / sentCount
    #print(asl)
    raw = 15.79 * pdw + 0.0496 * asl
    adj = 0
    if pdw > 0.05:
        adj = raw + 3.6365
    else:
        adj = raw
    return adj

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return [score['neg'], score['neu'], score['pos']]

def parseText(fileName):
    testTxt = open('/Users/Sophia/Documents/SeniorResearch/fakeNewsDataset/legit/' +fileName, "r")
    sampleSentences = testTxt.read()
    sampleSentences = str(sampleSentences).replace("\n", ". ")
    sampleSentences = str(sampleSentences).replace('\"', "")
    sampleSentences = tokenize.sent_tokenize(sampleSentences)
    for sentence in set(sampleSentences):
        if len(sentence) == 1:
            sampleSentences.remove(sentence)
    #print(sampleSentences)

    negSum = 0
    neuSum = 0
    posSum = 0
    for sentence in sampleSentences:
        scores = sentiment_analyzer_scores(sentence)
        negSum += scores[0]
        neuSum += scores[1]
        posSum += scores[2]

    return [negSum/len(sampleSentences), neuSum/len(sampleSentences), posSum/len(sampleSentences)]

text_file = open("legitData2ft.txt", "w")
for j in {"biz", "edu", "entmt", "polit", "sports", "tech0"}:
    for i in range(1, 41):
        name = ""
        if i < 10:
            name = j + "0" + str(i) + ".legit.txt"
        else:
            name = j + str(i) + ".legit.txt"
        #text_file.write(str(calculateScore(name)) + " " + str(parseText(name)[0]) + " " + str(parseText(name)[1]) + " " + str(parseText(name)[2]) + "\n")
        text_file.write(str(calculateScore(name)) + " " + str(parseText(name)[0]) + "\n")

text_file.close()

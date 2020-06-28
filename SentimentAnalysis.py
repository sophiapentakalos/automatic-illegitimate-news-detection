import nltk
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

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

    return neuSum

print(parseText("tech001.legit.txt"))

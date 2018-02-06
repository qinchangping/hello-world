__author__ = 'qcp'
# -*- coding:utf-8 -*-
from numpy import *
from random import choice

# wan = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4
# tong = [11, 12, 13, 14, 15, 16, 17, 18, 19] * 4
# tiao = [21, 22, 23, 24, 25, 26, 27, 28, 29] * 4
def generatePile():
    pileRandom = []
    wan = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4
    tong = [11, 12, 13, 14, 15, 16, 17, 18, 19] * 4
    tiao = [21, 22, 23, 24, 25, 26, 27, 28, 29] * 4
    pileInit = wan + tong + tiao
    for i in range(108):
        card = choice(pileInit)
        pileRandom.append(card)
        # print(card)
        # pileInit=delete(pileInit,card)
        del pileInit[pileInit.index(card)]
        # print(len(pileInit))
    # print(pileRandom)
    return pileRandom


def dealCard(pileRest, start=True):
    retCard = []
    if len(pileRest) > 0:
        if start == True:  # 开局发牌13张
            retCard = pileRest[0:13]
            del pileRest[0:13]
        else:
            retCard = pileRest[0]
            del pileRest[0]
        print('after deal, %d cards left' % len(pileRest))
    else:
        print('no card left')
    return retCard


def isHu(cards):
    cards = array(cards)
    print('array cards=', cards)
    wan = cards[cards < 10]
    tong = cards[(cards > 10) & (cards < 20)]
    tiao = cards[cards > 20]
    #print('wan cards=', wan)
    #print('tong cards=', tong)
    #print('tiao cards=', tiao)

    pairIndex = findPair(cards)
    if (len(pairIndex) != 0):
        for i in range(len(pairIndex)):
            mianCards = cards.copy()
            mianCards[pairIndex[i][0]] = 0
            mianCards[pairIndex[i][1]] = 0
            mianCards = mianCards[mianCards > 0]
            if isTingPai(mianCards) == 1:
                print('Hu le: ', mianCards)
                print('with pair: ', cards[pairIndex[i][0]])


def isTingPai(cards,numMing=0):
    if howManySuits(cards) == 4-numMing:
        print('ting pai')
        return 1
    else:
        print('not ting pai')
        return 0


def findPair(cards):
    pairIndex = []
    for i in range(len(cards)):
        #restCards=cards[:i]+cards[i+1:]
        restCards = cards.copy()
        restCards[i] = 0
        for j in range(len(restCards)):
            if (i < j) & (cards[i] == restCards[j]):
                #print('pair found:', cards[i], cards[j])
                pairIndex.append([i, j])
    return pairIndex


def howManySuits(cards):
    numCards = len(cards)
    maxSuits = int(numCards / 3)
    numKe = isKe(cards)
    numShun = isShun(cards)
    if numKe + numShun > maxSuits:
        return maxSuits
    else:
        return numKe + numShun


def isShun(cards):
    '''
    返回顺（三个数字连续）的个数
    '''
    n = len(cards)
    flag = 0
    if n < 3:
        print('cards must >=3')
    else:
        for i in range(n - 2):
            if (cards[i] == cards[i + 1] - 1) & (cards[i] == cards[i + 2] - 2):
                #print('Shun: ', cards[i], cards[i + 1], cards[i + 2])
                flag += 1
    return flag


def isKe(cards):
    '''返回刻子（三个牌相同）的个数
    '''
    n = len(cards)
    flag = 0
    if n < 3:
        print('cards must >=3')
    else:
        for i in range(n - 2):
            if (cards[i] == cards[i + 1]) & (cards[i] == cards[i + 2]):
                #print('Kezi: ', cards[i], cards[i + 1], cards[i + 2])
                flag += 1
    return flag


def isPair(cards):
    '''返回对子的个数
    '''
    n = len(cards)
    flag = 0
    if n < 2:
        print('cards must >=2')
    else:
        for i in range(n - 1):
            if (cards[i] == cards[i + 1]):
                #print('pair: ', cards[i], cards[i + 1])
                flag += 1
    return flag


def game():
    pileRest = generatePile()

    player1Cards = dealCard(pileRest, start=True)
    player1Cards.append(dealCard(pileRest, start=False))
    print('dealt cards=', player1Cards)
    player1Cards.sort()
    print('sorted cards=', player1Cards)
    #isHu(player1Cards)


#game()

a = array([1, 2, 3, 4])
b = array([1, 1, 1, 2, 3, 4])
c = array([1,1, 1, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14])
isHu(c)
#isHu(b)
#print(findPair(a))
#print(findPair(b))
#print(findPair(c))
#print(isShun(a))
#print(isShun(b))
#print(isKe(b))
#print(howManySuits(a))
#print(howManySuits(c))
#isHu(c)
#print(c)

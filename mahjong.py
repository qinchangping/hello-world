__author__ = 'qcp'
# -*- coding:utf-8 -*-
from numpy import *
from random import choice

# wan = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4
# tong = [11, 12, 13, 14, 15, 16, 17, 18, 19] * 4
# tiao = [21, 22, 23, 24, 25, 26, 27, 28, 29] * 4
class Player:
    tableCards = []  # 桌子上被打出的牌
    tableMingCards = []  # 桌子上的明牌，所有人杠和碰的牌

    def __init__(self, handCards, newCard=[], mingCards=[], myTableCards=[]):
        self.handCards = list(handCards)
        self.newCard = newCard  # 新摸上的牌
        self.mingCards = mingCards  # 已经杠和碰的牌，明牌
        self.numPeng = 0
        self.numGang = 0
        self.numMing = self.numGang + self.numPeng  # 明牌套数
        self.myTableCards = myTableCards  # 桌面上自己打出的牌
        self.huList = []  # whatHu(self.handCards)
        # self.value = [1.0] * 27
        self.fan = 1  # 计算翻数

    def play(self, tableCards, tableMingCards):
        '''
        摸牌、计算、打出
        返回：
        '''
        self.handCards += self.newCard
        self.handCards.sort()
        print(self.handCards)
        # dis = self.discard()
        # dis = 11
        # self.handCards.remove(dis)
        # self.myTableCards.append(dis)
        # tableCards.append(dis)
        self.gang(tableCards, tableMingCards)
        self.peng(tableCards, tableMingCards)
        self.calcP(tableCards, tableMingCards)
        #print('tableMingCards:', tableMingCards)
        self.ChooseDiscard(tableCards, tableMingCards)


    def ChooseDiscard(self,tableCards,tableMingCards):
        '''
        决定弃哪张牌
        '''
        print('handCards=',self.handCards)
        return 0

        # @staticmethod

    # def updateTableCards(Play,dis):
    # Play.tableCards.append(dis)

    def gang(self, tableCards, tableMingCards):
        '''
        杠牌
        '''
        for item in set(self.handCards):  # 自摸杠牌
            if self.handCards.count(item) == 4:
                self.mingCards.extend([item] * 4)
                tableMingCards.extend([item] * 4)
                self.numGang += 1
                self.fan = self.fan * 2
                print('zimo gang:', self.mingCards)
                for i in range(4): self.handCards.remove(item)
                print('after gang, handCards=', self.handCards)

        if len(tableCards) > 0:  # 杠别人牌
            lastCard = tableCards[-1]
            numKe, keList = isKe(self.handCards)
            if lastCard in keList:
                self.mingCards.extend([lastCard] * 4)
                tableMingCards.extend([lastCard] * 4)
                self.numGang += 1
                self.fan = self.fan * 2
                print('other gang:', self.mingCards)
                while (lastCard in self.handCards):
                    self.handCards.remove(lastCard)
                print('after gang, handCards=', self.handCards)


    def peng(self, tableCards, tableMingCards):
        '''
        碰牌
        '''
        if len(tableCards) > 0:
            lastCard = tableCards[-1]
            pairIndex = findPair(self.handCards)
            if len(pairIndex) > 0:
                for i in range(len(pairIndex)):
                    if lastCard == pairIndex[i][0]:
                        self.mingCards.extend([lastCard] * 3)
                        tableMingCards.extend([lastCard] * 3)
                        self.numGang += 1
                        self.fan = self.fan * 2
                        print('peng:', self.mingCards)
                        self.handCards.remove(lastCard)
                        self.handCards.remove(lastCard)
                        print('after peng, handCards=', self.handCards)


    def calcP(self, tableCards, tableMingCards):
        '''
        计算胡牌概率，所胡的牌在剩余未知牌中的张数
        '''
        self.huList = whatHu(self.handCards)
        print('huList = ', self.huList)
        huProb = []
        if len(self.huList) > 0:
            for i in range(len(self.huList)):
                count = self.handCards.count(self.huList[i]) + \
                        tableCards.count(self.huList[i]) + \
                        tableMingCards.count(self.huList[i])  # 所胡的牌已经出现了几张
                probility = (4.0 - float(count)) / float(
                    108 - len(self.handCards) - len(tableCards) - len(tableMingCards))
                huProb.append(probility)
            print('huProb = ', huProb)
        else:
            print('no Hu yet')
        return huProb


def generatePile():
    '''返回：
    随机排序生成的初始牌堆，类型list
    '''
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


def dealCard(pileRest, start=False):
    '''
    牌局初start=True，返回牌堆最顶上13张牌，类型list
    之后start=False，返回牌堆最顶上1张牌，类型list
    '''
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


def isTingPai(cards, numMing=0):
    if howManySuits(cards) == 4 - numMing:
        # print('ting pai')
        return 1
    else:
        # print('not ting pai')
        return 0


def findPair(cards):
    '''返回：
    cards中对子的下标，类型list
    '''
    pairIndex = []
    for i in range(len(cards)):
        # restCards=cards[:i]+cards[i+1:]
        restCards = cards.copy()
        restCards[i] = 0
        for j in range(len(restCards)):
            if (i < j) & (cards[i] == restCards[j]):
                # print('pair found:', cards[i], cards[j])
                pairIndex.append([i, j])
    return pairIndex


def howManySuits(cards):
    '''返回：
    顺子+刻子的套数，类型int
    '''
    cards = list(cards)
    restCards = []
    numKe, keList = isKe(cards)
    if numKe != 0:
        for i in range(len(cards)):
            if cards[i] in keList:
                pass
            else:
                restCards.append(cards[i])
        numShun = isShun(restCards)
    else:
        numShun = isShun(cards)

    return numKe + numShun


def isShun(cards):
    '''返回：
    顺子（三个牌连续）的套数，类型int
    '''
    shunList = []
    n = len(cards)
    i = 0
    while (i < n - 2):
        if (cards[i] == cards[i + 1] - 1) & (cards[i] == cards[i + 2] - 2):
            # print('Shun: ', cards[i], cards[i + 1], cards[i + 2])
            shunList.append(cards[i] * 10000 + cards[i + 1] * 100 + cards[i + 2])
            i += 3
        else:
            i += 1
    shunSet = set(shunList)
    # print(shunList)
    # print(shunSet)
    return len(shunSet)


def isKe(cards):
    '''返回：
    刻子（三个牌相同）的个数，类型int
    刻子牌，类型list
    '''
    n = len(cards)
    cards = list(cards)
    countKe = 0
    keList = []
    if n < 3:
        print('cards must >=3')
    else:
        # for i in range(n - 2):
        i = 0
        while (i < n - 2):
            # if (cards[i] == cards[i + 1]) & (cards[i] == cards[i + 2]):
            if cards.count(cards[i]) == 3:
                keList.append(cards[i])
                countKe += 1
                i += 3
            else:
                i += 1
                # print(countKe)
                # print(keList)
    return countKe, keList


def isPair(cards):
    '''返回对子的个数，类型int
    '''
    n = len(cards)
    flag = 0
    pairList = []
    if n < 2:
        print('cards must >=2')
    else:
        for i in range(n - 1):
            if (cards[i] == cards[i + 1]):
                # print('pair: ', cards[i], cards[i + 1])
                pairList.append(cards[i])
                flag += 1
    pairSet = set(pairList)
    return len(pairSet)


def isHu(cards):
    '''返回：
    cards胡多少张牌，类型int
'''
    cards = array(cards)
    # print('cards=', cards)
    wan = cards[cards < 10]
    tong = cards[(cards > 10) & (cards < 20)]
    tiao = cards[cards > 20]
    # print('wan cards=', wan)
    # print('tong cards=', tong)
    # print('tiao cards=', tiao)
    #cards.sort()
    #print('cards=', cards)
    countHu = 0
    pairIndex = findPair(cards)
    if (len(pairIndex) != 0):
        for i in range(len(pairIndex)):
            mianCards = cards.copy()
            mianCards[pairIndex[i][0]] = 0
            mianCards[pairIndex[i][1]] = 0
            mianCards = mianCards[mianCards > 0]  # 除去对子之后判断面牌是否四套
            if isTingPai(mianCards) == 1:
                print('Hu le: ', mianCards)
                print('with pair: ', cards[pairIndex[i][0]], cards[pairIndex[i][1]])
                countHu += 1
    return countHu


def whatHu(cards):
    '''返回：
    胡哪些牌，类型list
    '''
    wan = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tong = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    tiao = [21, 22, 23, 24, 25, 26, 27, 28, 29]
    fullList = wan + tong + tiao
    cards = list(cards)
    cardSet = set(cards)
    huList = []
    for item in cardSet:
        if cards.count(item) == 4:
            gangIndex = fullList.index(item)
            del fullList[gangIndex]
    for i in range(len(fullList)):
        newCards = cards.copy()
        newCards.append(fullList[i])
        newCards.sort()
        # print(newCards)
        if isHu(newCards) > 0:
            huList.append(fullList[i])
    return huList


def game():
    pileRest = generatePile()
    player1Cards = dealCard(pileRest, start=True)
    player1newCard = dealCard(pileRest, start=False)
    p1 = Player(player1Cards, player1newCard)
    # tableCards = []  # 桌子上被打出的牌
    # tableMingCards = []  # 桌子上的明牌，所有人杠和碰的牌
    print('game initialized.')
    # initialize

    p1.play(p1.tableCards, p1.tableMingCards)
    # isHu(player1Cards)


def test():
    # a = array([1, 1, 1, 3, 4])
    #b = array([1, 1, 1, 1, 2, 3, 4])
    #c = array([1, 1, 1, 2, 3, 4, 5, 6, 7, 11, 11, 12, 13, 14])
    #d = array([1, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14])
    # print(whatHu(d))
    a = array([1, 1, 1, 3, 4, 6, 9, 11, 12, 13, 15, 16, 17])
    b = array([1, 1, 1, 1, 2, 3, 4,5,6])
    c = array([1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15])
    d = array([1, 1, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14])
    e = array([1,1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14,15])
    # print(whatHu(d))

    newCard = [5]
    p = Player(a, newCard)
    #p.tableCards.append(2)
    print('tableCards=', p.tableCards)
    print('tableMingCards=', p.tableMingCards)
    p.play(p.tableCards, p.tableMingCards)
    #print(isShun(a))
    #print(whatHu(e))

# newCard = [2]
# p.tableCards.append(12)
# print('tableCards=',p.tableCards,p.tableMingCards)
# p.play(p.tableCards,p.tableMingCards)
# p.tableCards.append(p.discard())
#
#print(isKe(b))
# game()
#    newCard = [1]
#    p = Player(d, newCard)
#    p.play(p.tableCards, p.tableMingCards)
# isHu(c)
# print(isKe(b))
# isHu(b)
# print(findPair(a))
# print(findPair(b))
# print(findPair(c))
# print(isShun(a))
# print(isShun(b))
# print(isKe(b))
# print(howManySuits(a))
# print(howManySuits(c))
# isHu(c)
# print(c)


if __name__ == '__main__':
    test()




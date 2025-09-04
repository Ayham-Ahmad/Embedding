import math
import pandas as pd
import numpy as np

class Embed:
    __letter_dict = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz", 1)}

    __special_chars = {
        '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
        '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',
        ';', ':', "'", '"', ',', '<', '.', '>', '/', '?', '`', '~'
    }

    def __init__(
            self, 
            query: str, 
            contextList: list[str]
            ):
        self.__query = query
        self.__contextList = contextList
        self.__embeddedQuery = list()
        self.__embeddedContextList = list()
        self.__similarityList = list()

    def getEmbedForChar(self, char) -> int: # Inefficient
        if char.isdigit():
            return int(char)
        elif char.isalpha():
            if char in self.__letter_dict:
                return self.__letter_dict.get(char)
            else: return 0
        else: return 0

    def getEmbedForAQuery(self, query: str) -> tuple[list[int], dict[str, int]]:
        embeddedWord = 0
        embeddedQuery = list()
        debugDict = dict()
        for word in query.split(' '):
            # print(word)
            for char in word:
                if char in self.__special_chars:
                    continue
                else:
                    # print(self.getEmbedForChar(char))
                    embeddedWord += self.getEmbedForChar(char.lower())

                embeddedQuery.append(embeddedWord)
                debugDict[word] = embeddedWord

                #reset vars
                embeddedWord = 0

        return embeddedQuery, debugDict
    
    def getVector(self):
        debugDict = dict()
        debugDictList = list()
        self.__embeddedQuery, debugDict = self.getEmbedForAQuery(query=self.__query)
        debugDictList.append(debugDict)

        for context in self.__contextList:
            embeddedQuery, debugDict = self.getEmbedForAQuery(query=context)
            self.__embeddedContextList.append(embeddedQuery)
            debugDictList.append(debugDict)
        
        return self.__embeddedQuery, self.__embeddedContextList, debugDictList[0], debugDictList[1:]

    def getSimilarity(self, context):
            total = 0
            for num in context:
                total += math.cos(num) * math.sin(num) * math.tan(num)
            self.__similarityList.append(total/len(context))
            
    def getBestContext(self) -> int:
        self.getSimilarity(self.__embeddedQuery)
        for context in self.__embeddedContextList:
            self.getSimilarity(context)

        return self.__similarityList[1:].index(min(self.__similarityList[1:], key=lambda x:abs(x-self.__similarityList[0])))

    @staticmethod
    def accuracy(answers, results) -> float:
        trueValues = 0

        if len(answers) <= 0 or len(results) <=  0:
            return 'Error: Answers or Results lists is empty'
        
        for i in range(len(answers)):
            # print(answers[i] , results[i])
            if answers[i] == results[i]:
                trueValues += 1
        return trueValues/len(answers)
 
    def embed(
                self,
                printEmbeddedQuery: bool = False,
                printEmbeddedContext: bool = False,
                printDebugContext: bool = False, printDebugQuery: bool = False,
                printDebugContexAndQuerey: bool = False,
                printSimilarty: bool = False,
                printBestContextIndex: bool = False, 
                printBestContext: bool = False,
                debug: bool = False
            ): 
        vector = self.getVector()
        result = self.getBestContext()

        if printEmbeddedQuery or debug:
            print(f'\nEmbedded Query:\n{vector[0]}\n')
        if printEmbeddedContext or debug:
            print(f'\nEmbedded Context:\n{vector[1]}\n')
        if printDebugQuery or printDebugContexAndQuerey or debug:
            print(f'\nDebug Query:\n{vector[2]}\n')
        if printDebugContext or printDebugContexAndQuerey or debug:
            print(f'\nDebug Context:\n{vector[3]}\n')
        if printSimilarty or debug:
            print(f'\nSimilarity: {self.__similarityList}\n')
        if printBestContextIndex or debug:
            print(f'\nBest Context Index: {result}\n')
        if printBestContext or debug:
            print(f'\nBest Context: {self.__contextList[result]}\n')

        return vector, result    

if __name__ == '__main__':

    examples = pd.read_json('examples.json')

    answers = [0,0,0,0,0,0,0,0,0,0]
    results = []

    for query, context in examples.items():
        # print(f'\nQuery: {query}\n')

        embed = Embed(query, context)
        vector, result = embed.embed()
        results.append(result)

    print(f'\n\nAccuracy: {embed.accuracy(answers, results)}')


# Convert the list to numpy array
# make the accuracy as a class function
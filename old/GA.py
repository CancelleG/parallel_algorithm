import numpy as np
import time
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


class City:         #创建一个城市类型，
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):       #计算两个城市间的距离
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:      #创建适应度函数，将路径距离作为倒数；
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):     #构建从一个城市到route的下个城市，并且最后一个城市将回到第一个城市
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)   #计算i城市到i+1城市的距离的累计距离
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):         #适应度函数为距离的倒数
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):      #将城市列表随机打乱
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):       #初始化种群，按照种群大小生成具有随机连接各个城市的路线作为一个population
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):     #进行适应度排名
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)     #返回一个适应度从大到小排列的list


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()     #对fitness得分按排列进行累加
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()        #按0-100归一化

    for i in range(0, eliteSize):       #选择输入的前eliteSize名最优fitness
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):      #往selectionResults中加入len(popRanked) - eliteSize名随机抽取的人
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):           #将由selection选择出来的population编号进行提取population
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):        #又叫基因交叉crossover
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    #采用顺序交叉
    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):         #交叉产生下一代
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):       #变异，采用交换基因策略；每个基因都需要进行一定概率变异
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    progress = []                               #这三句用来画图
    progress.append(1 / rankRoutes(pop)[0][1])
    # for i in range(0, generations):
    #     pop = nextGeneration(pop, eliteSize, mutationRate)
    #
    #     if (1 / rankRoutes(pop)[0][1]) < min(progress):
    #         print("Gen:%d,   distance:%s" % (i, str(1 / rankRoutes(pop)[0][1])))
    #
    #     progress.append(1 / rankRoutes(pop)[0][1])


    i = 0
    while(1):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        if (1 / rankRoutes(pop)[0][1]) < min(progress):
            print("Gen:%d,   distance:%s" % (i, str(1 / rankRoutes(pop)[0][1])))
        progress.append(1 / rankRoutes(pop)[0][1])
        if int(1 / rankRoutes(pop)[0][1]) == 7544:      #通过brute_forces_tsp运行得出结果，11：4038
            break
        i += 1
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]


    return progress, bestRoute


def read_tsp():
    with open("./tspfiles/berlin52.tsp") as f:
        line_count = f.readlines()
        store_line = []
        for count in range(len(line_count)-6):
            store_line.append([float(string) for string in line_count[count+6].split()])
            count += 1
        print("read finish")
    return store_line


def main():
    cityList = []

    data = read_tsp()
    data_len  = len(data)
    for i in range(data_len):
        cityList.append(City(x=data[i][1], y=data[i][2]))

    progress,bestRoute = geneticAlgorithm(population=cityList, popSize=data_len, eliteSize=5, mutationRate=0.01, generations=500)  ###看是否缩进
    print("This took", time.clock() - start_time, "seconds to calculate.")
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == '__main__':
    start_time = time.clock()
    main()

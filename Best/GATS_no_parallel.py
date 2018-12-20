import numpy as np
import time
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
exitFlag = 0

elite_all = []
progress = []
bestRoute_forplot = []

def read_tsp():
    with open(r"D:\Users\HZ.Guo\PycharmProjects\parallel_algorithm\tspfiles\berlin52.tsp") as f:
        line_count = f.readlines()
        store_line = []
        for count in range(len(line_count)-6):
            store_line.append([float(string) for string in line_count[count+6].split()])
            count += 1
        print("read finish")
    return store_line


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

        if i ==0:               #选择全局精英
            elite_all.append(population[index])
            if len(elite_all)==5 :
                elite_all.pop(0)

    matingpool[-len(elite_all):] = elite_all      #将全局精英加入到各个线程的后面len(elite_all)个弱者中
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


def mutate(individual, mutationRate, Tabu_table):       #变异，采用交换基因策略；每个基因都需要进行一定概率变异
                                            #输入为个体的基因、变异率，计算每个基因的变异率，并且随机与其他基因交换

    for swapped in range(len(individual)):
        if (random.random() < 1*mutationRate):
            individual = TabuSearch(individual, swapped, Tabu_table)
        if(random.random() < 0.25*mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def TabuSearch(individual, swapped, Tabu_table):       #swapped为要交换的位置
    candidate_num = 5              #设置候选集的大小
    tabu_length = 4
    generate_candidate = list(range(len(individual)))       #用于生成随机的交换基因，采用数组的形式用于定位

    individual_score = 1/Fitness(individual).routeFitness()
    if individual_score <= Tabu_table["best"]:          #用于初始化”best“的值
        Tabu_table["best"] = individual_score

    generate_candidate.pop(swapped)                     #将swapped的基因的index排除
    candidate_swap = random.sample(generate_candidate, candidate_num)

    score_list = []     #存放得分
    i = 0           #用于定位candidate_swap的值
    candidate_tackel_store = []     #存放生成的candidate
    for gene in candidate_swap:         #这里的gene为individual的index
        candidate = [gene for gene in individual]
        city1 = candidate[swapped]
        city2 = candidate[gene]
        candidate[swapped] = city2
        candidate[gene] = city1
        candidate_score = 1 / Fitness(candidate).routeFitness()
        if [city1, city2] in Tabu_table["table"] and candidate_score>=Tabu_table["best"]:       #如果(city1-city2)对在禁忌表中，且不是最优，则选择下一个
            candidate_swap.pop(i)
            continue
        elif [city2, city1] in Tabu_table["table"] and candidate_score>=Tabu_table["best"]:
            candidate_swap.pop(i)
            continue

        #此时candidate_score<Tabu_table["best"]
        a = 0
        for gene in Tabu_table["table"]:        #如果(city1-city2)对在禁忌表中，且是最优，则将这一对从禁忌表中删除
            if gene == [city1, city2] or gene == [city2, city1]:
                Tabu_table["table"].pop(a)
                return candidate
            a += 1

        i += 1
        score_list.append(candidate_score)
        candidate_tackel_store.append(candidate)

    best_index = 0
    for gene in range(len(score_list)):
        if score_list[gene] <= score_list[0]:
            best_index = gene
    # if score_list[best_index] <= Tabu_table["best"]:            #判断score_list[best_index]的值是否是最优，若不是，则加入
    #                                                             #禁忌表，并且将该个体返回


    if len(score_list) == 0:
        return individual
    if score_list[best_index] < Tabu_table["best"]:
        Tabu_table["best"] = score_list[best_index]


    if len(Tabu_table["table"]) == tabu_length:
        Tabu_table["table"].pop(0)
        Tabu_table["table"].append([individual[swapped], individual[candidate_swap[best_index]]])
    else:
        Tabu_table["table"].append([individual[swapped], individual[candidate_swap[best_index]]])
    return candidate_tackel_store[best_index]

def mutatePopulation(population, mutationRate, Tabu_table):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate, Tabu_table)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate, Tabu_table):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate,Tabu_table)
    return nextGeneration


def geneticAlgorithm(threadName, population, popSize, eliteSize, mutationRate, generations):
    global exitFlag, progress, bestRoute_forplot
    Tabu_table = {}
    Tabu_table["table"] = []
    Tabu_table["best"] = float('inf')
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    progress_sub = []                               #这两句用来画图
    progress_sub.append(1 / rankRoutes(pop)[0][1])

    gener_count = 0
    i = 0
    while(1):
        try:
            if exitFlag == 1:
                raise ValueError("invalid thread id")
        except(ValueError):
            break
        pop = nextGeneration(pop, eliteSize, mutationRate, Tabu_table)
        if (1 / rankRoutes(pop)[0][1]) < min(progress_sub):
            print("Name:%s, Gen:%d,   distance:%s" % (threadName, i , str(1 / rankRoutes(pop)[0][1])))
        progress_sub.append(1 / rankRoutes(pop)[0][1])
        if int(1 / rankRoutes(pop)[0][1]) == 7754 or gener_count == generations:      #通过brute_forces_tsp运行得出结果，11：4038;  52:7544
                                                        #数据#输入
            bestRouteIndex = rankRoutes(pop)[0][0]
            bestRoute = pop[bestRouteIndex]
            bestRoute_forplot = bestRoute
            progress = progress_sub
            print("Final distance: " + str(min(progress)))
            break
        i += 1
        gener_count += 1
    exitFlag = 1

def main():
    # global bestRoute_forplot
    cityList = []
    data = read_tsp()
    data_len  = len(data)
    for i in range(data_len):
        cityList.append(City(x=data[i][1], y=data[i][2]))


    geneticAlgorithm("main", population=cityList, popSize=data_len, eliteSize=5,
                                           mutationRate=0.01, generations=5000)  ###看是否缩进
    # 创建新线程

    # print("退出主线程")
    print("This took", time.clock() - start_time, "seconds to calculate.")
    #画出最优路径的路线图
    bestRoute_forplot.append(bestRoute_forplot[0])
    x_coords = []
    y_coords = []
    for individual in bestRoute_forplot:
        x_coords.append(individual.x)
        y_coords.append(individual.y)
    plt.plot(x_coords[0], y_coords[0], 'ro-')
    plt.plot(x_coords, y_coords, 'rx-')
    for elem in range(0, len(bestRoute_forplot)):
        elem_num = elem + 1
        point_x = bestRoute_forplot[elem].x
        point_y = bestRoute_forplot[elem].y
        if elem<len(bestRoute_forplot)-1:
            plt.annotate("%d" % elem_num, xy=(point_x, point_y))
        else:
            plt.annotate(" ", xy=(point_x, point_y))
    plt.title("GATS")
    plt.show()
    #画出距离随迭代次数的变化
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title("GATS")
    plt.show()


if __name__ == '__main__':
    start_time = time.clock()
    main()

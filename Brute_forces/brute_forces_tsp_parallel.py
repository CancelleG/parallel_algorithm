"""
TSP Brute Force Calculator
CECS 545
Sarah Mullins
Fall 2014
"""

import matplotlib.pyplot as plt         #visualization
from Brute_forces import calcs
import time                             #clock
from itertools import permutations      #find route permutations
import multiprocessing
from scipy.misc import factorial


def parallel_cal(permu_list,point_objects,list_size):
    min_local_dist = [0, 999999999999]
    for permu in permu_list:
        # initialize distance
        distance = 0
        # find the first point & set it as the current point
        first_point = calcs.findPoint(point_objects, permu, 0)
        curr_point = first_point

        # for each point in the permutation...
        for val in range(1, list_size):
            # calculate the distance to the next point & add this to the total distance
            next_point = calcs.findPoint(point_objects, permu, val)
            distance += calcs.calcDistance(curr_point, next_point)
            curr_point = next_point

        # add the distance to return to the starting point
        distance += calcs.calcDistance(curr_point, first_point)
        # add total distance to the list
        if distance < min_local_dist[1]:
            min_local_dist = [permu, distance]
    return min_local_dist

if __name__ == '__main__':

    #input data
    file_name = "./tspfiles/berlin52_raw.tsp"       #添加TSP路径

    #begin clock
    start_time = time.clock()

    #extract point information from file
    all_points = calcs.getPoints(file_name)

    #assign point information to a list of objects
    point_objects = calcs.assignPoints(all_points)
    min_dist = [0, 999999999999]
    #point list attributes
    list_size = len(point_objects)
    list_elems = list(range(1, len(point_objects) + 1))

    #initalize list of distances for each route
    distance_list = list()
    record_index = 0    #为parallel_0、1、2、3计数
    max_record = 50000     #parallel_0、1、2、3存放的最大容量

    parallel_number = 4

    parallel_0 = []
    parallel_1 = []
    parallel_2 = []
    parallel_3 = []
    best_record = []
    pool = multiprocessing.Pool(processes=4)
    # lock = multiprocessing.Lock()
    #for each permutation of points...
    count = 0       #计数总循环次数
    max_num = factorial(list_size)
    for elem in permutations(list_elems):          #逐渐递减，elem=(1,2,....,20)，type=tuple下一次为(1,2,3,......,19)
        count += 1
        if record_index < max_record and count != max_num:
            parallel_0.append(elem)
            record_index += 1
        elif record_index < 2*max_record and count != max_num:
            parallel_1.append(elem)
            record_index += 1

        elif record_index < 3*max_record and count != max_num:
            parallel_2.append(elem)
            record_index += 1

        elif record_index < 4*max_record and count != max_num:
            parallel_3.append(elem)
            record_index += 1
        else:
            # parallel_0.append(elem)         #防止执行到这一步时，当前的elem丢失
            record0 =pool.apply_async(parallel_cal, (parallel_0,point_objects,list_size))
            record1 =pool.apply_async(parallel_cal, (parallel_1,point_objects,list_size))
            record2 =pool.apply_async(parallel_cal, (parallel_2,point_objects,list_size))
            record3 =pool.apply_async(parallel_cal, (parallel_3,point_objects,list_size))
            best_record.append(record0.get())
            best_record.append(record1.get())
            best_record.append(record2.get())
            best_record.append(record3.get())
            for best in best_record:
                if best[1] < min_dist[1]:
                    min_dist = best
                    print(min_dist)
            best_record = []
            parallel_0 = []
            parallel_1 = []
            parallel_2 = []
            parallel_3 = []
            print("Progress：%.2f%s;  current: %d, total: %d,  remaining time:%.2f h" % (
                count / factorial(list_size) * 100,
                '%', count, factorial(list_size),
                ((time.clock() - start_time + 0.01) / count * factorial(list_size)-((time.clock() - start_time))) / 60 / 60))
            record_index = 0
    pool.close()
    pool.join()

    #create a list of x,y coordinates of each point in the minimum path, including the start/finish point
    x_coords = list()
    y_coords = list()
    for item in min_dist[0]:
        x_coords.append(calcs.findPoint2(point_objects, item).x)
        y_coords.append(calcs.findPoint2(point_objects, item).y)

    x_coords.append(calcs.findPoint2(point_objects, min_dist[0][0]).x)
    y_coords.append(calcs.findPoint2(point_objects, min_dist[0][0]).y)


    #plot the points in the minimum path
    plt.plot(x_coords[0], y_coords[0], 'ro-')
    plt.plot(x_coords, y_coords, 'rx-')

    for elem in range(0, list_size):
        elem_num = elem + 1
        point_x = point_objects[elem].x
        point_y = point_objects[elem].y
        plt.annotate("%d" % elem_num, xy=(point_x, point_y))

    print(min_dist[0], "is the route with the lowest cost,", min_dist[1], ".")
    print("This took", time.clock() - start_time, "seconds to calculate.")

    plt.show()
import matplotlib.pyplot as plt         #visualization
from Brute_forces import calcs
import time                             #clock
from itertools import permutations      #find route permutations
from scipy.misc import factorial


#input data
file_name = "./tspfiles/berlin52_raw.tsp"       #添加TSP路径

#begin clock
start_time = time.clock()

#extract point information from file
all_points = calcs.getPoints(file_name)

#assign point information to a list of objects
point_objects = calcs.assignPoints(all_points)

#point list attributes
list_size = len(point_objects)
list_elems = list(range(1, len(point_objects) + 1))

#initalize list of distances for each route
distance_list = list()
min_dist = [0, 999999999999]


record_acomplish = 0
#for each permutation of points...
for elem in permutations(list_elems):          #逐渐递减，elem=(1,2,....,20)，type=tuple下一次为(1,2,3,......,19)
    #initialize distance
    distance = 0
    #find the first point & set it as the current point
    first_point = calcs.findPoint(point_objects, elem, 0)
    curr_point = first_point

    #for each point in the permutation...
    for val in range(1, list_size):
        #calculate the distance to the next point & add this to the total distance
        next_point = calcs.findPoint(point_objects, elem, val)
        distance += calcs.calcDistance(curr_point, next_point)
        curr_point = next_point

    #add the distance to return to the starting point
    distance += calcs.calcDistance(curr_point, first_point)
    #add total distance to the list
    if distance < min_dist[1]:
        min_dist = [elem, distance]         #记录最短的排列和距离
    record_acomplish += 1
    if record_acomplish%100000 == 0:
        print("Accomplish：%.2f%s;  current: %d, total: %d, excess time:%.2f h" % (
        record_acomplish / factorial(list_size) * 100,
        '%', record_acomplish, factorial(list_size),
        (time.clock() - start_time) / record_acomplish * factorial(list_size) / 60 / 60))

#create a list of x,y coordinates of each point in the minimum path, including the start/finish point
x_coords = list()
y_coords = list()
print('asdasdasd', min_dist)
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
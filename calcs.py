'''
Class & function definitions for TSP Brute Force program
'''

import csv      #extract data from a csv file
import math     #calculate distance between two points

#each point object contains its number id, x coord and y coord
class Point:
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


#import point information from file into a list
def getPoints(csv_file):
    prelim_list = list()
    with open(csv_file, "r") as f:
        file_reader = csv.reader(f, delimiter=' ')
        for row in file_reader:
            if row[0] == "1":
                prelim_list.append(row)
                for second_row in file_reader:
                    prelim_list.append(second_row)
    return prelim_list


#create Point objects from list
def assignPoints(orig_list):
    point_list = list()
    for item in orig_list:
        point = Point(int(item[0]), float(item[1]), float(item[2]))
        point_list.append(point)
    return point_list


#calculates the distance between two points
def calcDistance(point1, point2):
    return math.sqrt((point2.x - point1.x)**2+(point2.y - point1.y)**2)


#find a point given its assigned number and permutation
def findPoint(point_objects, elem, val):
    for obj in point_objects:
        if obj.num == elem[val]:
            return obj

#find a point given its assigned number
def findPoint2(point_objects, val):
    for obj in point_objects:
        if obj.num == val:
            return obj


#

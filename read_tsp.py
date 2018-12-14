import numpy as np

def read_tsp():
    with open("./tspfiles/berlin52.tsp") as f:
        line_count = f.readlines()
        count = 6
        store_line = []
        while(1):
            try:
                store_line.append([float(string) for string in line_count[count].split()])
                count += 1
                print([float(string) for string in line_count[count].split()])
            except(ValueError):
                break
        print("read finish")

    print(store_line)




if __name__ == '__main__':
    read_tsp()
import multiprocessing
import time
def func(msg):
  for i in range(3):
    print(msg)
    time.sleep(1)
def running():
    pool = multiprocessing.Pool(processes=4)
    for i in range(10):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
        pool.apply_async(func, ('cool',))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")

if __name__ == "__main__":
    running()


from multiprocessing import Pool
import os
import time

def main():
    work1 = [["A", 20], ["B", 10], ["C", 1], ["D", 3]]
    work2 = [["X", 10], ["Y", 10]]
    work3 = [["X", 10], ["Y", 10]]
    work4 = [["X", 10], ["Y", 10]]
    work5 = [{"a":"bikas", "b":2.0}, {"c":"bikas", "d":2.0}]

    work = (work1, work2, work3, work4, work5)
    pool_handler(work)
    
def work_log(work_data):
    if type(work_data[0]) == dict:
        print(os.getpid())
        print (work_data)
    else:
        print(os.getpid())
        print(" Process %s waiting %s seconds" % (work_data[0][1], work_data[1][1]))
        time.sleep(int(work_data[1][1]))
        print(" Process %s Finished." % work_data[0])


def pool_handler(work):
    p = Pool(6)
    p.map(work_log, work)
    p.close()

if __name__ == '__main__':
    main()
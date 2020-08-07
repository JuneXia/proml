import time
import ray


def func1():
    time.sleep(1)
    return 1


@ray.remote
def func2():
    print('I func2')
    time.sleep(10)
    print('I func2, end!')
    return 1


if __name__ == '__main__':
    t1 = time.time()
    rslt = [func1() for i in range(4)]
    print(rslt, 'execute time: ', time.time() - t1)


if __name__ == '__main__':
    ray.init()

    t1 = time.time()
    a = [func2.remote() for i in range(4)]
    rslt = ray.get(a[-1])  # ray.get 会阻塞等待子进程返回
    print(rslt, 'execute time: ', time.time() - t1)
    rslt = ray.get(a[0])
    print(rslt, 'execute time: ', time.time() - t1)


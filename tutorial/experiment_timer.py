import time
import datetime
from datetime import date
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import os


def func():
    now = datetime.datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    print('do func  time :',ts)

def func2():
    #耗时2S
    now = datetime.datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    print('do func2 time：',ts)
    time.sleep(2)

def dojob():
    #创建调度器：BlockingScheduler
    scheduler = BlockingScheduler()
    #添加任务,时间间隔2S
    scheduler.add_job(func, 'interval', seconds=2, id='test_job1')
    #添加任务,时间间隔5S
    scheduler.add_job(func2, 'interval', seconds=3, id='test_job2')
    scheduler.start()


def job_func(text):
    print(text)
    # cmd = 'python /home/tangni/dev/proml/tutorial/experiment_timer.py > /home/tangni/tmp_timer.txt'
    os.system("echo 'hello world' > /home/tangni/tmp.txt")


if __name__ == '__main__1':
    dojob()


if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    # 在 2017-12-13 时刻运行一次 job_func 方法
    # scheduler.add_job(job_func, 'date', run_date=date(2017, 12, 13), args=['text'])
    # 在 2017-12-13 14:00:00 时刻运行一次 job_func 方法
    # scheduler.add_job(job_func, 'date', run_date=datetime(2017, 12, 13, 14, 0, 0), args=['text'])
    # 在 2017-12-13 14:00:01 时刻运行一次 job_func 方法
    scheduler.add_job(job_func, 'date', run_date='2020-09-28 09:37:00', args=['text'])

    scheduler.start()

    while True:
        print('debug')
        time.sleep(1)
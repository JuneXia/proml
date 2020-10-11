if __name__ == '__main__1':
    f = open('foo.txt')
    line = f.readline()
    while line:
        # print(line),
        print(line, end='')
        line = f.readline()
    f.close()


if __name__ == '__main__2':
    import sys
    sys.stdout.write("Enter your name:")
    name = sys.stdin.readline()
    print("Hello ", name)


if __name__ == '__main__3':
    # ⚠️ input 只能在命令行运行时奏效
    name = input("Enter your name:")
    print("Hello ", name)


if __name__ == '__main__4':
    text = '''
    content-type: 
    hello world
    '''

    print(text)


if __name__ == '__main__5':
    x = 3.4
    print(str(x))
    print(repr(x))


if __name__ == '__main__6':
    prices = {
        'MSFT': 1,
        'AMOSON': 2
    }

    print('MSFT' in prices)
    print('abcd' in prices)
    print(1 in prices)

    print(prices.get('MSFT', 1.1))
    print(prices.get('MSFT'))


if __name__ == '__main__7':
    f = open('foo.txt')

    for line in f:
        print(line)



# 生成器
if __name__ == '__main__8':
    def countdown(n):
        print('counting down!')
        while n > 0:
            print('before yield')
            yield n
            print('after yield')
            n -= 1

    c = countdown(5)
    # print(c.__next__())
    # print(c.__next__())
    for p in c:
        print(p)
        print('*********************')


# 生成器
if __name__ == '__main__9':
    import time

    def tail(f):
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

    def grep(lines, search_text):
        for line in lines:
            if search_text in line:
                yield line

    fpath = 'C:\\Users\\Administrator\\dev\\proml\\tutorial\\pystu\\foo.txt'
    log = tail(open(fpath))
    pylines = grep(log, 'bc')
    for line in pylines:
        print(line)


# ***************************************************


def print_matches(match_text):
    print("Looking for", match_text)

    while True:
        print('before yield')
        line = (yield)
        print('after yield, line: ', line)
        if match_text in line:
            print(line)
    print('will return')


# 协程
if __name__ == '__main__10':
    matcher = print_matches("python")
    print('*************************')
    print(matcher.__next__())
    print('*************************')
    # print(matcher.__next__())  # 会崩溃
    print('*************************')
    matcher.send('Hello world')
    print('*************************')
    matcher.send('python is cool')
    print('*************************')
    matcher.send('yow')
    print('*************************')
    matcher.close()
    print('*************************')


# 协程
if __name__ == '__main__':
    matchers = [
        print_matches('python1'),
        print_matches('guido3'),
        print_matches('jython7'),
    ]

    for m in matchers:
        m.__next__()

    for i in range(10):
        for m in matchers:
            m.send(str(i))
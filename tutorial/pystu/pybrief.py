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


if __name__ == '__main__':
    x = 3.4
    print(str(x))
    print(repr(x))

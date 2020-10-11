import sys


# 3.3 引用计数与垃圾收集
if __name__ == '__main__1':
    a = 37
    print(sys.getrefcount(37))
    print(sys.getrefcount(a))
    del a
    print(sys.getrefcount(37))
    print(sys.getrefcount(a))


# 3.4 引用与复制
if __name__ == '__main__2':
    a = [1, 2, 3, 4]
    b = a
    print(b is a)
    print(b == a)

    a = [1, 2, [3, 4]]
    b = list(a)
    print(b is a)
    print(b == a)
    b[2][0] = 10
    print(b is a)
    print(b == a)


# 3.6.2 数字类型
if __name__ == '__main__3':
    z = 3

    # 为了兼容有理数，整数有如下属性
    print(z.numerator, z.denominator)  # z.numerator 是分子, z.denominator 是分母

    # 为了兼容复数，整数和浮点数都有如下属性：
    print(z.real, z.imag)  # z.real 是实部, z.imag 是虚部


# 3.6.3 序列类型 > 所有序列的通用操作
if __name__ == '__main__4':
    s = '1314acad'

    print(max(s), min(s), sorted(s))
    print(len(s), all(s), any(s))

    s = [2, 1, 3, True, False]
    print(max(s), min(s))
    print(len(s), all(s), any(s))

    del s[1:3]  # 切片删除
    print(s)

    # 字符和数字不可比较（也就不可排序），下面会运行报错
    s = [2, 1, 3, True, False, 'b', 'a']
    print(max(s), min(s))
    print(len(s), all(s), any(s))


if __name__ == '__main__':
    pass
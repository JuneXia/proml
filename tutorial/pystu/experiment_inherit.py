# 使用 super(..., self).__init__() 初始化方法，而不是使用Base.__init()

if False:
    class Base(object):
        def __init__(self):
            print("Base init")


    class Medium1(Base):
        def __init__(self):
            Base.__init__(self)
            print("Medium1 init")


    class Medium2(Base):
        def __init__(self):
            Base.__init__(self)
            print("Medium2 init")


    class Leaf(Medium1, Medium2):
        def __init__(self):
            Medium1.__init__(self)
            Medium2.__init__(self)
            print("Leaf init")

    leaf = Leaf()
    """
    output:  注意：这里的Base会被初始化两次，这是错误的
    Base init
    Medium1 init
    Base init
    Medium2 init
    Leaf init
    """

else:
    class Base(object):
        def __init__(self):
            print("Base init")


    class Medium1(Base):
        def __init__(self):
            super(Medium1, self).__init__()
            print("Medium1 init")


    class Medium2(Base):
        def __init__(self):
            super(Medium2, self).__init__()
            print("Medium2 init")


    class Leaf2(Medium1, Medium2):
        def __init__(self):
            super(Leaf2, self).__init__()
            print("Leaf2 init")

    leaf2 = Leaf2()
    """
    output: 这时候父类只会被初始化一次，正确
    Base init
    Medium2 init
    Medium1 init
    Leaf2 init
    """




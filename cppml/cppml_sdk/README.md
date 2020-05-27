本项目主要是用于封装算法实现为sdk，包括算法静态库与动态库的编译以及相关测试，下面会具体介绍。

# 目录结构
```angular2html
$ cd cppml_sdk
.
├── CMakeLists.txt
├── gcai                       # 我们自己的算法实现代码，详见该目录下的 README.md
├── inc                        # 准备对外发布的头文件
│   ├── amcomdef.h             # 数据类型的定义
│   ├── asvloffscreen.h        # 关于图像颜色空间的定义
│   ├── gc_face_sdk.h          # 算法API
│   └── merror.h               # 错误码文件
├── linux_a                    # 算法编译好的静态库或者动态库都将发布到这里
├── README.md                  # 该目录下的README文件
└── test.cpp                   # sdk 测试代码
```

# 安装依赖
opencv == 4.2.0


# 测试步骤
在安装完上述依赖之后依次执行下面步骤即可。

**step1**:
先 cd 到 gcai 目录下，编译静态库，具体方法请参见 gcai/README.md

**step2**:
```bash
$ cd /path/to/cppml_sdk
```

**step3**:
```bash
$ mkdir build && cd build && cmake ..

$ make
# 此时会在当前目录下生成cppml_demo可执行文件
```

**step4**:
```bash
./cppml_demo ../ml/models/mobilenetv1_conv1x1_in96_netclip_eph112.pt ../test_00000008.jpg
```

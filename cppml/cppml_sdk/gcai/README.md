本目录下为算法具体实现代码，下面会具体介绍。

# 目录结构
```angular2html
.
├── CMakeLists.txt         # 用于将我们自己的算法代码编译成静态库或者动态库(这里推荐编译为静态库)
├── example.cpp            # 算法测试代码，已无用可删除
├── gc_face_sdk.cpp        # SDK 接口代码实现，其头文件位于 ../inc/gc_face_sdk.h(因为要对外发布，所以放在了外面)
├── inc                    # 算法代码头文件目录
│   └── maskface.h         # 算法实现头文件
├── maskface.cpp           # 算法代码实现
└── README.md
```

# 安装依赖
opencv == 4.2.0


# 测试步骤
在安装完上述依赖之后依次执行下面步骤即可。

**step1**:
```bash
$ cd /path/to/cppml_sdk/gcai
```

**step2**:
```bash
$ mkdir build && cd build && cmake ..

$ make
# 此时会在 /path/to/cppml_sdk/linux_a 目录下生成静态库后者动态库文件。
```

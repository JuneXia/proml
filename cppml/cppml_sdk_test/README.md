# 目录结构
```angular2html
.
├── CMakeLists.txt                                       # cmake编译所需要的文件
├── inc                                                  # 头文件目录
│   ├── amcomdef.h
│   ├── asvloffscreen.h
│   ├── gc_face_sdk.h
│   └── merror.h
├── linux_a                                              # 静态库目录
│   ├── libgcai_maskface.a
├── ml
│   ├── libtorch                                         # torch发行版目录，直接从pytorch官网下载，并将完整目录拷贝到此处即可。（具体内容请到该目录查看）
│   └── models                                           # 模型目录
│       └── mobilenetv1_conv1x1_in96_netclip_eph112.pt
├── README.md
├── test_00000008.jpg                                    # 测试图片
└── test.cpp                                             # 测试程序
```

# 安装依赖
opencv == 3.2.0


# 测试步骤
在安装完上述依赖之后依次执行下面步骤即可。

**step1**:
安装所需要的依赖(如OpenCV)，如果已安装请忽略.

**step2**:
```bash
$ cd /path/to/cppml_sdk_test

$ mkdir build && cd build && cmake ..

$ make
# 此时会在当前目录下生成cppml_demo可执行文件
```

**step3**:
```bash
./cppml_demo ../ml/models/mobilenetv1_conv1x1_in96_netclip_eph112.pt ../test_00000008.jpg
```

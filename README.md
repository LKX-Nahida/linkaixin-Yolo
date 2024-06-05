本项目用于存放我的YOLOv7改进项目，主要是针对轻量化的YOLOv7-tiny进行改进。
项目针对backbone、neck层、上采样层、新型且有效的卷积块、检测头、激活函数、损失函数、锚框聚类算法、NMS、注意力机制等模块都会有所介绍和加入，并且会长期不定期进行更新。


拉取项目结束后，需要先安装对应的torch环境。
CPU训练或测试采用CPU环境;GPU则采用GPU环境进行操作，具体步骤可浏览我的博客：https://blog.csdn.net/2401_84870184/article/details/138527393?spm=1001.2014.3001.5501。
哪些点可以快速改进YOLO？具体可浏览我的博客：https://blog.csdn.net/2401_84870184/article/details/138688349
项目目前加入了如下改进：

一、针对backbone部分
1、替换骨干网络为MobileNetv3，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138706294?spm=1001.2014.3001.5501
2、替换骨干网络为VanillaNet，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138731835?spm=1001.2014.3001.5501
3、替换骨干网络为MobileOne，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138754980?spm=1001.2014.3001.5501
4、替换骨干网络为PP-LCNet，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138764091?spm=1001.2014.3001.5501
5、替换骨干网络为ShuffleNetv2，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138824684?spm=1001.2014.3001.5501
6、未完待续......

二、针对一些模块、卷积的引入
1、引入YOLOv5的C3模块，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138864574?spm=1001.2014.3001.5501
2、引入YOLOv8的C2f模块，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138872738?spm=1001.2014.3001.5501
3、引入AKConv，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139300859?spm=1001.2014.3001.5501
4、引入ODConv，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139338004?spm=1001.2014.3001.5501
5、未完待续......

三、替换上采样层
1、替换上采样层为CARAFE，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138902742?spm=1001.2014.3001.5501
2、替换上采样层为转置卷积，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139010981?spm=1001.2014.3001.5501
3、替换上采样层为双三次插值方式，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138956928?spm=1001.2014.3001.5501
4、替换上采样层为双线性插值方式，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/138952302?spm=1001.2014.3001.5501
5、未完待续......

四、替换激活函数
1、替换激活函数为SiLU、ReLU、LeakyReLU、FReLU、PReLU、Hardswish、Mish、ELU等，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139184177?spm=1001.2014.3001.5501
2、未完待续......

五、针对head部分
1、替换特征融合网络为AFPN，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139127571?spm=1001.2014.3001.5501
2、引入Slimneck-GSConv，详情可见博客：https://blog.csdn.net/2401_84870184/article/details/139427295?spm=1001.2014.3001.5501
3、未完待续......




如不想拉取项目而是尝试在你的项目中进行修改，可以按照我的CSDN博客中的步骤进行操作。
我的CSDN博客网址：https://blog.csdn.net/2401_84870184?spm=1000.2115.3001.5343
欢迎关注，共同进步！

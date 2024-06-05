This project is used to store my YOLOv7 improvement project, mainly aimed at improving the lightweight YOLOv7 tiny. 
The project will introduce and incorporate modules such as
backbone, neck layer, upsampling layer, novel and effective convolutional blocks, detection heads, activation functions, loss functions, anchor box clustering algorithm, NMS, attention mechanism, etc.,
and will be updated periodically over a long period of time.


After the pull project is completed, the corresponding torch environment needs to be installed first.
CPU training or testing using a CPU environment; 
The GPU operates in the GPU environment, and the specific steps can be found on my blog: https://blog.csdn.net/2401_84870184/article/details/138527393?spm=1001.2014.3001.5501 .
What points can be quickly improved for YOLO? You can browse my blog specifically: https://blog.csdn.net/2401_84870184/article/details/138688349
The project has currently added the following improvements:

1、 For the backbone 
  1. replace the backbone network with MobileNetv3. For details, please refer to the blog: https://blog.csdn.net/2401_84870184/article/details/138706294?spm=1001.2014.3001.5501 
  2. Replace the backbone network with VanillaNet, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/138731835?spm=1001.2014.3001.5501 
  3. Replace the backbone network with MobileOne, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/138754980?spm=1001.2014.3001.5501
  4. Replace the backbone network with PP-LCNet, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/138764091?spm=1001.2014.3001.5501
  5. Replace the backbone network with ShuffleNetv2, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/138824684?spm=1001.2014.3001.5501
  6. Unfinished to be continued

2、 For the introduction of some modules and convolutions,
  1. Introduce YOLOv5's C3 module. For details, please refer to the blog: https://blog.csdn.net/2401_84870184/article/details/138864574?spm=1001.2014.3001.5501
  2. Introducing the C2f module of YOLOv8, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/138872738?spm=1001.2014.3001.5501
  3. Introducing AKConv, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/139300859?spm=1001.2014.3001.5501
  4. Introducing ODConv, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/139338004?spm=1001.2014.3001.5501
  5. Unfinished to be continued

3、 Replace Upsampling Layer 
  1. Replace Upsampling Layer with CARAFE. For details, please refer to the blog: https://blog.csdn.net/2401_84870184/article/details/138902742?spm=1001.2014.3001.5501 
  2. Replace the upsampling layer with transposed convolution, details can be found in the blog: https://blog.csdn.net/2401_84870184/article/details/139010981?spm=1001.2014.3001.5501 
  3. Replace the upsampling layer with bicubic interpolation method, details can be found in the blog: https://blog.csdn.net/2401_84870184/article/details/138956928?spm=1001.2014.3001.5501
  4. Replace the upsampling layer with bilinear interpolation method, details can be found in the blog: https://blog.csdn.net/2401_84870184/article/details/138952302?spm=1001.2014.3001.5501
  5. Unfinished to be continued

4、 Replace activation function 
1. Replace activation functions with SiLU, ReLU, LeakyReLU, FReLU, PReLU, Hardswish, Mish, ELU, etc. Please refer to the blog for details: https://blog.csdn.net/2401_84870184/article/details/139184177?spm=1001.2014.3001.5501
2. Unfinished to be continued

5、 For section of the head
  1. replace the feature fusion network with AFPN. For details, please refer to the blog: https://blog.csdn.net/2401_84870184/article/details/139127571?spm=1001.2014.3001.5501 
  2. Introducing Slimneck GSConv, details can be found on the blog: https://blog.csdn.net/2401_84870184/article/details/139427295?spm=1001.2014.3001.5501
  3. Unfinished to be continued
     
If you don't want to pull the project and try to make modifications in your project, you can follow the steps in my CSDN blog.
My CSDN blog website: https://blog.csdn.net/2401_84870184?spm=1000.2115.3001.5343
Welcome to follow and make progress together

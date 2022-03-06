---
theme: default
background: /begin.jpg
highlighter: shiki
lineNumbers: false
layout: cover
drawings:
  persist: false
defaults:
  layout: page
title: 基于深度学习的<br />毫米波图像处理研究
---

# 基于深度学习的<br />毫米波图像处理研究

<div>

导师：林澍

汇报人：熊滔

汇报日期：2022-03-08

</div>

---

## 研究内容

毫米波成像技术是通过接收成像物体与背景之间的辐射差异来进行成像的技术，被广泛应用于医学观察、机场安检、军事侦察等领域。但是毫米波所成图像分辨率较低，图像噪声大，难以区分出人体背景以及物体，检测人员无法有效识别图像，从而为大规模商用带来困难。

毫米波成像存在的问题：

- 噪声高，物体成像不明显
- 分辨率低

拟采用深度学习技术处理毫米波图像，包括

- 去噪处理
- 超分辨率重建

---

## 研究进度

目前，已经完成了三个工作：

| 工作             | 完成情况                                         |
| ---------------- | ------------------------------------------------ |
| 数据采集与处理   | <ic-baseline-check-box class="text-green-500" /> |
| 图像去噪处理     | <ic-baseline-check-box class="text-green-500" /> |
| 图像超分辨率重建 | <ic-baseline-check-box class="text-green-500" /> |

<br/>

1. 制作了一个用以训练的毫米波图像训练集，包含成对的原始未处理图像与使用图像处理软件人工处理后的去噪图像。
2. 对图像去噪算法进行了研究，分别使用了卷积神经网络和生成对抗网络进行了去噪处理，并研究残差结构对模型收敛及性能的影响，改善了生成对抗网络，使模型稳定收敛。
3. 对图像超分辨率算法进行了研究，并结合去噪算法，提出了两种思路对图像进行超分辨率重建并去噪。

---

## 训练环境

本文的所有的模型均在租借的云服务器上进行，云服务器的环境参数如下

| 参数         | 值           |
| ------------ | ------------ |
| CPU          | 24 核/80G    |
| GPU          | RTX 3090/24G |
| 操作系统     | Linux        |
| Pytorch 版本 | 1.9.0        |
| Python 版本  | 3.8          |
| CUDA 版本    | 11.2         |

---

## 数据集

数据采集与处理：

- 因为深度学习对于数据量的要求比较高，为了达到比较好的训练效果，可能需要上千幅图像，总共采集了 2000 余幅毫米波成像图像。
- 我们分别将物体放在身体的不同位置，包括胸部、腰部、上身口袋、下身口袋、小腿和大腿等部分，并且对安检可能会遇到的物体进行了成像，包括手机、仿真枪、刀具和金属等物品。
- 收集到的 2000 余幅图像中，筛选出相似的以及质量不好的图片，人工处理了 991 幅图。所做的人工处理包括：
  1. 背景噪声的处理，将背景噪声处理为黑色背景，将重点聚焦在人体和物体上。
  2. 对人体和物体进行增强，使得二者之间的对比更加的明显，方便后续的检测工作。

数据集的划分：

- 每隔 4 张取一张图片作为测试集
- 随机取 10 张作为验证集
- 剩余的作为训练集

---

## 数据集

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;gap: 10px;">
<img src="/origin/015.png" style="width: 120px;" />
<img src="/origin/031.png" style="width: 120px;" />
<img src="/origin/070.png" style="width: 120px;" />
<img src="/origin/282.png" style="width: 120px;" />
<img src="/origin/476.png" style="width: 120px;" />
<img src="/train/015.png" style="width: 120px;" />
<img src="/train/031.png" style="width: 120px;" />
<img src="/train/070.png" style="width: 120px;" />
<img src="/train/282.png" style="width: 120px;" />
<img src="/train/476.png" style="width: 120px;" />
</div>

---

## 卷积神经网络

网络结构：

<img src="/denoise-res-cnn.svg" style="zoom: 50%;">

<!--
本文提出的去噪卷积神经网络如图3所示，这是一个端对端的网络结构，无需为输入图像做任何的预处理，也不需要手工提取图像的任何相关特征，直接输入原始未处理的图像，输出一个已处理的图像，这也是深度学习的特点，无需对图像领域的知识有所了解，也不需要有特征工程相关的知识，即可得到远超传统图像处理方法的一流性能。

图像的大小为1×678×384，因为我们使用的是灰度图像，所以只使用一个通道即可。图像进入网络后，首先经过一个7×7的卷积层，该卷积层输出64通道的特征，并且不改变特征的大小，所以输出的特征大小为64×678×384，接着经过四个残差块，其具体结构稍后介绍，经过四个残差块后通道增加了四倍，而特征的大小也变为了原来的四分之一，此时输出的特征大小为256×170×96，接着依次经过三个2×2的反卷积层，反卷积的作用与卷积层相反，它可以上采样特征，将特征变大，并且相应的减少通道数，经过三个反卷积层之后，输出的特征大小为1×678×384，在最后我们应用了Tanh激活函数，它可以将输出限制在[-1, 1]。
-->

---

## 残差块

残差块由两个 3×3 的卷积层组成，每个卷积层之后都应用了 ReLU 激活函数，除此之外，我们在输入与输出之间引入了一条“高速公路”，可以加快模型的收敛速率，考虑到卷积层可能改变输入特征的通道大小，为了能使得输出特征与输入特征进行相加，我们需要对输入特征投影，1×1 的卷积层只改变特征的通道数，而不改变特征的大小，所以 1×1 的卷积层可以使得输入特征的通道与输出特征的通道数相同，从而实现相加。

<img src="/残差块.svg" style="zoom: 60%;max-width: 100%; margin: 0 auto;" />

---

## 有无残差块的比较

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
  <img src="/res-psnr.svg" />
  <img src="/res-losses.svg" />
</div>

---

## 结果

<Tabs type="card">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-765.png" />
      <img src="/results/res-765.png" />
      <img src="/results/target-765.png" />
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-991.png" />
      <img src="/results/res-991.png" />
      <img src="/results/target-991.png" />
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-717.png" />
      <img src="/results/res-717.png" />
      <img src="/results/target-717.png" />
    </div>
  </template>
</Tabs>

---

## 生成对抗网络

生成对抗网络由生成器和判别器两个模块组成，生成器接收一个低维的随机噪声 $z$ 作为输入，$z$ 一般服从高斯分布或者均匀分布，生成器根据输入的随机噪声生成随机图像，以期望骗过判别器。判别器的作用就是判断输入的图像是真还是假，判别器要尽可能的区分真实图像与生成图像，而生成器要骗过判别器，让其无法区分自己生成的图像是真还是假。

此时生成的输出由随机噪声决定，无法控制生成器的生成。

<img src="/origin-gan.svg" style="zoom: 40%; max-width: 100%; margin: 0 auto;">

<!--
该网络结构由生成器和判别器两个模块组成，生成器接收一个低维的随机噪声z作为输入，z一般服从高斯分布或者均匀分布，生成器根据输入的随机噪声生成随机图像，以期望骗过判别器。判别器的作用就是判断输入的图像是真还是假，判别器要尽可能的区分真实图像与生成图像，而生成器要骗过判别器，让其无法区分自己生成的图像是真还是假。二者之间互相博弈，当二者达到平衡时，生成器生成的图像便可以假乱真，也就是说生成器生成的图像与真实的图像已经相差不大了，便达到了训练的目的，最终判别器对无论是生成器生成的图像还是真实图像，它给出的概率始终为1/2，因为此时已经无法区分真假了。
-->

---

## 条件生成对抗网络

在原始版本的生成对抗网络的基础上，为生成器添加一个条件输入，来控制生成器的输出，这种网络结构为条件生成对抗网络。

本篇论文采用的研究方案就是条件生成对抗网络，研究方案如图所示，因为不需要生成器生成的图像具有随机性，于是去掉了随机噪声 $z$ 作为输入，条件 $y$ 是原始未处理的图像，生成器希望输出的是去噪图像，判别器的作用同上，尽可能区分生成器生成的图像以及来自于真实的图像，判别器的输出为输入图像为真实图像的概率。

<div style="display: grid; grid-template-columns: 3fr 5fr; gap: 10px;">
<img src="/condition-gan.svg" />
<img src="/gan-research.svg" />
</div>

---

## 训练流程

<div style="display: grid; grid-template-columns: 3fr 5fr; gap: 20px;">

<img src="/program-struc.svg" style="zoom: 29%;" />

<div>

训练生成对抗网络过程分为两步，首先固定生成器 G，将处理好的高分辨率图像与当前生成器生成的图像输入判别器 D，我们标记高分辨率图像的预期输出为 1，生成图像的预期输出为 0，用此数据集训练判别器 D；训练判别器完成之后，固定判别器 D，将低分辨率图像输入生成器，生成器根据低分辨率图像生成一副新的图像，这个新的图像将会输入到判别器 D，判别器给出结果，根据结果将误差反向传播到生成器 G 来调整生成器 G 的网络参数。依次往复，直至达到预期的精度，训练结束。

<img src="/train-D.svg" />
<img src="/train-G.svg" />

</div>
</div>

<!-- 程序开始时，首先判断是否处于训练模式，如果处于训练模式，则读取训练数据，数据包含两部分，未处理图像与已处理的去噪图像。训练生成对抗网络过程分为两步，首先固定生成器G，将处理好的高分辨率图像与当前生成器生成的图像输入判别器D，如图15所示，我们标记高分辨率图像的预期输出为1，生成图像的预期输出为0，用此数据集训练判别器D；训练判别器完成之后，固定判别器D，将低分辨率图像输入生成器，如图16所示，生成器根据低分辨率图像生成一副新的图像，这个新的图像将会输入到判别器D，判别器给出结果，根据结果将误差反向传播到生成器G来调整生成器G的网络参数。依次往复，直至达到预期的精度，训练结束。 -->

---

## 网络结构

<Tabs type="card" :show="false" :labels="['生成器', '判别器']">
  <template v-slot:g1>
    <div>
      <img src="/generator.svg" />
    </div>
  </template>
  <template v-slot:g2>
    <div style="width: 80%;">
      <img src="/discriminator.svg" />
    </div>
  </template>
</Tabs>

---

## InsatnceNorm2d 层的作用

在随着网络的加深或者训练的进行，卷积层的输出的整体分布会发生偏移和变化，使得模型收敛变慢，甚至无法收敛，而 InstanceNorm2d 就是通过规范化的手段，使得卷积层的输出始终满足均值为 0 方差为 1 的高斯分布，对于较敏感的值会产生一个较大的梯度，模型的收敛速度加快。

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
  <img src="/with-in.svg" />
  <img src="/without-in.svg" />
</div>

---

## 收敛曲线

判别器与生成器轮流训练，一般训练 m 次判别器，然后训练 n 次生成器，m 和 n 是两个超参数，它们的选择将影响到模型的训练。因为要小心的均衡生成器与判别器的训练，一旦判别器训练的很好，那么就会反向传播的梯度为 0，使得生成器无法有效的学习，也就无法达到训练的效果。

<Tabs type="card" :labels="['m = 1, n = 1', 'm = 1, n = 5', 'm = 5, n = 5']">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr;gap: 50px;">
      <img src="/m1n1-1.svg" />
      <img src="/m1n1-2.svg" />
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr;gap: 50px;">
      <img src="/m1n5-1.svg" />
      <img src="/m1n5-2.svg" />
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr;gap: 50px;">
      <img src="/m5n5-1.svg" />
      <img src="/m5n5-2.svg" />
    </div>
  </template>
</Tabs>

---

## 训练结果

生成的图像根本无法满足要求。

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 90px;">
  <img src="/results/g5d1-1.png" />
  <img src="/results/g5d1-2.png" />
  <img src="/results/g5d1-3.png" />
</div>

---

## 理论分析

标准 GAN 的损失函数

$$
\begin{align}
\min\limits_{G}\max\limits_{D}V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log(D(x))] + \mathbb{E}_{x \sim p_{\text{G}}}[\log(1 - D(G(x)))] \notag
\end{align}
$$

当 $G$ 固定的时候，当

$$
\begin{align}
D = \frac{p_{\text{data}}}{p_{\text{data}} + p_{\text{G}}} \notag
\end{align}
$$

$V(G, D)$ 可以取到最大值，将上式代入损失函数，可以得到

$$
\begin{align}
\min\limits_G \max\limits_D V(G,D) &= \min\limits_G\int_x p_{\text{data}}(x) \log \frac{p_{\text{data}}}{p_{\text{data}} + p_{\text{G}}} + p_{\text{G}}(x) \log \frac{p_{\text{G}}}{p_{\text{data}} + p_{\text{G}}} dx \notag \\
&= -2\log 2 + \min\limits_G 2{\rm{JSD}}({p_{\text{data}}}||{p_{\text{G}}}) \notag \\
\end{align}
$$

当判别器最优时，GAN 的损失函数就相当于 JS 散度，JS 散度存在一个问题，当两个分布没有重叠时，JS 散度为零，这在训练的前期是极有可能发生的，也就是说 GAN 的损失函数的值为一个常数 $-2\log2$，而常数的梯度为 0，这样生成器就没法进行学习，也就生成不出想要的图像。

---

## 解决办法

我们需要提出一种新的损失函数，即使是在两个分布没有交叠时，也能得到一个有意义的值，表示二个分布之间的差异，本文引入了 Wasserstain 距离，定义如下

$$
W(\mathbb{P}_r,\mathbb{P}_g) = \mathop {\inf }\limits_{\gamma  \in \Pi ({\mathbb{P}_r},{\mathbb{P}_g})} {\mathbb{E}_{(x,y)}}\left[ {\left\| {x - y} \right\|} \right]
$$

由于上式不可计算，在满足 1-Lipschitz 连续的条件下，可以转换为如下形式

$$
W(\mathbb{P}_r,\mathbb{P}_g) = \sup\limits_{ \small \left\| f \right\|_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r}\left[ f(x) \right] - \mathbb{E}_{x \sim \mathbb{P}_g}\left[ f(x) \right]
$$

其中 f 必须满足 1-Lipschitz 连续。

---

## 损失函数

为了使用 Wasserstain GAN(WGAN)，我们需要做出三点修改：

1. 去掉判别器最后的 sigmoid 层
2. 损失函数不取 log
3. 保证判别器满足 1-Lipschitz 连续，一般通过权重裁剪或梯度惩罚满足条件，本文使用梯度惩罚

判别器的总损失函数：

$$
L_D = -D(x) + D(G(x^{'})) + 10 * L_{gp}
$$

生成器的总损失函数：

$$
L_G = -D(G(x^{'}))
$$

---

## 结果

<Tabs type="card">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 90px;">
      <img src="/results/origin-765.png" />
      <img src="/results/gan-765.png" />
      <img src="/results/target-765.png" />
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 90px;">
      <img src="/results/origin-991.png" />
      <img src="/results/gan-991.png" />
      <img src="/results/target-991.png" />
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 90px;">
      <img src="/results/origin-717.png" />
      <img src="/results/gan-717.png" />
      <img src="/results/target-717.png" />
    </div>
  </template>
</Tabs>

---

## 内容损失

从结果可以看到 WGAN 生成的图片相比于目标图像太亮了，从直方图可以观察到像素值分布在 0 和 1，为了使得图片具有更加丰富的细节，在训练生成器时，为生成的损失函数添加一个内容损失

$$
L_G = L_{\text{GAN}} + 10^3 * L1
$$

<Tabs type="card" :style="{width: '75%'}">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-765.png" />
      <img src="/results/wgan-765.png" />
      <img src="/results/target-765.png" />
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-991.png" />
      <img src="/results/wgan-991.png" />
      <img src="/results/target-991.png" />
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr;gap: 50px;">
      <img src="/results/origin-717.png" />
      <img src="/results/wgan-717.png" />
      <img src="/results/target-717.png" />
    </div>
  </template>
</Tabs>

---

## 指标对比

生成对抗网络与卷积神经网络的峰值信噪比与结构相似性指数对比如表所示，可以看到加上内容损失的生成对抗网络在峰值信噪比(PSNR)以及结构相似性系数(SSIM)两个方面都是最好的。

|                    | PSNR(dB)  | SSIM     |
| ------------------ | --------- | -------- |
| 卷积神经网络       | 22.26     | 0.64     |
| WGAN（无内容损失） | 19.99     | 0.86     |
| WGAN（有内容损失） | **25.67** | **0.95** |

---

## 超分辨率重建

超分辨率重建是指将给定的低分辨率图像重建为相应的高分辨率图像，目的是为了补偿由于毫米波成像图像采集系统或环境带来的限制。

我们的目的是希望直接输入一张未处理的图像，直接输出一张高分辨率且去噪的图像，本文提出了两种思路：

1. 端对端的网络，训练一个网络，输入是低分辨率带噪图像，输出是高分辨率去噪图像。
2. 训练两个网络，一个用来进行去噪处理，另一个用来进行超分辨率重建。图像首先经过去噪网络，输出去噪图像，然后将去噪图像输入超分辨率重建网络，得到去噪高分辨率图像。

---

## 端对端网络

<Tabs type="card" :labels="['研究方案', '使用方法', '网络结构']">
  <template v-slot:g1>
    <div>
      <img src="/e2e-research.svg" style="zoom: 50%;">
    </div>
  </template>
  <template v-slot:g2>
    <div>
      <img src="/e2e-usage.svg" style="zoom: 50%;">
    </div>
  </template>
  <template v-slot:g3>
    <div>
      <img src="/e2e-net.svg" style="zoom: 50%;">
    </div>
  </template>
</Tabs>

---

## 结果

<Tabs type="card">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-765.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-765.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/e2e-765.png" style="" />
      </div>
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-717.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-717.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/e2e-717.png" style="" />
      </div>
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-991.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-991.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/e2e-991.png" style="" />
      </div>
    </div>
  </template>
</Tabs>

---

## 两阶段网络

两阶段网络指图像的超分辨率重建与去噪分别通过两个网络完成，分别处理去噪和超分辨率重建任务，那么就带来了新的问题，网络的前后顺序是什么，是先进行超分辨率重建，然后进行去噪，还是先进行去噪，再进行超分辨率重建？由于前文所述的去噪网络固定了图像的输入只能为 678×384，为了复用前文所述的去噪网络，本篇文章采取的是先进行去噪，然后再进行超分辨率重建的策略。

<img src="/two-stage-research.svg" />

因为超分辨率重建网络是对已去噪的图像进行超分辨率，所以我们需要训练一个对去噪图像进行超分辨率重建的网络，设计的网络结构与端对端网络结构相同。

---

## 结果

<Tabs type="card">
  <template v-slot:g1>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-765.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-765.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/res2x-765.png" style="" />
      </div>
    </div>
  </template>
  <template v-slot:g2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-717.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-717.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/res2x-717.png" style="" />
      </div>
    </div>
  </template>
  <template v-slot:g3>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 40px;width: 70%;">
      <div style="position: relative;">
        <img src="/results/origin-991.png" style="position: absolute; bottom: 0;" />
      </div>
      <div style="position: relative;">
        <img src="/results/target-991.png" style="position: absolute; bottom: 0;" />
      </div>
      <div>
        <img src="/results/res2x-991.png" style="" />
      </div>
    </div>
  </template>
</Tabs>

---
class: 'text-center'
layout: cover
background: /end.jpg
---

# 谢谢

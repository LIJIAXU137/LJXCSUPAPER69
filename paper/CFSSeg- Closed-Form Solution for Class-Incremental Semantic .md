<description>这幅图片展示了一个圆形的标志，中间有一个红色的书签，背景是白色的。书签是红色的，看起来像是一个书页的边缘，暗示着一个阅读或学习的主题。标志的背景是白色的，与红色的书签形成鲜明的对比。标志上没有其他可辨认的特征或文字。标志的设计简单，侧重于红色书签和白色背景之间的对比。</description>
<text>Check for updates</text>
ACM DL
DIGITAL
LIBRARY
acm
Association for
Computing Machinery
acm open
Latest updates: https://dl.acm.org/doi/10.1145/3746027.3755023
RESEARCH-ARTICLE
CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds
JIAXU LI, Central South University, Changsha, Hunan, China
RUI LI, Central South University, Changsha, Hunan, China
JIANYU QI, Central South University, Changsha, Hunan, China
SONGNING LAI, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, Guangdong, China
LINPU LV, Zhengzhou University, Zhengzhou, Henan, China
KEJIA FAN, Central South University, Changsha, Hunan, China
View all
Open Access Support provided by:
Central South University
The Hong Kong University of Science and Technology (Guangzhou)
South China University of Technology
Peking University
Zhengzhou University
Shanghai Artificial Intelligence Laboratory
PDF Download
3746027.3755023.pdf
19 January 2026
Total Citations: 0
Total Downloads: 75
Published: 27 October 2025
Citation in BibTeX format
MM '25: The 33rd ACM International
Conference on Multimedia
October 27 - 31, 2025
Dublin, Ireland
Conference Sponsors:
SIGMM
MM '25: Proceedings of the 33rd ACM International Conference on Multimedia (October 2025)
https://doi.org/10.1145/3746027.3755023
ISBN: 9798400720352
CFSSeg: Closed-Form Solution for Class-Incremental Semantic
Segmentation of 2D Images and 3D Point Clouds
Jiaxu Li
School of Computer Science and
Engineering, Central South University
Changsha, China
lijiaxu@csu.edu.cn
Rui Li
School of Computer Science and
Engineering, Central South University
Changsha, China
lirui.work0@gmail.com
Jianyu Qi
School of Computer Science and
Engineering, Central South University
Changsha, China
qijianyu@csu.edu.cn
Songning Lai
Hong Kong University of Science and
Technology (Guangzhou)
Guangzhou, China
songninglai@hkust.gz.edu.cn
Linpu Lv
Zhengzhou University
Zhengzhou, China
llp_2883@stu.zzu.edu.cn
Kejia Fan
Central South University
Changsha, China
kejiafan@csu.edu.cn
Jianheng Tang$^*$
Peking University
Peking, China
tangentheng@gmail.com
Yutao Yue
Hong Kong University of Science and
Technology (Guangzhou)
Guangzhou, China
yutaoyue@hkust.gz.edu.cn
Dongzhan Zhou
Shanghai Artificial Intelligence
Laboratory
Shanghai, China
dongzhan.zhou@gmail.com
Yunhuai Liu
Peking University
Peking, China
yunhuai.liu@pku.edu.cn
Abstract
2D images and 3D point clouds are foundational data types for mul- timedia applications, including real-time video analysis, augmented reality (AR), and 3D scene understanding. Class-incremental se- mantic segmentation (CSS) requires incrementally learning new- semantic categories while retaining prior knowledge. Existing mo- dards typically rely on computationally expensive training based on stochastic gradient descent, employing complex regularization or exemplar replay. However, stochastic gradient descent-based ap- proaches inevitably update the model's weights for past know- ing, leading to catastrophic forgetting, a problem exacerbated by pixel-point-level granularity. To address these challenges, we propose CFSSeg, a novel exemplar-free approach that leverages a closed-form solution, offering a practical and theoretically grounded solution for continual semantic segmentation tasks. This eliminates the need for iterative gradient-based optimization and storage of past data, requiring only a single pass through new samples per
Corresponding author
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
MM '25, Dublin, Ireland.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10
https://doi.org/10.1145/3746027.3755023
Huiping Zhuang∗
South China University of Technology
Guangzhou, China
bruis_zhuang@hotmail.com
step. It not only enhances computational efficiency but also provides a practical solution for dynamic, privacy-sensitive multimedia environments. Extensive experiments on 2D and 3D benchmark datasets such as Pascal VOC2012, S3DIS, and ScanNet demonstrate CFSSeg's superior performance.
CCS Concepts
* Computing methodologies → Computer vision; Computer vision tasks; Computer vision problems; Image segmentation; 3D imaging.
Keywords
Semantic segmentation; Continual learning
ACM Reference Format:
Jiaxu Li, Rui Li, Jianyu Qi, Songning Lai, Linpu Lv, Kejia Fan, Jianheng Tang, Yutao Yue, Dongzhan Zhou, Yunhuai Liu, and Huiping Zhuang. 2025. CF-SSeg: Closed-Form Solution for Class-Incremental Semantic Segement of 2D Images and 3D Point Clouds. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25). October 27-31, 2025, Dublin, Ireland.
ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3746027.3755023
1 Introduction
2D images and 3D point clouds are fundamental data modalities
that underpin modern multimedia applications, including real-time
video analysis, augmented reality (AR), robotics, and immersive
3D scene understanding [15, 21, 50, 51, 68]. However, real-world
multimedia systems rarely use fixed and predefined sets of object
categories. They often encounter new objects or concepts after
initial deployment, requiring the ability to adapt and expand their knowledge base over time [49, 55]. A naive approach is to train models directly on newly arrived data, but this strategy is plagued by catastrophic forgetting [18, 37, 70], where the model forgets previously acquired knowledge while adapting to new information. To address this issue, continual learning methods have been proposed to mitigate the effects of forgetting while allowing models to gradually adapt to new data [55].
247
MM '25, October 27-31, 2025, Dublin, Ireland. Jiaxu Li et al.
Class-incremental semantic segmentation (CSS) [3, 16, 35, 36] presents unique challenges. As shown in Figure 1, the model needs to continuously learn to segment new categories. Compared to image classification tasks, semantic segmentation tasks involve pixel/point-level granularity, which typically requires substantial computational resources and makes models more susceptible to catastrophic forgetting [67]. A challenge in CSS is the stability-plasticity dilemma [55], which involves balancing two conflicting goals: stability and plasticity. Stability refers to the model's ability to retain knowledge from past tasks, while plasticity requires the model to adapt to new incoming data. Striking the right balance is crucial for successful continual learning.
Recent advancements in CSS can be broadly categorized into exemplar-free and exemplar-based approaches. Exemplar-free methods aim to perform class-incremental learning without relying on historical data or features to reduce knowledge forgetting. These methods often employ self-supervised learning [46], regularization techniques [3, 35], or dynamic network architectures [2, 19]. On the other hand, exemplar-based methods depend on strategies such as sample replay, feature replay, auxiliary dataset integration [71], or pseudo-data generated by generative models [66]. While these methods show promise in retaining knowledge, they are all based on gradient descent and inevitably erase past knowledge through gradient updates [18, 72]. Moreover, they often demand significant computational resources, and some exemplar-based methods may not be suitable in scenarios where data privacy is paramount.
Analytic learning, as an alternative to stochastic gradient descent methods, overcomes key challenges associated with backpropagation, including gradient vanishing and the instability of iterative training processes, by directly computing neural network parameters [20, 40, 41, 52, 53, 56]. Inspired by this, we have proposed CFSSeg, a closed-form solution for CSS. Unlike existing incremental learning methods based on stochastic gradient descent, which require multiple training epochs, our approach needs only a single training epoch. Specifically, we freeze the encoder and update the model using a closed-form solution to achieve stability, while mapping features to a higher-dimensional space to make them more linearly separable, thereby enhancing plasticity. At the same time, it is effective and privacy-preserving, making it suitable for practical applications. Additionally, in disjoint and overlapped settings, semantic drift can occur [3, 16, 67], where previously learned categories collapse into background class labels in new datasets. We introduce a pseudo-labeling strategy that leverages uncertainty [16, 63] to mitigate semantic drift. The overview of our method is shown in Figure 2. Extensive experiments on 2D and 3D benchmark datasets such as Pascal VOC2012 [17], S3DIS [1], and ScanNet [14] have demonstrated its superior performance.
The key contributions are summarized as follows:
<description>这幅图展示了一个名为“半分类半段学习”的图表，其中包含一个由三个主要部分组成的模型，分别标记为Model 1、Model 2和Model 3。每个部分都包含一个带有不同类别的图像：飞机、人和椅子。这些图像通过连接它们的线条连接在一起，表示它们之间的关系或关系。在Model 1和Model 2之间，有一条蓝色箭头，表示模型从一个类别到另一个类别。在Model 3中，有一条紫色箭头，表示模型从一个类别到另一个类别。在模型的右侧，有一个带有“新数据”标签的箭头，表示模型正在从新数据中学习。在模型下方，有三个带有“测试集”标签的测试集，每个测试集包含一个类别：飞机、人和椅子。在模型的左侧，有一个带有“新数据”标签的箭头，表示模型正在从新数据中学习。图表的背景是白色的，文本和箭头是紫色和蓝色的。</description>
<text>Class-incremental semantic segmentation learning

Model 1 Model 2 Model 3 Model T

Test set 1 Test set 2 Test set 3 Test set T

Airplane, Airplane, Person, Person, Car, Airplane, Person, Person, Car, ...</text>
Class-incremental semantic segmentation learning
Figure 1: Illustration of the class-incremental semantic segmentation learning process. At each step, the model is incrementally trained on new classes while retaining knowledge of previously learned classes. For example, initially, the model is introduced to the "airplane" class. Subsequently, the model learns additional classes, with each step introducing new categories, progressively learning "person," "car," and "chair," among others, while expanding its knowledge and maintaining understanding of earlier classes.
* We propose a novel, gradient-free, closed-form solution for exemplar-free continual semantic segmentation in both 2D images and 3D point clouds.
* We develop a recursive update mechanism for the classification head, enabling efficient single-pass incremental learning without storing past data.
* Through extensive experiments on Pascal VOC2012, S3DIS, and ScanNet, we demonstrate that our method achieves advanced results while offering significant advantages in computational efficiency and data privacy.
2 Related Work
**Semantic Segmentation.** Semantic segmentation, a dense prediction task, involves assigning a semantic label to every pixel in an image. In recent years, significant progress has been made in this domain, primarily driven by the development of convolutional neural network (CNN)-based models [5-8]. More recently, Transformer-based architectures [10, 11, 60] and innovative Mamba frameworks [30, 32, 33] have gained prominence, introducing novel methodologies and perspectives for addressing the challenges of semantic segmentation. DeeplabV3 [7] has been widely used in previous CSS work, and we selected it as our 2D segmentation model.
For the 3D point cloud modality, key methods include PointNet
[44] and its derivative architecture PointNet++ [45], which are used
to directly process point cloud data; Transformer models such as
Point Transformer V3 [58], which improve performance by captur-
ing long-range dependencies; and DGCNN [57], which is based on
the EdgeConv module, captures local neighborhood information,
and learns global shape properties by stacking multiple layers. This
dynamic graph approach makes it particularly suitable for handling
the unstructured nature of point clouds. In this paper, we adopt
DGCNN as our 3D segmentation model due to its simplicity and effectiveness.
248
CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds MM '25, October 27-31, 2025, Dublin, Ireland.
**2D Class-Incremental Semantic Segmentation.** Class-incremental semantic segmentation, initially proposed in medical imaging applications [38, 39], has since been extended to natural image datasets [3, 35]. Unlike standard classification tasks, CSS poses unique challenges due to its pixel-level granularity, which exacerbates the issue of catastrophic forgetting [67]. CSS methods are broadly categorized into exemplar-free and exemplar-based approaches. Exemplar-free methods often leverage strategies such as self-supervised learning [36, 46, 61], regularization techniques [3, 16, 35], or dynamic network architectures [2, 19, 65] to retain knowledge from previously seen data. On the other hand, exemplar-based methods employ mechanisms such as sample replay, feature replay, and auxiliary dataset integration [4, 24, 54, 71], or utilize pseudo-data or pseudo-features generated by generative models [31, 34, 48, 66], combining these with new data to enable continual training.
**3D Class-Incremental Semantic Segeneration.** There is limited work on continual learning for 3D semantic segmentation, and it has only recently begun to be explored. Yang et al. [63] proposed a class-incremental learning method combining geometric features and uncertainty estimation. LGKD [64] introduced a label-guided knowledge distillation loss. Chen et al. [9] investigated class-incremental learning for mobile LiDAR point clouds, proposing strategies for feature representation preservation and loss cross-coupling.
3 Background
To begin, we define the objective of the semantic segmentation task. The input space is represented as $X \in \mathbb{R}^{N \times C_{in}}$, where $N$ denotes the number of input elements (pixels or point clouds), and $C_{in}$ represents the number of channels per element (e.g., RGB for images, or RGB, XYZ, normals for point clouds). The output label space is $Y \in C^N$, where the set of classes is $C$, including the background class $c_b \in C$. Given the training dataset $T = X \times Y$, the goal is to learn a mapping function $q$ parameterized by $\theta$ that predicts a per-element class probability distribution: $q_\theta: X \to \mathbb{R}^{N \times |C|}$. The segmentation mask is then computed as: $\hat{y} = \argmax q_\theta(x)[i, c]$, where $q_\theta(x)[i, c]$ indicates the probability of element $i$ belonging to class $c$.
In the traditional supervised learning paradigm, the entire training set $\mathcal{T}$ is provided at once, and the model is trained in a single step. However, in continual learning, training is performed iteratively, with each step introducing new categories along with their corresponding subset of training data. This process spans multiple steps, denoted as {step 1, step 2, ..., step $T$}. In step $t$, the label set $C_{t-1}$ is expanded by adding a new set of categories $S_t$, resulting in an updated label set $C_t = C_{t-1} \cup S_t$. Simultaneously, a new training subset $\mathcal{T}_t$ is introduced to update the previous model $q_{\theta_{t-1}}$ to $q_{\theta_t}$. According to the CSS principle, the newly introduced category sets are mutually exclusive, i.e., $S_i \cap S_j = \emptyset$ for $i \neq j$.
Different learning settings are considered for CSS, depending on the availability and labeling of categories during incremental learning. Sequential, disjoint, and overapped settings are detailed below:
1. **Sequential Setting.** In the sequential setting, labels for both previously learned and newly introduced categories are available.
simultaneously during each incremental learning step. 2) Disjoint Setting. The disjoint setting introduces complexity by labeling previously learned categories as background in the current task. This phenomenon, known as semantic drift, challenges the model to differentiate between real background and previously learned classes. 3) Overlapped Setting. The overlapped setting further complicates the learning process. Here, only new categories and the background are labeled, but the background label can encompass true background, previously learned categories, and future categories that have not yet been introduced.
4 Method
4.1 Ridge Regression
In step 1, we use stochastic gradient descent to train an encoder.
Notably, a powerful pre-trained encoder (e.g., SAM [25]) can also
be used to avoid this training process. We then save and freeze the
encoder, treating it as a feature extractor.
After training to obtain an encoder using the gradient descent method, during the continual learning phase, we do not use the gradient descent method to train the model. This is because the update of gradients will inevitably interfere with the weights of previous tasks [18, 72], leading to forgetting. Therefore, we adopt a simpler ridge regression, which has a closed-form solution.
Although freezing the backbone resolves the stability issue, it affects the model's plasticity. In order to increase the plasticity, according to Cover's Theorem [13], non-linearly mapping the features to a high-dimensional space can increase the probability of the features being linearly separable, we adopt a simple high-dimensional mapping method. The features extracted from the encoder are passed through a Randomly-initialized Hidden Layer (RHL) followed by a non-linear activation function (ReLU). The RHL is a linear layer whose weights are initialized from a normal distribution. Let $X_1^{\text{encoder}}$ and $Y_1^{\text{train}}$ be the feature matrix encoded by the encoder and the label matrix in step 1. We denote
$\mathbf{E}_1 = \text{ReLU}(\mathbf{X}_1^{\text{encoder}} \Phi^E), \quad (1)$
where $\Phi^E \in \mathbb{R}^{d_{\text{encoder}} \times d_E}$, and $d_{\text{encoder}}$ is the number of channels of the features extracted by the encoder. Generally, $d_{\text{encoder}} \ll d_E$. The features $E_1 \in \mathbb{R}^{K_1 \times d_E}$ are employed to predict the label matrix $Y_1^{\text{train}} \in \mathbb{R}^{K_1 \times C_1}$, where $K_1$ is the input number of elements. The matrix $Y_1^{\text{train}}$ is a label matrix formed by stacking one-hot labels. This prediction is carried out via the ridge regression approach, which entails solving the following optimization problem:
$$\operatorname *{argmin}_{\Phi _{1}}\ (\|\mathbf {Y}_{1}^{\mathrm {train}}-\mathbf {E}_{1}\mathbf {\Phi }_{1}\|_{\mathrm {F}}^{2}+\gamma \|\mathbf {\Phi }_{1}\|_{\mathrm {F}}^{2}),$$
(2)
where $\gamma$ is a regularization parameter, $\|\cdot\|_F$ represents the Frobenius norm. The optimal solution to this problem is:
$$\widehat {\mathbf {\Phi }}_{1}=(\mathbf {E}_{1}^{\top }\mathbf {E}_{1}+\gamma \mathbf {I})^{-1}\mathbf {E}_{1}^{\top }\mathbf {Y}_{1}^{\mathrm {train}},$$
(3)
where $I$ is the identity matrix, $^{-1}$ represents the matrix inversion operation, and $^\top$ denotes the matrix transpose operation.
249
MM '25, October 27-31, 2025, Dublin, Ireland. Jiaxu Li et al.
<description>这幅图像展示了一个名为“Dataset t-1”的图像，它似乎是一个用于机器学习任务的图像分类数据集。图像的左侧显示了各种物体和场景的图像，如自行车、人、椅子、桌子、窗户、沙发、椅子、桌子、椅子、沙发、椅子、桌子、椅子、桌子、椅子、桌子、椅子、桌子、椅子、桌子、椅子、桌子</description>
<text>Dataset t-1 or Novel: person or wall or background: t-1 or bike or floor or dataset t or Novel: dog or chair or stool or old or ground truth or Encoder: $x_t^{encoder}$ RHL Encoder: $x_t^{encoder}$ RHL</text>
Figure 2: Overview of the proposed method CFSSeg. In step $t$, the model from step $t-1$ is used to generate pseudo labels via Pseudo Labeling, which are then combined with ground truth labels to form mixed labels. The model inherits the classification head $\hat{\Phi}_{t-1}$ learned in step $t-1$, and combines it with the mixed labels, the extracted features $E_t$, and $\Psi_{t-1}$ from step $t-1$. The C-RLS algorithm is then used to update and obtain $\hat{\Phi}_t$ and $\Psi_t$.
4.2 Recursive Ridge Regression for CSS
The previous subsection introduced ridge regression learning, which, however, is not suitable for continual learning. Next, we will propose the concatenated recursive least squares (C-RLS) algorithm. Without loss of generality, let $Y_{1:t-1}^{\text{train}}$, $Y_{1:t}^{\text{train}}$ and $E_{1:t-1}$, $E_{1:t}$, be the accumulated label and feature matrices in step $t-1$ and $t$, and they are related via
$$\mathbf {Y}_{1:t}^{\mathrm {train}}=\begin{bmatrix}\mathbf {Y}_{1:t-1}^{\mathrm {train}}&\mathbf {0}\\
{\tilde {\mathbf {Y}}}_{t}^{\mathrm {train}}&{\tilde {\mathbf {Y}}}_{t}^{\mathrm {train}}\end{bmatrix},\quad \mathbf {E}_{1:t}=\begin{bmatrix}\mathbf {E}_{1:t-1}&\\
{\mathbf {E}}_{t}\end{bmatrix}.$$
(4)
The block matrix is due to the covered-uncovered partition
$$\mathbf {Y}_{t}^{\mathrm {train}}=\begin{bmatrix}\bar {\mathbf {Y}}_{t}^{\mathrm {train}}&\tilde {\mathbf {Y}}_{t}^{\mathrm {train}}\end{bmatrix},$$
(5)
where $\tilde{Y}_t^{\text{train}} \in \mathbb{R}^{N_t \times d_{C_{t-1}}}$ is the projected covered matrix and $\tilde{Y}_t^{\text{train}} \in \mathbb{R}^{N_t \times (d_{C_t} - d_{C_{t-1}})}$ is the projected uncovered matrix. They correspond to segments displaying the appearance of covered classes and uncovered classes, representing the already learned classes and the yet-to-be-learned classes, respectively. The learning problem can then be formulated as:
$$\underset {\Phi _{t-1}}{\text {argmin}}\ (\|\mathbf {Y}_{1:t-1}^{\text {train}}-\mathbf {E}_{1:t-1}\mathbf {\Phi }_{t-1}\|_{\mathrm {F}}^{2}+\gamma \|\mathbf {\Phi }_{t-1}\|_{\mathrm {F}}^{2}),$$
(6)
according to Eqn (3), at step $t-1$, we have:
$$\hat {\Phi }_{t-1}=(\mathbf {E}_{1:t-1}^{\top }\mathbf {E}_{1:t-1}+\gamma \mathbf {I})^{-1}\mathbf {E}_{1:t-1}^{\top }\mathbf {Y}_{1:t-1}^{\text {train}}$$
(7)
where $\hat{\Phi}_{t-1} \in \mathbb{R}^{d_E \times \sum_{i=1}^{t-1} d_{Si}}$, with the column size expanding as $t$ increases. Let
$$\boldsymbol {\Psi }_{t-1}=(\mathbf {E}_{1:t-1}^{\top }\mathbf {E}_{1:t-1}+\gamma \mathbf {I})^{-1}$$
(8)
be the inverted auto-correlation matrix, which captures the correa-
tion information from both current and past samples. Building on
this, the goal is to compute $\hat{\Phi}_t$ using only $\hat{\Phi}_{t-1}, \Psi_{t-1}$, and the current step's data, without involving historical samples. The process is formulated as shown in the following theorem.
## THEOREM 1. The $\Phi_t$ weights, recursively obtained by
$$\hat {\bm {\Phi }}_{t}=\begin{bmatrix}\hat {\bm {\Phi }}_{t-1}-\Psi _{t}\mathbf {E}_{t}^{\top }\mathbf {E}_{t}\hat {\bm {\Phi }}_{t-1}+\Psi _{t}\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{train}&\Psi _{t}\mathbf {E}_{t}^{\top }\tilde {\mathbf {Y}}_{t}^{train}\end{bmatrix}\quad (9)$$
are equivalent to those obtained from Eqn (7) for step t. The matrix $\Psi_t$ can also be recursively updated by
$$\Psi _{t}=(\Psi _{t-1}^{-1}+\mathbf {E}_{t}^{\top }\mathbf {E}_{t})^{-1}.$$
(10)
PROOF. See https://g-aqqk9025Coding.net/s/12800b98-8daa-47ca-ee7-52a17768e5b3.
4.3 Theoretical Analysis
**Privacy Protection.** Our method ensures data privacy in two ways: first, by eliminating the need to store historical data samples; second, by guaranteeing that historical raw data samples cannot be recovered from the $\Psi$ matrix through reverse engineering.
**Computational Complexity.** The computational complexity analysis reveals that the time complexity for each step includes $O(d_E^3)$ for updating $\Psi_t$ via matrix inversion, and $O(d_E^2N_t + d_EN_t^2 + d_E^2C_t)$ for updating $\Phi_t$ via matrix multiplication. These operations can be efficiently parallelized on GPU.
Space Complexity. The space complexity is $O(d_E^2 + d_E N_t + d_E C_t)$: $O(d_E^2)$ is for storing the $\Psi_t$ matrix, $O(d_E N_t)$ for storing the feature matrix $E_t$, $O(d_E C_t)$ for storing the classifier matrix $\Phi_t$.
4.4 Pseudo-Labeling for 2D Images
At step $t$, $(x_t, y_t) \in \mathcal{T}_t$, where $x_t^i, y_t^i$ represent the elements and their corresponding ground truth labels, respectively. In both disjoint and
overlapped settings, previously learned classes are treated as back-
ground in the current task, a phenomenon commonly referred to
as semantic drift. To address this issue, we adopt a pseudo-labeling
approach. We define the uncertainty of an element as follows:
250
CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds  
MM '25, October 27-31, 2025, Dublin, Ireland.
$$U_{i}=1-\sigma (\max _{c}(q_{\theta _{t-1}}(i,c))),$$
(11)
where $q_{\theta_{t-1}}(i, c)$ denotes the logit output of the model for element $i$ in class $c$, $\sigma$ represents the Sigmoid activation function, and $\hat{g}_i^{t-1} = \argmax_{c} q_{\theta_{t-1}}(i, c)$ denotes the predicted label from the model at step $t-1$. The pseudo-labeling strategy is then defined as:
$$\tilde {y}_{t}^{i}=\begin{cases}y_{t}^{i},&y_{t}^{i}\in S_{t},\\
y _{t}^{i},&(y_{t}^{i}=c_{b})\wedge (U_{i}>\tau ),\\
hat {y}_{t-1}^{i},&(y_{t}^{i}=c_{b})\wedge (U_{i}\leq \tau ),\end{cases}$$
(12)
where $\tau$ is a threshold that determines whether the pixel labeled as background should adopt pseudo-label generated by the prior model. This approach mitigates the semantic drift issue in both disjoint and overlapped settings.
4.5 pseudo-labeling for 3D Point Clouds
For point cloud $i$, we employ the KNN algorithm to identify its $K$ nearest neighbors based on the $xyz$ coordinates and compute the cosine similarity $w_k$ between point cloud $i$ and its $K$ neighbors using their $xyz$ coordinates. We adopt a neighborhood spatial aggregation method based on Monte Carlo dropout (MC-dropout) technique, which achieves efficient estimation of point distribution uncertainty through a single forward propagation. This approach utilizes a spatial dependency sampling mechanism, and its effectiveness has been validated in the literature [43, 63]. For uncertainty quantification, we employ the Bayesian Active Learning Disagreement (BALD) criterion [23] as the core evaluation function for point cloud spatial sampling. Specifically, given an input point cloud, its uncertainty metric function can be expressed as:
$$\begin{array}{c}U_{i}=-\displaystyle \sum _{c}\left [\frac {1}{K}\sum _{k}q_{t-1}(i,c)\cdot w_{k}\right ]\log \left [\frac {1}{K}\sum _{k}q_{t-1}(i,c)\cdot w_{k}\right ]\\
\\
+\displaystyle \frac {1}{K}\displaystyle \sum _{c,k}\left ( q_{t-1}(i,c)\cdot w_{k}\right )\log \left ( q_{t-1}(i,c)\cdot w_{k}\right ).\end{array}$$
And then the pseudo-labeling method is defined as:
(13)
$$\tilde {y}_{t}^{i}=\begin{cases}y_{t}^{i},& y_{t}^{i}\in S_{t},\\
 \hat {y}_{t-1}^{i},&(y_{t}^{i}=c_{b})\wedge (\hat {y}_{t-1}^{i}\neq c_{b})\wedge (U_{i}\leq \tau ),\\
 \hat {y}_{t-1}^{i,k^{\prime }},&(y_{t}^{i}=c_{b})\wedge \left ((\hat {y}_{t-1}^{i}=c_{b})\vee (U_{i}>\tau )\right ),\\
 c_{b},&\text {otherwise},\end{cases}$$
(14)
where $\hat{y}_{t-1}^{i,k'}$ represents the predicted label of the nearest neighbor of point cloud $i$, with $\hat{y}_{t-1}^{i,k'} \neq c_b$ and $U_{i,k'} \leq \tau$. It is worth noting that we may not be able to find such a point, in which case we mark the point as belonging to the background class.
5 Experiments
5.1 Experimental Setup
**2D Dataset.** We evaluate our method using public 2D semantic segmentation benchmarks: Pascal VOC2012 [17]. It contains 21 classes (including background class). This dataset features wild scenes, with 10,582 images used for training and 1,449 images for validation.
**3D Dataset.** We evaluate our method using two public 3D point cloud segmentation benchmarks: S3DIS [1] and ScanNet [14]. These datasets are selected for their diversity, relevance to our problem domain, and ability to facilitate fair comparisons with existing benchmark methods [26, 63]. S3DIS comprises point clouds from 272 rooms across 6 indoor areas, with each point containing xyz coordinates and RGB information, manually annotated with one of 13 predefined classes. Following standard practice [63], we designate the more challenging Area 5 as the validation set, while the remaining areas are used for training. ScanNet, on the other hand, is an RGB-D video dataset featuring 1,513 scans from 707 indoor scenes. Each point is labeled with one of 21 classes, including 20 semantic classes and an additional category for unannotated places. Adhering to the standard dataset splits [14], we allocate 1,210 scans for training and 312 scans for validation. We adopt a sliding window [45, 57, 63] to partition the rooms in the S3DIS and ScanNet datasets, generating 7,547 and 36,350 1m×1m blocks respectively, and randomly sample 2,048 points from each block as input data. We use two sequences, $S^0$ and $S^1$, to partition the 3D dataset. $S^0$ follows the original dataset's annotation order, while $S^1$ follows the alphabetical order of the category names.
**CSS Learning Protocol.** The classes of the images for the current step include $C_{t-1} \cup S_t$. In each step, we continuously introduce new classes for learning. In an $m-n$ setting, the model first learns $m$ classes, and in each subsequent step, it incrementally learns $n$ classes. For 2D CSS, we adopt the three settings: sequential, disjoint, and overlapped. For 3D CSS, we follow [63] and use the disjoint setting.
**Evaluation Metrics.** We use the widely adopted mean Intersection-over-Union (mIoU) metric to calculate the average IoU value across all classes. The IoU for a single class is computed as: $IU = \frac{TP}{TP+FP+FN}$, where $TP, FP, and FN$ represent true positives, false positives, and false negatives respectively. To comprehensively evaluate CSS performance, we compute mIoU values separately for initial classes $C_1$, incremental classes $C_T \setminus C_1$, and all classes $C_T$.
**Comparison Methods.** Our method is an exemplar-free method. For fairness, we also compare it with other exemplar-free methods. For 2D CSS, see Table 3 and Table 4 for details. As for 3D CSS, due to the limited number of relevant baselines, we consider Yang et al.'s method [63] as a strong baseline because it is the only open-source SOTA method to the best of our knowledge, along with the classic baseline EWC [26]. At the same time, we establish a naive baseline: FT, which fine-tunes both the backbone and the classification head. In addition, we include an upper bound, namely JT, which stands for joint training.
**Implementation Details.** For the initial training in step 1, we adopt DeepLabv3 [7] with a ResNet-101 [22] backbone pre-trained on ImageNet-1K. We set the number of epochs to 50 and the batch size to 32. We use SGD as the optimizer with a learning rate of 0.001.
Table 1: CSS Quantitative comparison of 3D class-incremental segmentation methods on the S3DIS and ScanNet datasets in $S^0$ split. BT stands for training only on the dataset from step 1. The best results achieved by the incremental methods are highlighted in bold.
251
MM '25, October 27–31, 2025, Dublin, Ireland. Jiaxu Li et al.
<table border="1"><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">8-1(6 steps)</td><td colspan="3">10-1(4 steps)</td><td colspan="3">12-1(2 steps)</td></tr><tr><td>0-7</td><td>8-12</td><td>all</td><td>0-9</td><td>10-12</td><td>all</td><td>0-11</td><td>12</td><td>all</td></tr><tr><td rowspan="6">S3DIS</td><td>BT</td><td>-</td><td>DGCNN</td><td>49.85</td><td>-</td><td>47.03</td><td>-</td><td>-</td><td>45.37</td><td>-</td><td>-</td></tr><tr><td>FT</td><td>-</td><td>DGCNN</td><td>5.45</td><td>4.94</td><td>5.25</td><td>11.54</td><td>15.48</td><td>28.34</td><td>34.02</td><td>28.77</td></tr><tr><td>EWC [26]</td><td>PNAS2017</td><td>DGCNN</td><td>30.99</td><td>4.16</td><td>20.67</td><td>39.14</td><td>14.73</td><td>33.23</td><td>15.49</td><td>35.53</td></tr><tr><td>Yang et al.[63]</td><td>CVPR2023</td><td>DGCNN</td><td>36.90</td><td>13.76</td><td>27.80</td><td>36.98</td><td>26.60</td><td>34.59</td><td>38.47</td><td>44.07</td></tr><tr><td>Ours</td><td></td><td>DGCNN</td><td>49.77</td><td>28.69</td><td>41.66</td><td>45.26</td><td>34.05</td><td>42.67</td><td>40.71</td><td>44.08</td></tr><tr><td>JT</td><td>-</td><td>DGCNN</td><td>48.99</td><td>41.76</td><td>46.21</td><td>46.72</td><td>45.11</td><td>46.35</td><td>35.65</td><td>46.33</td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">15-1(6 steps)</td><td colspan="3">17-1(4 steps)</td><td colspan="3">19-1(2 steps)</td></tr><tr><td>0-14</td><td>15-19</td><td>all</td><td>0-16</td><td>17-19</td><td>all</td><td>0-18</td><td>19</td><td>all</td></tr><tr><td rowspan="6">ScanNet</td><td>BT</td><td>-</td><td>DGCNN</td><td>37.52</td><td>-</td><td>34.72</td><td>-</td><td>-</td><td>32.64</td><td>-</td><td>-</td></tr><tr><td>FT</td><td>-</td><td>DGCNN</td><td>4.41</td><td>3.20</td><td>4.10</td><td>4.19</td><td>2.97</td><td>11.37</td><td>8.86</td><td>11.25</td></tr><tr><td>EWC [26]</td><td>PNAS2017</td><td>DGCNN</td><td>12.32</td><td>2.48</td><td>9.86</td><td>11.34</td><td>1.20</td><td>9.82</td><td>11.11</td><td>17.51</td></tr><tr><td>Yang et al.[63]</td><td>CVPR2023</td><td>DGCNN</td><td>8.46</td><td>4.44</td><td>7.46</td><td>12.29</td><td>6.48</td><td>11.42</td><td>25.56</td><td>11.31</td></tr><tr><td>Ours</td><td></td><td>DGCNN</td><td>32.56</td><td>10.22</td><td>26.97</td><td>29.59</td><td>12.83</td><td>27.98</td><td>28.54</td><td>16.88</td></tr><tr><td>JT</td><td>-</td><td>DGCNN</td><td>37.59</td><td>16.42</td><td>32.30</td><td>34.78</td><td>16.69</td><td>32.07</td><td>32.28</td><td>18.12</td></tr></table>
Table 2: CSS Quantitative comparison of 3D class-incremental segmentation methods on the S3DIS and ScanNet datasets in $S^1$ split. BT stands for training only on the dataset from step 1. The best results achieved by the incremental methods are highlighted in bold.
<table border="1"><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">8-1(6 steps)</td><td colspan="3">10-1(4 steps)</td><td colspan="3">12-1(2 steps)</td></tr><tr><td>0-7</td><td>8-12</td><td>all</td><td>0-9</td><td>10-12</td><td>all</td><td>0-11</td><td>12</td><td>all</td></tr><tr><td rowspan="6">S3DIS</td><td>BT</td><td>-</td><td>DGCNN</td><td>37.61</td><td>-</td><td>-</td><td>-</td><td>-</td><td>44.68</td><td>-</td><td>-</td></tr><tr><td>FT</td><td>-</td><td>DGCNN</td><td>12.69</td><td>2.57</td><td>8.80</td><td>10.11</td><td>2.33</td><td>8.31</td><td>25.05</td><td>11.75</td></tr><tr><td>EWC [26]</td><td>PNAS2017</td><td>DGCNN</td><td>17.50</td><td>2.59</td><td>11.77</td><td>20.54</td><td>8.86</td><td>17.84</td><td>23.20</td><td>11.40</td></tr><tr><td>Yang et al.[63]</td><td>CVPR2023</td><td>DGCNN</td><td>41.28</td><td>11.64</td><td>29.88</td><td>36.45</td><td>20.14</td><td>31.13</td><td>37.38</td><td>22.40</td></tr><tr><td>Ours</td><td></td><td>DGCNN</td><td>51.33</td><td>30.72</td><td>43.40</td><td>45.16</td><td>35.98</td><td>43.04</td><td>45.65</td><td>27.53</td></tr><tr><td>JT</td><td>-</td><td>DGCNN</td><td>39.29</td><td>57.42</td><td>46.26</td><td>42.19</td><td>59.08</td><td>46.09</td><td>46.96</td><td>38.89</td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">15-1(6 steps)</td><td colspan="3">17-1(4 steps)</td><td colspan="3">19-1(2 steps)</td></tr><tr><td>0-14</td><td>15-19</td><td>all</td><td>0-16</td><td>17-19</td><td>all</td><td>0-18</td><td>19</td><td>all</td></tr><tr><td rowspan="6">ScanNet</td><td>BT</td><td>-</td><td>DGCNN</td><td>29.01</td><td>-</td><td>-</td><td>30.24</td><td>-</td><td>31.59</td><td>-</td><td>-</td></tr><tr><td>FT</td><td>-</td><td>DGCNN</td><td>4.20</td><td>1.61</td><td>3.55</td><td>3.63</td><td>1.00</td><td>3.24</td><td>9.04</td><td>0.22</td></tr><tr><td>EWC [26]</td><td>PNAS2017</td><td>DGCNN</td><td>14.93</td><td>33.30</td><td>19.52</td><td>8.78</td><td>31.74</td><td>12.22</td><td>12.65</td><td>3.19</td></tr><tr><td>Yang et al.[63]</td><td>CVPR2023</td><td>DGCNN</td><td>12.17</td><td>0.16</td><td>9.17</td><td>10.19</td><td>5.67</td><td>9.52</td><td>25.08</td><td>15.28</td></tr><tr><td>Ours</td><td></td><td>DGCNN</td><td>28.20</td><td>16.09</td><td>25.18</td><td>28.43</td><td>15.01</td><td>26.42</td><td>28.71</td><td>11.84</td></tr><tr><td>JT</td><td>-</td><td>DGCNN</td><td>28.93</td><td>39.06</td><td>31.46</td><td>30.12</td><td>40.33</td><td>31.65</td><td>31.29</td><td>30.03</td></tr></table>
$10^{-2}$, a momentum of 0.9, and a weight decay of $10^{-4}$, combined with a polynomial learning rate scheduler. The loss function is binary cross-entropy (BCE). For the 3D CSS encoder, we employ DGCNN [57] with a batch size of 32 and the Adam optimizer, using an initial learning rate of 0.001 and a weight decay of 0.0001 for 100 epochs. In the continual learning step, we freeze the encoder and insert an RHL layer. In the 2D experiments, we set $d_E$ to 8192,
$\gamma$ to 1, and $\tau$ to 0.4. In the 3D experiments, we set $d_E$ to 5000, $\gamma$ to 1, and $\tau$ to 0.0035 and 0.001 on the S3DIS and ScanNet datasets via cross-validation respectively.
252
CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds MM '25, October 27-31, 2025, Dublin, Ireland.
Table 3: CSS quantitative comparison on Pascal VOC2012 in mIoU (%) under sequential setting. The results of the comparison method are directly taken from the original work and [67]. For the best results, we use bold formatting.
<table border="1"><tr><td rowspan="2"></td><td rowspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">15-1(6 steps)</td><td colspan="3">15-5(2 steps)</td></tr><tr><td>0-15</td><td>16-20</td><td>all</td><td>0-15</td><td>16-20</td><td>all</td></tr><tr><td rowspan="9">Sequential</td><td>FT</td><td></td><td>DeepLabv3+</td><td>49.0</td><td>17.8</td><td>41.6</td><td>62.0</td><td>38.1</td><td>56.3</td></tr><tr><td>LwF [28]</td><td>TPAMI2018</td><td>DeepLabv3+</td><td>33.7</td><td>13.7</td><td>29.0</td><td>68.0</td><td>43.0</td><td>62.1</td></tr><tr><td>LwF-MC[47]</td><td>CVPR2017</td><td>DeepLabv3+</td><td>12.1</td><td>1.9</td><td>9.7</td><td>70.6</td><td>19.5</td><td>58.4</td></tr><tr><td>ILT [35]</td><td>ICCIVW2019</td><td>DeepLabv3+</td><td>49.2</td><td>30.3</td><td>48.3</td><td>71.3</td><td>47.8</td><td>65.7</td></tr><tr><td>CIL [27]</td><td>ITSC2020</td><td>DeepLabv3+</td><td>52.4</td><td>22.3</td><td>45.2</td><td>63.8</td><td>39.8</td><td>58.1</td></tr><tr><td>MiB [3]</td><td>CVPR2020</td><td>DeepLabv3+</td><td>35.7</td><td>11.0</td><td>29.8</td><td>73.0</td><td>44.4</td><td>66.1</td></tr><tr><td>SDR [36]</td><td>CVPR2021</td><td>DeepLabv3+</td><td>58.5</td><td>10.1</td><td>47.0</td><td>73.6</td><td>46.7</td><td>67.2</td></tr><tr><td>SDR+MiB [36]</td><td>CVPR2021</td><td>DeepLabv3+</td><td>58.1</td><td>11.8</td><td>47.1</td><td>74.6</td><td>43.8</td><td>67.3</td></tr><tr><td>Ours</td><td></td><td>DeepLabv3</td><td>78.1</td><td>42.0</td><td>70.0</td><td>78.1</td><td>42.0</td><td>70.0</td></tr></table>
Table 4: CSS quantitative comparison on Pascal VOC2012 in mIoU (%) under *disjoint and *overlapped settings. The results of the comparison method are directly taken from the original work and [67]. For the best results, we use bold formatting.
<table border="1"><tr><td rowspan="2" colspan="2">Method</td><td rowspan="2">Year</td><td rowspan="2">Model</td><td colspan="3">15-1(6 steps)</td><td colspan="3">10-1(11 steps)</td></tr><tr><td>0-15</td><td>16-20</td><td>all</td><td>0-10</td><td>11-20</td><td>all</td></tr><tr><td rowspan="6">Disjoint</td><td>FT</td><td></td><td>DeepLabv3</td><td>0.20</td><td>1.80</td><td>0.60</td><td>6.30</td><td>1.10</td><td>3.80</td></tr><tr><td>MiB[3]</td><td>CVPR2020</td><td>DeepLabv3</td><td>46.20</td><td>12.90</td><td>37.90</td><td>9.50</td><td>4.10</td><td>6.90</td></tr><tr><td>PLOP [16]</td><td>CVPR2021</td><td>DeepLabv3</td><td>57.86</td><td>13.67</td><td>46.48</td><td>9.70</td><td>7.00</td><td>8.40</td></tr><tr><td>SDR [36]</td><td>CVPR2021</td><td>DeepLabv3+</td><td>59.40</td><td>14.30</td><td>48.70</td><td>17.30</td><td>11.00</td><td>14.30</td></tr><tr><td>RCIL [69]</td><td>CVPR2022</td><td>DeepLabv3</td><td>66.10</td><td>18.20</td><td>54.70</td><td>30.60</td><td>4.70</td><td>18.20</td></tr><tr><td>Ours</td><td></td><td>DeepLabv3</td><td>77.66</td><td>40.33</td><td>68.77</td><td>70.85</td><td>42.13</td><td>57.17</td></tr><tr><td rowspan="15">Overlapped</td><td>FT</td><td></td><td>DeepLabv3</td><td>0.20</td><td>1.80</td><td>0.60</td><td>6.30</td><td>2.80</td><td>4.70</td></tr><tr><td>EWC [26]</td><td>PNAS2017</td><td>DeepLabv3</td><td>0.30</td><td>4.30</td><td>1.30</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LwF-MC [47]</td><td>CVPR2017</td><td>DeepLabv3</td><td>6.40</td><td>8.40</td><td>6.90</td><td>4.65</td><td>5.90</td><td>4.95</td></tr><tr><td>ILT [35]</td><td>ICCVV2019</td><td>DeepLabv3</td><td>4.90</td><td>7.80</td><td>5.70</td><td>7.15</td><td>3.67</td><td>5.50</td></tr><tr><td>MiB[3]</td><td>CVPR2020</td><td>DeepLabv3</td><td>34.22</td><td>13.50</td><td>29.29</td><td>12.25</td><td>13.09</td><td>12.65</td></tr><tr><td>PLOP [16]</td><td>CVPR2021</td><td>DeepLabv3</td><td>65.12</td><td>21.11</td><td>54.64</td><td>44.03</td><td>15.51</td><td>30.45</td></tr><tr><td>UCD+PLOP [61]</td><td>TPAMI2022</td><td>DeepLabv3</td><td>66.30</td><td>21.60</td><td>55.10</td><td>42.30</td><td>28.30</td><td>35.30</td></tr><tr><td>REINDER [42]</td><td>CVPR2022</td><td>DeepLabv3</td><td>68.30</td><td>27.23</td><td>58.52</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RCIL [69]</td><td>CVPR2022</td><td>DeepLabv3</td><td>70.60</td><td>23.70</td><td>59.40</td><td>55.40</td><td>15.10</td><td>34.30</td></tr><tr><td>SPPA [29]</td><td>ECCV2022</td><td>DeepLabv3</td><td>66.20</td><td>23.30</td><td>56.00</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CAF [62]</td><td>TMM2022</td><td>DeepLabv3</td><td>55.70</td><td>14.10</td><td>45.30</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AWT+MiB [19]</td><td>WACV2023</td><td>DeepLabv3</td><td>59.10</td><td>17.20</td><td>49.10</td><td>33.20</td><td>18.00</td><td>26.00</td></tr><tr><td>EWF+MiB [59]</td><td>CVPR2023</td><td>DeepLabv3</td><td>78.00</td><td>25.50</td><td>65.50</td><td>56.00</td><td>16.70</td><td>37.30</td></tr><tr><td>GSC[12]</td><td>TMM2024</td><td>DeepLabv3</td><td>72.10</td><td>24.40</td><td>60.80</td><td>50.60</td><td>17.30</td><td>34.70</td></tr><tr><td>Ours</td><td></td><td>DeepLabv3</td><td>79.16</td><td>38.00</td><td>69.36</td><td>75.02</td><td>41.20</td><td>58.91</td></tr><tr><td>JT</td><td>-</td><td></td><td>DeepLabv3</td><td>79.77</td><td>72.35</td><td>77.43</td><td>78.41</td><td>76.35</td><td>77.43</td></tr></table>
5.2 Main Results
**2D Experimental Results.** Extensive experiments on the Pascal VOC2012 dataset demonstrate the outstanding performance of our method across all evaluation settings (Tables 3 and 4). Under the sequential 15-1 configuration, our approach achieves an overall mIoU of 70.0%, significantly surpassing current state-of-the-art methods. The method maintains strong performance on base classes (78.1% mIoU) while retaining high accuracy for novel classes (42.0% mIoU), effectively addressing the inherent stability-plasticity dilemma in continual learning. Notably, identical performance metrics in both the 15-1 and 15-5 sequential settings confirm the mathematical consistency of our closed-form solution. The method's advantages become even more pronounced in challenging scenarios: under the disjoint 15-1 setting, it achieves 68.77% mIoU across all classes (77.66% for base classes, 40.33% for novel classes), while in the disjoint 10-1 setting, the performance gap widens dramatically (57.17% vs. 18.20% for the next competitor). Similarly, in overlapped scenarios, our method maintains its superiority with 69.36% mIoU in the 15-1 configuration and 58.91% in the 10-1 setting, outperforming existing approaches across all class categories. These compelling results validate the theoretical advantages of the closed-form solution.
in mitigating catastrophic forgetting while efficiently integrating new knowledge.
3D Experimental Results. Our method demonstrates exceptional performance in 3D point cloud segmentation tasks across multiple benchmark datasets. On the S3DIS dataset with the $S^0$ split, it outperforms existing approaches in all evaluation protocols: the 8-1 configuration achieves an overall mIoU of 41.66% (49.77% for base classes, 28.69% for novel classes), while the 10-1 and 12-1 configurations reach 42.67% and 44.08% mIoU, respectively. With the $S^1$ split, the 8-1 configuration improves further to 43.40% mIoU (51.33% for base classes, 30.72% for novel classes), setting a new state-of-the-art benchmark for 3D continual semantic segmentation.

The method also proves effective on the more challenging ScanNet dataset: under the $S^0$ split, the 15-1 configuration achieves 26.97% mIoU, significantly outperforming previous methods that struggled to exceed 10% mIoU. The 17-1 and 19-1 configurations yield overall mIoU scores of 27.98% and 27.96%, respectively. On the $S^1$ split, the three configurations achieve 25.18%, 26.42%, and 27.84% mIoU. The consistent performance across diverse 3D configurations underscores the versatility and robustness of our closed-form solution in the point cloud domain, providing a theoretically grounded and practical solution for real-world 2D and 3D semantic segmentation applications.
253
MM '25, October 27-31, 2025, Dublin, Ireland. Jiaxu Li et al.
**Class Order Robustness.** Our method demonstrates strong robustness to class order variations, which is a critical aspect in continual learning scenarios. This robustness stems from two key factors: First, the closed-form solution ensures deterministic and unique classification head weights for a given set of training data. As evidenced in Table 3, our method achieves identical performance in both the sequential 15-1 and 15-5 settings. This consistency is a direct consequence of the closed-form nature of our solution, which guarantees the same optimal weights regardless of the training sequence. Second, for 3D datasets, we observe that performance variations across different class orders are primarily influenced by the backbone network's feature extraction capabilities. When using the same backbone architecture, the performance remains remarkably stable across different class sequences. Minor variations in performance can be attributed to the backbone's sensitivity to different class orders during the initial training phase, rather than the continual learning mechanism itself. This robustness to class order variations is particularly valuable in real-world applications where the sequence of class introduction cannot be predetermined. Our method's ability to maintain consistent performance regardless of the learning order makes it more practical for deployment in dynamic environments.
5.3 Ablation Studies
To rigorously evaluate the contributions of each component in our framework, we conducted comprehensive ablation experiments using the challenging Pascal VOC2012 overlapped 10-1 setting. The results presented in Table 5 reveal several key insights into the effectiveness of our approach.
**Effect of RHL.** First, it is evident from the table that removing the RHL component led to a significant performance drop, particularly on new classes (moU decreased from 41.20% to 9.36%), while the overall performance also sharply declined from 58.91% to 37.94%. This result underscores the critical role of RHL in enhancing the model's plasticity while maintaining stability for previously learned classes. By mapping features into a higher-dimensional space, RHL makes them more linearly separable, thereby improving the model's ability to learn new categories.
**Effect of Pseudo-Labeling.** Second, our analysis demonstrates that the pseudo-labeling mechanism plays a vital role in preventing semantic drift, which is particularly evident in incremental learning scenarios with both disjoint and overlapped classes. Without the pseudo-labeling mechanism, the model's performance on old classes dropped from 75.02% to 71.83%, and on new classes from 41.20% to 36.19%, with the overall performance decreasing to 54.86%. This indicates that pseudo-labeling effectively mitigates representation shifts, helping the model retain accurate recognition capabilities for previously learned classes.
These experimental results empirically validate our theoreti-
cal framework, confirming that each architectural component ad-
resses specific challenges in continual semantic segmentation,
collectively contributing to the robust performance observed in our
comprehensive experiments.
Table 5: "Ours" represents the complete model setting; "w/o RHL" means the model excludes RHL; "w/o Pseudo-Labeling" indicates that the model does not use the pseudo-labeling.
<table border="1"><tr><td>Settings</td><td>0-10</td><td>11-20</td><td>all</td></tr><tr><td>Ours</td><td>75.02</td><td>41.20</td><td>58.91</td></tr><tr><td>w/o RHL</td><td>63.91</td><td>9.36</td><td>37.94</td></tr><tr><td>w/o Pseudo-Labeling</td><td>71.83</td><td>36.19</td><td>54.86</td></tr></table>
5.4 Efficiency Studies
As analyzed in Sec 4.3, our method demonstrates remarkable computational efficiency advantages over traditional gradient-based optimization approaches. Updating the classification head parameters only requires matrix multiplication and matrix inversion, which can be efficiently parallelized on GPU. As quantified in Table 6, our closed-form solution achieves convergence in just a single training epoch, taking only 43.25 seconds, whereas fine-tuning methods requiring multiple gradient descent iterations need 651.46 seconds.

This represents a 15× acceleration in training time while maintaining superior segmentation performance. Furthermore, despite supporting larger batch sizes (64 vs. 32), our method significantly reduces memory consumption (51.61 GB vs. 59.55 GB) due to the closed-form nature of our solution, which eliminates the need to store historical data or intermediate gradient states. These efficiency metrics highlight the practical advantages of our approach for resource-constrained deployment scenarios and time-sensitive applications.
Table 6: Comparison of Training Time and GPU Memory, with fine-tuning for 10 epochs on the Pascal VOC2012 dataset.
<table border="1"><tr><td>Method</td><td>One epoch Time</td><td>Total Time</td><td>GPU Memory</td><td>Batch Size</td></tr><tr><td>FT</td><td>64.39 s</td><td>651.46 s</td><td>59.55 GB</td><td>32</td></tr><tr><td>Ours</td><td>43.25 s</td><td>43.25 s</td><td>51.61 GB</td><td>64</td></tr></table>
6 Conclusion
We presented CFSSeg, a novel method for class-incremental semantic segmentation (CSS) designed for both 2D images and 3D point clouds. CFSSeg distinguishes itself through a gradient-free, closed-form update mechanism, computed recursively for efficiency. This core component, combined with a frozen encoder for stability, high-dimensional feature mapping for plasticity, and a tailored pseudo-labeling strategy for semantic drift, allows the model to learn new classes exemplatively without catastrophic forgetting or reliance on stored exemplars. Consequently, CFSSeg operates with significantly reduced computational costrequiring only a single training pass per step-and enhanced data privacy. Our extensive evaluations onascal VOC2012, S3DIS, and ScanNet show that CFSSeg achieves outstanding results, outperforming prior methods and providing a robust, efficient, and effective solution for continual segmentation tasks, forming a complete closed loop from theoretical foundation to practical implementation.
254
CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds MM '25, October 27-31, 2025, Dublin, Ireland.
Acknowledgments
This research was supported by the National Natural Science Foundation of China (62306117), the Guangzhou Basic and Applied Basic Research Foundation (2024A04J3681), and GJYC program of Guangzhou (2024D03J0005).
References
[1] Iro Armeni, Ozan Sener, Amir Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer,
and Silvio Savarese. 2016. 3D Semantic Parsing of Large-Scale Indoor Spaces.
1534-1543. doi:10.1109/CVPR.2016.170
[2] Donghyeon Baek, Youngmin Oh, Sanghoon Lee, Junghyup Lee, and Bumsub Ham. 2022. Decomposed knowledge distillation for class-incremental semantic segmentation. *Advances in Neural Information Processing Systems* 35 (2022), 10380-10392.
[3] Fabio Cermelli, Massimiliano Mancini, Samuel Rota Bulo, Elisa Ricci, and Barbara Caputo. 2020. Modeling the background for incremental learning in semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 9233-9242.
[4] Sungmin Cha, YoungJoon Yoo, Taeup Moon, et al. 2021. Ssul: Semantic seg-
mentation with unknown label for exemplar-based class-incremental learning.
*Advances in neural information processing systems* 34 (2021), 10919-10930.
[5] Liang-Chieh Chen. 2014. Semantic image segmentation with deep convolutional nets and fully connected CRFs. *arXiv preprint arXiv:1412.7062* (2014).
[6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. 2017. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. *IEEE transactions on pattern analysis and machine intelligence* 40, 4 (2017), 834-848.
[7] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam.
2017. Rethinking Atrous Convolution for Semantic Image Segmentation.
arXiv:1706.05587 [cs.CV]
[8] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation. In *Proceedings of the European conference on computer vision (ECCV)*. 801-818.
[9] Xucheng Chen, Haifeng Luo, Tianqiang Huang, Hanxian He, and Wenyan Hu.
2025. Class-Incremental Semantic Segmentation for Mobile Laser Scanning Point
Clouds Using Feature Representation Preservation and Loss Cross-Coupling.
*Remote Sensing* 17, 3 (2025), 541.
[10] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. 2022. Masked-attention mask transformer for universal image segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 1290-1299.
[11] Bowen Cheng, Alex Schwing, and Alexander Kirillov. 2021. Per-pixel classification is not all you need for semantic segmentation. Advances in neural information processing systems 34 (2021), 17864-17875.
[12] Wei Cong, Yang Cong, Jiahua Dong, Gan Sun, and Henghui Ding. 2024. Gradient-Semantic Compensation for Incremental Semantic Segmentation. *IEEE Transactions on Multimedia* 26 (2024), 5561-5574. doi:10.1109/TMM.2023.3336243
[13] Thomas M. Cover. 1965. Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition. *IEEE Transactions on Electronic Computers* EC-14, 3 (1965), 326-334. doi:10.1109/PGEC.1965.264137
[14] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. 2017. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2432-2443. doi:10.1109/CVPR.2017.261
[15] Chaorui Deng, Qi Wu, Qingyao Wu, Fuyuan Hu, Fan Lyu, and Mingkui Tan.
2022. Visual Grounding Via Accumulated Attention. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 3 (2022), 1670-1684. doi:10.1109/TPAMI.2020.3023438
[16] Arthur Douillard, Yifu Chen, Arnaud Dapogny, and Matthieu Cord. 2021. Plop: Learning without forgetting for continual semantic segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 4040-4050.
[17] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. 2010. The Pascal Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision* 88, 2 (June 2010), 303-338.
[18] Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio.
2013. An empirical investigation of catastrophic forgetting in gradient-based
networks. *arXiv preprint arXiv:1312.6211* (2013).
[19] Dipam Goswami, René Schuster, Joost van de Weijer, and Didier Stricker.
2023. Attribution-aware weight transfer: A warm-start initialization for class-
incremental semantic segmentation. In *Proceedings of the IEEE/CVF Winter Con-
ference on Applications of Computer Vision.* 3195-3204.
[20] Ping Guo, Michael R Lyu, and NE Mastorakis. 2001. Pseudoinverse learning algorithm for feedforward neural networks. *Advances in Neural Networks and Neural Networks* (2001) Vol. 1, No. 3, pp. 100-110.
Applications 1, 321-326 (2001).
[21] Yong Guo, Mingkui Tan, Zeshui Deng, Jingdong Wang, Qi Chen, Jiezhang Cao, Yanwu Xu, and Jian Chen. 2024. Towards Lightweight Super-Resolution With Dual Regression Learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 46, 12 (2024), 8365-8379. doi:10.1109/TPAMI.2024.3406556
[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition.* 770-778.
[23] Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máte Lengyel.
2011. Bayesian Active Learning for Classification and Preference Learning.
arXiv:1112.5745 [stat.] https://arxiv.org/abs/1112.5745
[24] Tobias Kalb, Björn Mauthe, and Jürgen Beyerer. 2022. Improving replay-based continual semantic segmentation with smart data selection. In 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 1114-1121.
[25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.
2023. Segment anything. In *Proceedings of the IEEE/CVF international conference on computer vision*. 4015-4026.
[26] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnizeszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Ku-maran, and Raia Hadell. 2017. Overcoming Catastrophic Forgetting in Neural Networks. *Proceedings of the National Academy of Sciences* 114, 13 (2017), 3521-3526. arXiv:https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114 doi:10.1073/pnas.1611835114
[27] Marvin Klingner, Andreas Bär, Philipp Donn, and Tim Fingscheidt. 2020. Class-incremental learning for semantic segmentation re-using neither old data nor old labels. In 2020 IEEE 23rd international conference on intelligent transportation systems (ITSC). IEEE, 1-8.
[28] Zhizhong Li and Derek Hoiem. 2017. Learning without Forgetting. IEEE Trans- actions on Pattern Analysis and Machine Intelligence 40, 12 (2017), 2935-2947.
doi:10.1109/TPAMI.2017.2773081
[29] Zihan Lin, Zilei Wang, and Yixin Zhang. 2022. Continual semantic segmentation via structure preserving and projected feature alignment. In *European Conference on Computer Vision. Springer*, 345-361.
[30] Jiarun Liu, Hao Yang, Hong-Yu Zhou, Yan Xi, Lequan Yu, Cheng Li, Yong Liang,
Guangming Shi, Yizhou Yu, Shaoting Zhang, et al. 2024. Swin-umamba: Mamba-
based unet with imagenet-based pretraining. In *International Conference on Med-
ical Image Computing and Computer-Assisted Intervention*. Springer, 615-625.
[31] Mingyang Liu, Li Xiao, Huiqin Jiang, and Qing He. 2022. A new generative replay approach for incremental class learning of medical image for semantic segmentation. In *Proceedings of the 2022 International Conference on Intelligent Medicine and Health*. 51-56.
[32] Jun Ma, Feifei Li, and Bo Wang. 2024. U-mamba: Enhancing long-range de- pendency for biomedical image segmentation. *arXiv preprint arXiv:2401.04722* (2024).
[33] Xianping Ma, Xiaokang Zhang, and Man-On Pun. 2024. RS 3 Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation. *IEEE Geoscience and Remote Sensing Letters* (2024).
[34] Andrea Maracani, Umberto Michieli, Marco Toldo, and Pietro Zanuttigh. 2021. RECALL: Replay-Based Continual Learning in Semantic Segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 7026-7035.
[35] Umberto Michieli and Pietro Zanuttigh. 2019. Incremental Learning Techniques for Semantic Segmentation. In *2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)*. 3205-3212. doi:10.1109/ICCVW.2019.00400
[36] Umberto Michieli and Pietro Zanuttigh. 2021. Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 1114-1124.
[37] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yafo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. 2022. Efficient test-time model adaptation without forgetting. *International conference on machine learning*, PMLR, 16888-16905.
[38] Firat Ozdemir, Philipp Fuernstahl, and Orcun Goksel. 2018. Learn the new, keep the old: Extending pretrained models with new anatomy and images. In *Medical Image Computing and Computer Assisted Intervention-MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part IV 11. Springer, 361-369.
[39] Firat Ozdemir and Orcun Goksel. 2019. Extained pretrained segmentation networks with additional anatomical structures. *International journal of computer assisted radiology and surgery* 14 (2019), 1187-1195.
[40] Jooyoung Park and Irwin W Sandberg. 1991. Universal approximation using radial-basis-function networks. *Neural computation* 3, 2 (1991), 246-257.
[41] Liangzu Peng, Juan Elenter, Joshua Agterberg, Alejandro Ribeiro, and Rene Vidal.
2025. TVSD: Bridging Theory and Practice in Continual Learning with Pre-trained
**Models.** In *The Thirteenth International Conference on Learning Representations.*
https://openreview.net/forum?id=bqv7M0wc4x
255
MM '25, October 27-31, 2025, Dublin, Ireland. Jiaxu Li et al.
[42] Minh Hieu Phan, Son Lam Phung, Long Tran-Thanh, Abdesselam Bouzerdoum, et al. 2022. Class similarity weighted knowledge distillation for continual semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 16866-16875.
[43] Chao Qi, Jianqin Yin, Huaping Liu, and Jun Liu. 2021. Neighborhood Spatial Aggregation based Efficient Uncertainty Estimation for Point Cloud Semantic Segmentation. In **2021 IEEE International Conference on Robotics and Automation (ICRA).** 14025-14031. doi:10.1109/ICRA48506.2021.9560972
[44] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
[45] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. 2017. Pointnet++:
Deep hierarchical feature learning on point sets in a metric space. Advances in
neural information processing systems 30 (2017).
[46] Yiqiao Qiu, Yixing Shen, Zhuohao Sun, Yanchong Zheng, Xiaobin Chang, Weishi Zheng, and Ruixuan Wang. 2023. SATS: Self-attention transfer for continual semantic segmentation. *Pattern Recognition* 138 (2023), 109383. doi:10.1016/j.patcog.2023.109383
[47] Sylvvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. 2017. iCaRL: Incremental Classifier and Representation Learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR). 5533-5542. doi:10.1109/CVPR.2017.587
[48] Lianlei Shan, Weiqiang Wang, Ke Lv, and Bin Luo. 2022. Class-incremental semantic segmentation of aerial images via pixel-level feature generation and task-wise distillation. *IEEE Transactions on Geoscience and Remote Sensing* 60 (2022), 1-17.
[49] Vinicius MA Souza, Denis M dos Reis, Andre G Maletzke, and Gustavo EAPA Batista. 2020. Challenges in benchmarking stream learning algorithms with real-world data. *Data Mining and Knowledge Discovery* 34, 6 (2020), 1805-1858.
[50] Mingkui Tan, Peihao Chen, Hongyan Zhi, Jiajie Mai, Benjamin Rosman, Dongyu Ji, and Runhao Zeng. 2025. Source-Free Elastic Model Adaptation for Vision-and-Language Navigation. *IEEE Transactions on Multimedia* (2025), 1-13. doi:10.1109/TMM.2025.3535356
[51] Mingkui Tan, Zhuangwei Zhuang, Sitao Chen, Rong Li, Kui Jia, Qicheng Wang, and Yuanqing Li. 2024. EPMF: Efficient Perception-Aware Multi-Sensor Fusion for 3D Semantic Segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 46, 12 (2024), 8258-8273. doi:10.1109/TPAMI.2024.3402232
[52] Jonathan Tapson and André van Schaik. 2013. Learning the pseudoinverse solution to network weights. *Neural Networks* 45 (2013), 94-100.
[53] Quyen Tran, Tung Lam Tran, Khanh Doan, Toan Tran, Dinh Phung, Khoat Than, and Trung Le. 2025. Boosting Multiple Views for pretrained-based Continual Learning. In *The Thirteenth International Conference on Learning Representations* https://openreview.net/formum?id=AZR4R3lw7y
[54] Guankun Wang, Long Bai, Yanan Wu, Tong Chen, and Hongliang Ren. 2023. Rethinking exemplars for continual semantic segmentation in endoscopy scenes: Entropy-based mini-batch pseudo-replay. *Computers in Biology and Medicine* 165 (2023), 107412.
[55] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. 2024. A Comprehensive Survey of Continual Learning: Theory, Method and Application. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 46, 8 (2024), 5362-5383.
[56] Xi-Zhao Wang, Tianlun Zhang, and Ran Wang. 2019. Noniterative Deep Learning: Incorporating Restricted Boltzmann Machine Into Multilayer Random Weight Neural Networks. *IEEE Transactions on Systems, Man, and Cybernetics: Systems* 49, 7 (2019), 1299-1308. doi:10.1109/TSMC.2017.2701419
[57] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. 2019. Dynamic Graph CNN for Learning on Point Clouds. *ACM Trans. Graph.* 38, 5, Article 146 (Oct. 2019), 12 pages. doi:10.1145/3326362
[58] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli
Ouyang, Tong He, and Hengshuang Zhao. 2024. Point Transformer V3: Simpler
Faster Stronger. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR). 4840-4851.
[59] Jia-Wen Xiao, Chang-Bin Zhang, Jiekang Feng, Xialei Liu, Joost van de Weijer, and Ming-Ming Cheng. 2023. endpoints weight fusion for class incremental semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 7204-7213.
[60] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. 2021. SegFormer: Simple and efficient design for semantic segmentation with transformers. *Advances in neural information processing systems* 34 (2021), 12077-12090.
[61] Guanglei Yang, Enrico Fini, Dan Xu, Paolo Rota, Mingli Ding, Moin Nabi, Xavier Alameda-Pineda, and Elisa Ricci. 2022. Uncertainty-aware contrastive distillation for incremental semantic segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 45, 2 (2022), 2567-2581.
[62] Guanglei Yang, Enrico Fini, Dan Xu, Paolo Rota, Mingli Ding, Hao Tang, Xavier Alameda-Pineda, and Elisa Ricci. 2022. Continual attentive fusion for incremental learning in semantic segmentation. *IEEE Transactions on Multimedia* 25 (2022), 3841-3854.
[63] Yuwei Yang, Munawar Hayat, Zhao Jin, Chao Ren, and Yinjie Lei. 2023. Geometry and Uncertainty-Aware 3D Point Class-Incremental Semantic Segmentation. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR). 21759–21768. doi:10.1109/CVFPR52729.2023.02084
[64] Ze Yang, Ruibo Li, Evan Ling, Chi Zhang, Yiming Wang, Dezhao Huang, Keng Teck Ma, Minhoe Hur, and Guosheng Lin. 2023. Label-Guided Knowledge Distilation for Continual Semantic Segmentation on 2D Images and 3D Point Clouds. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 18601-18612.
[65] Jingwen Ye, Yifang Fu, Jie Song, Xingyi Yang, Songhua Liu, Xin Jin, Mingli Song, and Xinchao Wang. 2022. Learning with recoverable forgetting. In *European Conference on Computer Vision*. Springer, 87-103.
[66] Zhidong Yu, Wei Yang, Xike Xie, and Zhenbo Shi. 2024. TIKP: Text-to-Image Knowledge Preservation for Continual Semantic Segination. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 38. 16596-16604.
[67] Bo Yuan and Danpei Zhao. 2024. A Survey on Continual Semantic Segmentation: Theory, Challenge, Method and Application. IEEE Transactions on Pattern Analysis & Machine Intelligence 46, 12 (Dec. 2024), 10891-10910. doi:10.1109/TPAML2024.3446949
[68] Runhao Zeng, Wenbing Huang, Mingkui Tan, Yu Rong, Peilin Zhao, Junzhou Huang, and Chuang Gan. 2022. Graph Convolutional Module for Temporal Action Localization in Videos. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44, 10 (2022), 6209-6223. doi:10.1109/TPAMI.2021.3090167
[69] Chang-Bin Zhang, Jia-Wen Xiao, Xialei Liu, Ying-Cong Chen, and Ming-Ming Cheng. 2022. Representation compensation networks for continual semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.* 7053-7064.
[70] Yifan Zhang, Ying Wei, Qingyao Wu, Peilin Zhao, Shuaicheng Niu, Junzhou Huang, and Mingkui Tan. 2020. Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis. *IEEE Transactions on Image Processing* 29 (2020), 7834-7844. doi:10.1109/TIP.2020.3006377
[71] Lanyun Zhu, Tianrun Chen, Jianxiong Yin, Simon See, and Jun Liu. 2023. Continual semantic segmentation with automatic memory sample selection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 3082-3092.
[72] Huiping Zhuang, Zhenyu Weng, Run He, Zhiping Lin, and Ziqian Zeng. 2023. GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. 7746-7755. doi:10.1109/CVPR52729.2023.00748
256
Learning Temporal Point Processes via Reinforcement Learning

## 3 强化学习框架

假设我们感兴趣的是模拟每天的犯罪模式，或者病人每月的疾病发生率，那么这些数据就被收集为预定时间窗t内事件的轨迹。我们将观察到的路径视为专家（自然）采取的行动。
		设ζ={τ1，τ2，。…，τnζt}表示专家提供的事件的单个轨迹，其中nζt是事件总数，最多为t，并且对于不同的序列可以不同。然后，每个轨迹ζπe可以看作是从专家策略πe抽样的专家演示。因此，在较高的水平上，给出了一组专家演示d={ζ1，ζ2，。…，Z. J，.…|ζjπe}，将时间点过程拟合到d可以看作是寻找学习者策略πθ，该策略πθ可以生成另一组序列d∮={η1，η2，.…，εj，。…|ηjπθ}具有与d相似的模式。我们将在下面详细说明这个强化学习框架。

强化学习公式（rl）。给定一个过去事件序列st={ti}ti<t，随机策略πθ（a st）采样一个事件间时间a作为其动作，以生成下一个事件时间ti+1=ti+a。然后，提供一个奖励r（ti+1），状态st将更新为$\boldsymbol{s}_{t}=\left\{t_{1}, \ldots, t_{i}, t_{i+1}\right\}$。基本上，策略πθ（a|st）对应于时间点过程中下一个事件时间的条件概率，这反过来唯一地确定了相应的强度函数为$\lambda_{\theta}\left(t | \boldsymbol{s}_{t_{i}}\right)=\frac{\pi_{\theta}\left(t-t_{i} | \boldsymbol{s}_{t_{i}}\right)}{1-\int_{t_{i}}^{t} \pi_{\theta}\left(\tau-t_{i} | \boldsymbol{s}_{t_{i}}\right) d \tau}$。这就建立了时间点过程中的强度函数与强化学习中的随机策略之间的联系。如果给出了报酬函数r（t），则最优策略πθ可以通过
$$
\pi_{\theta}^{*}=\arg \max _{\pi_{\theta} \in \mathcal{G}} J\left(\pi_{\theta}\right) :=\mathbb{E}_{\eta \sim \pi_{\theta}}\left[\sum_{i=1}^{N_{T}^{\eta}} r\left(t_{i}\right)\right]
$$
其中g是所有候选策略的族πθ，η={T1，.…，tnηt}是策略πθ的一个抽样推出，nηt对于不同的推出样本可以不同。

**逆强化学习（irl）。**式（2）表明，当报酬函数给定时，可以通过最大化期望累积报酬来确定最优策略。然而，在我们的例子中，只有专家的事件序列可以被观察到，但是真正的报酬函数是未知的。给定专家策略πe，irl可以通过
$$
r^{*}=\max _{r \in \mathcal{F}}\left(\mathbb{E}_{\xi \sim \pi_{E}}\left[\sum_{i=1}^{N_{T}^{\xi}} r\left(\tau_{i}\right)\right]-\max _{\pi_{\theta} \in \mathcal{G}} \mathbb{E}_{\eta \sim \pi_{\theta}}\left[\sum_{i=1}^{N_{T}^{\eta}} r\left(t_{i}\right)\right]\right)
$$
其中f是奖励函数的族类，ζ={τ1，.…，τnζt}是由专家πe生成的一个事件序列，且η={t1，.…，tnηt}是学习者πθ的一个推出序列。该公式意味着一个适当的奖励函数应该给予专家策略比任何其他学习者策略更高的奖励，因此学习者可以通过最大化该奖励来接近专家绩效。相应地，将程序（2）和（3）表示为rl（r）和irl（πe）。最优策略可以通过

**拟议学习框架概述**.求解优化问题（3）是非常耗时的，因为它需要反复求解内环rl问题。我们通过选择r（t）的函数f的空间作为rkhs h中的单位球来缓解计算上的挑战，这使得我们能够在任何当前学习者策略π_（θ）下获得更新的奖励函数r_（t）的解析表达式。该r_（t）由有限样本专家轨迹和当前学习者政策的有限样本推出确定，它直接量化专家政策（或强度函数）和当前学习者政策（或强度函数）之间的差异。然后，通过解决一个简单的rl问题，如（2），学习者策略可以改进，以弥补其差距，专家策略使用一个简单的策略梯度类型的算法。

## 4.模型

在这一部分中，我们提出了模型参数化和最优报酬函数的解析表达式。

策略网络。策略πθ∈g的函数类应具有足够的灵活性和表达能力，以捕捉专家潜在的复杂点过程模式。因此，我们采用具有随机神经元的递归神经网络（rnn）[4]来灵活地捕捉非线性和长程的序列依赖结构。更具体地说，
$$
a_{i} \sim \pi\left(a | \Theta\left(h_{i-1}\right)\right), \quad h_{i}=\psi\left(V a_{i}+W h_{i-1}\right), \quad h_{0}=0
$$
其中隐藏状态hi∈r d编码过去事件的序列{t1, . . . , ti},$$
a_{i} \in \mathbb{R}^{+}, V \in \mathbb{R}^{d}
$$，$W \in \mathbb{R}^{d \times d}$。这里，Ψ是一个按元素应用的非线性激活函数，而_是从r d到概率分布π的参数空间的非线性映射。例如，可以选择$\psi(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$作为tanh函数，并设计Θ_的输出层，使得Θ(hii1)是概率密度函数π的有效参数。输出ai=ti i tii1用作第i个事件间时间（设t0=0），ai>0。

模型π的选择是非常灵活的，只有在随机变量为正的约束下，因为a总是正的。常见的分布，如指数分布和瑞利分布，将满足这种约束，各自地导致$\pi\left(a | \Theta\left(h_{i-1}\right)\right)=\Theta(h) e^{-\Theta(h) a}$ and $\pi\left(a | \Theta\left(h_{i-1}\right)\right)=\Theta(h) a e^{-\Theta(h) a^{2} / 2}$。这样，我们指定了一个非线性和灵活的依赖于历史

![1568209103712](F:\Machine-learning-and-data-science-notebook\images\LearningTemporalPointProcesses\1568209103712.png)

（5）中模型的架构如图2所示。与传统的rnn不同，输出ai是从π采样的，而不是通过确定性变换得到的。这就是“随机”政策的含义。随机抽样将允许策略探索临时空间。此外，采样的时间点将反馈给rnn。该模型的目的是捕捉到hi状态由两部分组成。一个是先前隐藏状态hii1的确定性影响，另一个是最新采样动作ai的随机影响。动作ai是从先前的分布π（a（hii1））中用参数_（hii1）采样的，并将被反馈以影响当前的隐藏状态hi。

在某种意义上，我们的随机神经元rnn模拟了双随机点过程的事件发生机制，如hawkes过程和自校正过程。对于这些类型的点过程，强度是随机的，这取决于历史，强度函数将控制下一个事件的发生率。

奖励功能类。报酬函数直接量化了πe和πθ之间的差异，指导了最优策略π*θ的学习。一方面，我们希望它的函数类r∈f具有足够的灵活性，以便它能够表示各种形状的报酬函数。另一方面，它应该具有足够的限制性，以便用有限样本有效地学习[3，13]。考虑到这些相互竞争的因素，我们选择F作为RKHS H，KRKH 61的单位球。这个函数类的一个直接好处是，我们可以证明最优策略可以通过定理1中给出的最小化公式直接学习，而不是原来的minimax公式（3）。

下面是一个证明的草图。简而言之，我们表示

$\underbrace{\phi(\eta) :=\int_{[0, T)} k(t, \cdot) d N_{t}^{(\eta)}}_{\text {feature mapping from data space to } \mathrm{R}}, \quad$ and $\quad \underbrace{\mu_{\pi_{\theta}} :=\mathbb{E}_{\eta \sim \pi_{\theta}}[\phi(\eta)]}_{\text {mean embeddings of the intensity function in RKHS }}$

同样，我们可以得到j（pe）=hr，bi-ih。从（3）中，r由
$$
\max _{\|r\|_{\mathcal{H}} \leq 1} \min _{\pi_{\theta} \in \mathcal{G}}\left\langle r, \mu_{\pi_{E}}-\mu_{\pi_{\theta}}\right\rangle_{\mathcal{H}}=\min _{\pi_{\theta} \in \mathcal{G}} \max _{\|r\|_{\mathcal{H}} \leq 1}\left\langle r, \mu_{\pi_{E}}-\mu_{\pi_{\theta}}\right\rangle_{\mathcal{H}}=\min _{\pi_{\theta} \in \mathcal{G}}\left\|\mu_{\pi_{E}}-\mu_{\pi_{\theta}}\right\|_{\mathcal{H}}
$$
其中第一等式由极大极小定理保证，并且
$$
r^{*}\left(\cdot | \pi_{E}, \pi_{\theta}\right)=\frac{\mu_{\pi_{E}}-\mu_{\pi_{\theta}}}{\left\|\mu_{\pi_{E}}-\mu_{\pi_{\theta}}\right\|_{\mathcal{H}}} \propto \mu_{\pi_{E}}-\mu_{\pi_{\theta}}
$$
可以通过数据进行经验评估。这样，我们将原来求解πθ的minimax公式化为一个简单的极小化问题，在实际应用中会更有效、更稳定。我们总结了定理1中的公式。

定理1将奖励函数族设为RKHS H中的单位球，即KRKH 6 1。这样，由(4)得到的最优策略也可以通过求解以下公式得到。
$$
\pi_{\theta}^{*}=\arg \min _{\pi_{\theta} \in \mathcal{G}} D\left(\pi_{E}, \pi_{\theta}, \mathcal{H}\right)
$$
其中D(πE，πθ，H)是πE与πθ之间的最大期望累积报酬差。
$$
D\left(\pi_{E}, \pi_{\theta}, \mathcal{H}\right) :=\max _{\|r\|_{\mathcal{H} \leqslant 1}}\left(\mathbb{E}_{\xi \sim \pi_{E}}\left[\sum_{i=1}^{N_{T}^{(\xi)}} r\left(\tau_{i}\right)\right]-\mathbb{E}_{\eta \sim \pi_{\theta}}\left[\sum_{i=1}^{N_{T}^{(\eta)}} r\left(t_{i}\right)\right]\right)
$$
定理1表明，我们可以将（4）的逆强化学习过程转化为一个简单的最小化问题，使πe和πθ之间的最大期望累积报酬差最小化。这使得我们能够避免由于重复求解内部rl问题而导致的（4）的昂贵计算。更有趣的是，我们可以导出由（6）给出的（8）的解析解。

有限样本估计。给定专家点过程的L轨迹和πθ产生的事件的M轨迹，平均嵌入量μπe和μπθ可通过其各自的经验平均值估计：
$$
\hat{\mu}_{\pi_{E}}=\frac{1}{L} \sum_{l=1}^{L} \sum_{i=1}^{N_{T}^{(l)}} k\left(\tau_{i}^{(l)}, \cdot\right) \text { and } \hat{\mu}_{\pi_{\theta}}=\frac{1}{M} \sum_{m=1}^{M} \sum_{i=1}^{N_{T}^{(m)}} k\left(t_{i}^{(m)}, \cdot\right)
$$
注：该经验估计值在τ（l）i和t（m）i处有偏。无偏估计也可以得到，并将提供在算法rlpp稍后讨论的简单性。

内核选择。RKHS中的单位球密集且富有表现力。从根本上讲，我们提出的框架和理论结果是通用的，可以直接应用于其他类型的核。例如，我们可以使用matérn核，它生成可微函数空间，称为sobolev空间[10，2]。在以后的实验中，我们使用了高斯核，取得了很好的结果。

## 5 学习算法

通过策略梯度学习。在实际应用中，由于平方是单调变换，我们可以等价地最小化D(πE，πθ，H)2，而不是像(7)中那样最小化D(πE，πθ，H)。现在，我们可以学习π∗ θ从RL公式(2)使用策略梯度和方差减少。首先，利用似然比技巧，可以计算∇θD(πE，πθ，H)2的梯度
$$
\nabla_{\theta} D\left(\pi_{E}, \pi_{\theta}, \mathcal{H}\right)^{2}=\mathbb{E}_{\eta \sim \pi_{\theta}}\left[\sum_{i=1}^{N_{T}^{\eta}}\left(\nabla_{\theta} \log \pi_{\theta}\left(a_{i} | \Theta\left(h_{i-1}\right)\right)\right) \cdot\left(\sum_{i=1}^{N_{T}^{\eta}} \hat{r}^{*}\left(t_{i}\right)\right)\right]
$$
其中$$
\sum_{i=1}^{N_{T}^{\eta}}\left(\nabla_{\theta} \log \pi_{\theta}\left(a_{i} | \Theta\left(h_{i-1}\right)\right)\right)
$$是滚动样本η={t1，.....，TNηT}使用学习策略πθ。

为了减少梯度的方差，我们可以利用未来行动不依赖于过去奖励的观察结果。这导致方差约化梯度估计θd（πe，πθ，h）2=eη_πθh pnηti=1（θlogπθ（a i（hii1））·pnηtl=i[710;r_（tl））bl]i，其中pnt l=i r_（tl）被称为“奖励去”和bl是进一步减少方差的基线。整个过程在rlpp算法中给出。在该算法中，在对当前策略中的m个轨迹进行采样后，用一个轨迹ηm进行评价，其余的m 1个样本估计奖励函数。在算法的不同阶段学习到的示例奖励函数也如图3所示。

与mle比较。在训练过程中，我们的生成模型直接将生成的时间事件与观察到的事件进行比较，从而迭代地纠正错误，有效地避免了模型的错误指定。由于训练只涉及策略梯度，因此它绕过了似然中对数生存项的棘手问题（式（1））。另一方面，由于所学习的策略实际上是一个点过程的条件密度，因此我们的方法仍然类似于rl重新定义中的mle形式，因此可以用统计原理的方式来解释。

与GAN和GAIL相比。根据定理1，我们的策略是通过最小化πe和πθ之间的差异来直接学习的，πθ具有一个闭式表达式。因此，我们将原irl问题转化为一个关于策略只有一组参数的最小化问题。
在每次有策略梯度的训练迭代中，我们都有一个梯度的无偏估计，估计的回报函数也依赖于当前的策略πθ。相反，在gan或gail公式中，它们有两组与生成器和鉴别器相关的参数。梯度估计是有偏的，因为每一个最小/最大问题实际上是非凸的，不能一次性解决。因此，我们的框架比学习点过程的mini-max公式更稳定和有效。


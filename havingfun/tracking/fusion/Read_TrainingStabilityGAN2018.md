# Spectral Normalization for Generative Adversarial Networks  
--------------------------
omit consecutively nutshell consecutively reveal density variational persisting 
--------------------------
Propose a novel weight normalization tech called spectral normalization to stablize the training of the discriminator.
* GAN have been enjoying considerable success as a framework of generative models in recent years.
* A persisting challenge in the training of GAN is the performance control of the discriminator.

* Lipschiz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the
  only hyper-parameter for satisfactory performance.  
* Implementation is simple and the additional computational cost is small.

## Theoretical groundwork  
Assume there is a simple discriminator made of a neural nework  
$f(x, \theta) = W^{L + 1}_{aL} (W^L(a_{L - 1}(W^{L - 1}(\cdots a_1 (W^1 x) \cdots))))$,  
with the input $x$. Where $\theta = {W^1, \cdots, W^L, W^{L + 1}}$ is the learning parameters set.
$a_l$ is an element-wise non-linear activation function.  
Omit the bias term of each layer for simplicity. The final output of the discriminator is given by  
$D(x, \theta) = \mathit{A}(f(x, \theta))$  
where $\mathit{A}$ is an activation function corresponding to the divergence of distance measure of the user's choice.  
The standard formulation of GAN is given by $\min_G \max_D V(G, D)$

* The function space from which the discriminators are selected crucially affects the performance of GANs.
* It is advocated the importance of Lipschitz continuity in assuring the boundness of statistics.

To control the Lipschize constant of the discriminator by:  
* Adding regularization term defined on input examples $x$.  
* Search for the discriminator $D$ from the set of $K$-Lipschiz continuous functions  
* Allow for relatively easy formulations based on samples, they also suffer from the fact that, they cannot impose
  regularization on the space outside of the supports of the generator and data distributions without introducing
  somewhat heuristic means.

Spectral normalization skirt this issue by normalizing the weight matrics.  
Controls the Lipschiz constant of the discriminator function $f$ by literally constraining the spectral norm of each
layer $g$.  
Lipschiz norm  
$||g||_{Lip} = \sup_h \sigma(\triangledown g(h))$  
where $\sigma(A)$ is the spectral norm of the matrix A ($L_2$ matrix norm of A).  
$\bar W_{SN}(W) := W / \sigma(W)$

## Experiment
Conduct a set of experiments of unsupervised image generation on CIFA-10 and STL-10.  
Compared against other normalization techniques

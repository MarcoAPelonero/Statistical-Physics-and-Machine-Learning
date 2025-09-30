## Introduction to GD and SDG

We define a function $\hat y(\vec{x};\vec{\theta})$, where
$\vec{x}$ is a set of variables that the function depends on and $\vec{\theta}$ a set of **parameters**. Now, given a set of N data point that we call $(x_i, y_i)$, but leaving out the parameters as variables for the loss, we can define the L2 regularized **Mean Square Error** as 

$$
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left(\hat y (x_i; \theta) - y_i\right)^2 + \frac{\lambda}{2}\|\theta\|_2^2
$$

Now when we approach the gradient the gradient is specific to the function (there is a way of generalizing), but this time out function has a specific form, it's a polinomial of the kind $\hat y (x_i; \theta) = \sum_{k=0}^d\theta_kx^k$ so, when we perform the gradient over the various $\theta_k$, we get the following:

$$
    \frac{\partial\mathcal{L}(\theta)}{\partial \theta_k} = \frac{2}{M} \sum_{i\in \mathcal{B}} \left(\hat y_i - y_i\right) x_i^k + {\lambda}\theta_k = \nabla_{\theta}\mathcal{L}
$$

With M being the **batch size**. If we start from a random set of parameters theta to compute our set of $\hat y_i$, and we aim for the set of weights that minized the MSE, we end up with the recurrent update rule:

$$
    \theta_{t+1} = \theta_{t}-\eta\nabla_{\theta}\mathcal{L}
$$

Now we can specify that usually $\eta$ is the **learning rate**.
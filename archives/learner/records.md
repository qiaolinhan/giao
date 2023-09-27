# giao
 This repo is a test as learning how to use git with VScode.

 * PyTorch provides methods to create random or zero-filled tensors (which we use to create weights or biases for a simple linear model).
   * For the **weights**, set `requires_grad` **after** the initialization (we do not want that step included in the gradient).
   * A tailing `_` in PyTorch signifies that the operation is performed in-place.

---
## 20210107
 Implement `negative log-likelihood` to use as the loss function.  
 ```
 nll = -input[range(target.shape[0], target)]
 nll = nll.mean()
 ```
For each iteration:
* Select  mini-batch of data of size `bs`
* Use the model to make `preds`
* Calculate the `loss` on the batch
* Use `loss.backward()` to update the `grad` of the `model`
* Update the prameters of the model using the `grad`

```python
epochs = 2
lr = 1e-4

def fit():
  for epoch in range(epochs):
      for ii in range((n-1)//bs+1):
          start, end = ii * bs, ii * bs +bs
          xb, yb = x_train[start:end], y_train[start:end]

          pred = model(xb)
          loss = loss_func(pred, yb)

          loss.backward()
          with torch.no_gard():
              weights -= weights.grad * lr
              biases -= biases.grad * lr
              weights.grad.zero_()
              biases.grad.zero_() 
fit()   
```
### Refactor using `torch.nn.functional`
code refactoring: 代码重构  
concise： 简洁的  
attributes: 属性  
### Refactor using `nn.Module`
Use `nn.Module` and `nn.Parameter` for clearer and more concise training loop  
* `nn.Module`: we want to create a class  that holds our weights, biases, and method for the forward step.

### Refactor using `nn.Linear`
$ y = x * A^T + b$

## 2021-11-5
### Celluar Automata (CA)
CA are (typically) spatially and temporally discrete. Cellls evolve in parallel at discrete time step  
A CA consists of:
* A grid cell
* A set of states each cell can be in, e.g. burning, (heating,) not burning.
* A set of rules to determine the evolution of the system based on the states each cell is in.

from kan import *
import torch

model = KAN(width=[2,5,11,1], grid=5, k=3, seed=0)

f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
print(dataset['train_input'].shape)
print(dataset['train_label'].shape)

model(dataset['train_input'])
model.plot(beta=100)

model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
# model.plot()

model.prune()
model.plot(mask=True)

model = model.prune()
model(dataset['train_input'])
# model.plot()

model.train(dataset, opt="LBFGS", steps=50)
model.plot()

mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin')
    model.fix_symbolic(0,1,0,'x^2')
    model.fix_symbolic(1,0,0,'exp')
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

model.train(dataset, opt="LBFGS", steps=50)
print(model.symbolic_formula()[0][0])
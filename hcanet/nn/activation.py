# this module was created to prevent circular dependencies

import torch as th
import torch.nn.functional as F

activation_functions = {
   "sigmoid": th.sigmoid,
   "tanh": th.tanh,
   "relu": th.relu,
   "elu": F.elu,
   "leakyrelu": F.leaky_relu,
   "selu": F.selu, }


def get_activation(funcname: str):
   if funcname not in activation_functions:
      raise ValueError("Invalid activation function {}".format(funcname))
   return activation_functions[funcname]

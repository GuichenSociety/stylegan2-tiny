
from torchsummary import summary

from model import *
from train import main

main_ = main()
w = main_.get_w(5)
n = main_.get_noise(5)

gen = main_.generator
s = gen(w,n)
print(s.shape)
import mlx.core as mx
import mlx.nn as nn
import math

# Lora实现，封装linear，替换到父module里
class LoraLayer(nn.Module):
    def __init__(self,raw_linear,in_features,out_features,r,alpha):
        super().__init__()
        self.r=r
        self.alpha=alpha
        scale = 1 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(in_features, r),
        )
        self.lora_b = mx.zeros(shape=(r, out_features))

        self.raw_linear=raw_linear

    def forward(self,x):    # x:(batch_size,in_features)
        raw_output=self.raw_linear(x)
        lora_output=x@((self.lora_a@self.lora_b)*self.alpha/self.r)    # matmul(x,matmul(lora_a,lora_b)*alpha/r)
        return raw_output+lora_output

def inject_lora(model,name,layer):
    name_cols=name.split('.')

    children=name_cols[:-1]
    # print("====================================================================")
    # print(children)
    cur_layer=model
    # print(cur_layer)
    for i in range(0, len(children)-1, 2):
        child=children[i]
        cur_layer=getattr(cur_layer,child)[int(children[i+1])]

    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer=LoraLayer(layer,layer.weight.shape[1],layer.weight.shape[0],8,1)
    # print(lora_layer)
    # print(cur_layer)
    setattr(cur_layer,name_cols[-1],lora_layer)
import torch
import torch.nn.functional as F
import  torch.nn as nn

x = torch.rand(1,3,4,4)
net = nn.Conv2d(3,3,1)
net.eval()
out1= net(x).mean(dim=-1).mean(dim=-1)
out2 =net(x.mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True))
print(out1)
print(out2)



def main(kernel):
    base = torch.tensor([[11, 12,13, 14, 15, 16, 17, 18, 19 ]])
    matrix = torch.empty(9,9)
    for i in range(0,9):
        item = i*10+base
        matrix[i,:] = item
    print(f"Input:\n{matrix}")
    print(f"Kernel:\n{kernel}")
    matrix = matrix.view(1, 1, 9, 9)
    kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
    kernel_size = kernel.size()[-1]
    output = F.conv2d(matrix, kernel, stride=1, padding=(kernel_size-1)//2 )
    print(f"Output:\n{output}")







if __name__ =="__main__":
    kernel = torch.tensor([[0., 0., 0.],
                           [1., 0., 0.],
                           [0., 0., 0.]
                           ])

    # kernel = torch.tensor([[0., 0., 0., 0., 0.],
    #                        [0., 0., 0., 0., 0.],
    #                        [0., 0., 0., 0., 0.],
    #                        [0., 0., 0., 1., 0.],
    #                        [0., 0., 0., 0., 0.],
    #                        ])

    # kernel = torch.tensor([[0., 0., 0., 0., 0., 0., 0,],
    #                        [1., 0., 0., 0., 0., 0., 0,],
    #                        [0., 0., 0., 0., 0., 0., 0,],
    #                        [0., 0., 0., 0., 0., 0., 0,],
    #                        [0., 0., 0., 0., 0., 0., 0,],
    #                        [0., 0., 0., 0., 0., 0., 0,],
    #                        [0., 0., 0., 0., 0., 0., 0,]
    #                        ])

    main(kernel)

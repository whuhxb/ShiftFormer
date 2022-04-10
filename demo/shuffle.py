import torch
import torch.nn.functional as F


def main(size):
    # base = torch.tensor([[11, 12,13, 14, 15, 16, 17, 18, 19 ]])
    base = (torch.arange(0,size)+11).unsqueeze(dim=0)
    matrix = torch.empty(size,size)
    for i in range(0,size):
        item = i*10+base
        matrix[i, :] = item
    print(f"Input:\n{matrix}")
    print(matrix.shape)

    W,H = matrix.shape
    W,H = 2*(W//2), 2*(H//2)
    new_matrix = torch.empty(matrix.shape)
    print(matrix[:, 0:W:2])
    print(matrix[:, 1:W:2])








if __name__ =="__main__":
    size=7

    main(size)

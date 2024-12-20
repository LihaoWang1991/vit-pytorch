import torch
from vit_pytorch import ViT

def test():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    n_p = sum(x.numel() for x in v.parameters())
    print(f"Total number of parameters: {n_p}")

    img = torch.randn(2, 3, 256, 256)

    preds = v(img)
    print(preds.shape)
    # assert preds.shape == (1, 1000), 'correct logits outputted'

test()

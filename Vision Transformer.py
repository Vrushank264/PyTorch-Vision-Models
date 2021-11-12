import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    
    """
        Splits Image into patches and embeds them

        Parameters
        ----------
        img_size : int
            Size of the input image.
            
        patch_size : int
            Size of the the patch.
            
        emb_dim : int, optional
            Embedding Dimensions. The default is 768.

        Returns
        -------
        torch.Tensor of Shape(num_samples, num_patches, emb_dim).

        """
    
    def __init__(self, img_size, patch_size, emb_dim = 768):
        
        
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels = 3, out_channels = emb_dim, 
                                    kernel_size = patch_size, stride = patch_size)
        
    def forward(self, x):
        
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        
        return x
    

class SelfAttention(nn.Module):
    
    """
        Implementation of Self-Attention mechanism.

        Parameters
        ----------
        dim : int
            The input and output dimensions of tokens.
            
        num_heads : int, optional
            Number of attention heads. The default is 12.
            
        bias : bool, optional
            If True then includes bias in the qkv projections. The default is True.
            
        attn_dropout_p : float, optional
            Dropout Probablility applied to qkv tensors. The default is 0..
            
        proj_dropout_p : float, optional
            Dropout probability applied to output tensor. The default is 0..

        Returns
        -------
        torch.Tensor of shape(num_samples, num_patches + 1, dim)

    """
    
    def __init__(self, dim, num_heads = 12, bias = True, attn_dropout_p = 0., proj_dropout_p = 0.):
       
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias = bias)
        self.attn_dropout = nn.Dropout(attn_dropout_p)
        self.projection = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout_p)
        
    def forward(self, x):
        
        num_samples, num_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
            
        qkv = self.qkv(x)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        key_t = k.transpose(-2,-1)
        dot_prod = (q @ key_t) * self.scale
        
        attn = dot_prod.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        self_attn = attn @ v
        self_attn = self_attn.transpose(1,2).flatten(2)
        
        x = self.projection(self_attn)
        x = self.proj_dropout(x)
        
        return x
    

class MLP(nn.Module):
    
    """
        Multilayer Perceptron.
        
        Linear -> GELU -> Dropout -> Linear -> Dropout

        Parameters
        ----------
        in_c : int
            Number of input channels.
            
        hidden_c : int
            Number of hidden channels.
            
        out_c : int
            Number of output channels.
            
        dropout_p : float, optional
            Dropout probablity. The default is 0.0.

        Returns
        -------
        torch.Tensor of shape(num_samples, num_patches + 1, out_c)

    """
    
    def __init__(self, in_c, hidden_c, out_c, dropout_p = 0.0):
        
        
        super().__init__()
        self.fc1 = nn.Linear(in_c, hidden_c)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_c, out_c)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
    
    
class TransformerBlock(nn.Module):
    
    """
        Transformer Block. 
        Input -> LayerNorm -> Self-Attention --> LayerNorm -> MLP -> Output
              '                              +'                   +
              '------------------------------''-------------------'
              
        Parameters
        ----------
        dim : int
            Embedding Dimension.
            
        num_heads : int
            Number of Transformer Blocks.
            
        ratio : int, optional
            Determines the hidden dimensions/channels of MLP modules. The default is 4.0.
            
        bias : bool, optional
            If True then includes bias to qkv. The default is True.
            
        dropout_p : float, optional
            Dropout probablity . The default is 0.0.
            
        attn_dropout_p : float, optional
            Dropout probability for Self-Attention Module. The default is 0.0.

        Returns
        -------
        torch.Tensor of shape(num_samples, num_patches + 1, dim)

        """
    
    def __init__(self, dim, num_heads, ratio = 4.0, bias = True, dropout_p = 0.0, attn_dropout_p = 0.0):
        
        
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = SelfAttention(dim, num_heads = num_heads, bias = bias)
        hidden_c = int(dim * ratio)
        self.mlp = MLP(dim, hidden_c, dim)
        
    def forward(self, x):
        
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x
    
    
class VisionTransformer(nn.Module):
    
    """
        Implementation of Vision Transformer(ViT).

        Parameters
        ----------
        img_size : int, optional
            Size of the input image. The default is 384.
            
        patch_size : int, optional
            Size of the patches. The default is 16.
            
        num_classes : int, optional
            Total Number of classes in training data. The default is 1000.
            
        emb_dim : int, optional
            Embedding dimension. The default is 768.
            
        depth : int, optional
            Total number of Transformer blocks. The default is 12.
            
        num_heads : iny, optional
            Number of attention blocks. The default is 12.
            
        ratio : float, optional
            determines hidden channels of MLP. The default is 4..
            
        bias : bool, optional
            If True then includes bias to qkv. The default is True.
            
        dropout_p : float, optional
            Dropout Probability. The default is 0.0.
            
        attn_dropout_p : float, optional
            Dropout probability for attention. The default is 0.0.

        Returns
        -------
        torch.Tensor of shape(num_samples, num_classes)

        """
    
    def __init__(self, img_size = 384, patch_size = 16, num_classes = 1000, emb_dim = 768, depth = 12,
                 num_heads = 12, ratio = 4., bias = True, dropout_p = 0.0, attn_dropout_p = 0.0):
        
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, emb_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, emb_dim))
        self.pos_dropout = nn.Dropout(dropout_p)
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        self.ViTBlocks = nn.ModuleList([TransformerBlock(dim = emb_dim,
                                                         num_heads = num_heads)
                                        for _ in range(depth)
                                        ])
        
        self.norm = nn.LayerNorm(emb_dim, eps = 1e-6)
        self.cls_head = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        
        num_samples = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_tokens.expand(num_samples, -1, -1)
        x = torch.cat([cls_tokens, x], dim = 1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        
        for block in self.ViTBlocks:
            x = block(x)
            
        x = self.norm(x)
        cls_final_token = x[:, 0]
        x = self.cls_head(cls_final_token)
        
        return x
        

def test():

    vit = VisionTransformer()
    x = torch.randn((1, 3, 384,384))
    out = vit(x)
    print(out.shape)
    print(vit)


if __name__ == '__main__':

    test()    

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class MoCoLightning(pl.LightningModule):
    """
    PyTorch Lightning 版本的 MoCo 实现
    """
    def __init__(self, base_encoder, dim=256, K=3*256, m=0.999, T=0.07, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['base_encoder'])
        
        # MoCo 的主要参数
        self.K = K
        self.m = m
        self.T = T
        self.lr = lr

        # 创建编码器
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        # 初始化动量编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q):
        """推理时只使用查询编码器"""
        embedding, q = self.encoder_q(im_q)
        return embedding

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        im_q, im_k = batch
        
        # 计算查询特征
        embedding, q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # 计算键特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # 计算 logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # 标签: 正样本指示器
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        # 记录损失
        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.encoder_q.parameters(), lr=self.lr)
        return optimizer 
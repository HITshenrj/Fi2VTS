import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def dot_product_attention(query, key, value):
    attn_weights = torch.matmul(query, key.transpose(-2, -1))
    attn_weights = F.softmax(attn_weights, dim=-1)
    attended_values = torch.matmul(attn_weights, value)

    return attended_values


def STFT_for_Period(x, stft_window_len_list, k=10):
    N = x.shape[2]
    stft_results = []
    for stft_window_len in stft_window_len_list:
        xf_n_list = []
        for n in range(N):
            xf_n = torch.stft(
                input=x[:, :, n], n_fft=stft_window_len, return_complex=True)
            frequency_list = abs(xf_n).mean(0).mean(-1)
            _, top_list = torch.topk(frequency_list, k)
            top_list = top_list.detach().cpu().numpy()
            xf_n_list.append(torch.mean(xf_n[:,top_list,:], dim=2))
        xf_n_list_torch = torch.stack(xf_n_list, dim=-1)

        stft_results.append(xf_n_list_torch)
    return stft_results


class Fi2VBlock(nn.Module):
    def __init__(self, configs):
        super(Fi2VBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.stft_window_len_list = configs.stft_window
        self.max_stft_window_len = configs.pred_len + configs.seq_len
        self.mlp_list = nn.ModuleList([nn.Linear(self.k*2, self.max_stft_window_len)
                                    for stft_window_len in self.stft_window_len_list])
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        
        self.MultiheadAttention_real = nn.ModuleList([nn.MultiheadAttention(self.k, num_heads=2)
                                    for _ in self.stft_window_len_list])
        self.MultiheadAttention_img = nn.ModuleList([nn.MultiheadAttention(self.k, num_heads=2)
                                    for _ in self.stft_window_len_list])

    def forward(self, x):
        stft_results = STFT_for_Period(
            x, self.stft_window_len_list, self.k)

        attn_weights_out_list=[]
        for i in range(len(self.stft_window_len_list)):
            stft_result = stft_results[i].permute(0, 2, 1)
            attended_values1, _ = self.MultiheadAttention_real[i](stft_result.real,stft_result.imag,stft_result.real)
            attended_values2, _ = self.MultiheadAttention_img[i](stft_result.real,stft_result.imag,stft_result.imag)

            attn_weights = torch.concat([torch.pow(
                attended_values1.permute(0, 2, 1), 2),torch.pow(attended_values2.permute(0, 2, 1), 2)], dim=1)

            attn_weights_out = self.mlp_list[i](attn_weights.permute(0, 2, 1))

            attn_weights_out_list.append(attn_weights_out)

        attn_weights_out_list_torch = torch.stack(attn_weights_out_list, dim=-1)

        conv_dec=torch.mean(self.conv(attn_weights_out_list_torch), dim=-1).permute(0, 2, 1)

        # residual connection
        res = conv_dec + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([Fi2VBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # Fi2VBlocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

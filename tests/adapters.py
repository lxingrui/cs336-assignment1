from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import einops
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.BPE.decode_encode_bpe import Tokenizer
from cs336_basics.BPE.train_bpe import *
from cs336_basics.nerual_net.Embedding import Embedding
from cs336_basics.nerual_net.Linear import Linear
from cs336_basics.nerual_net.RMSNorm import RMSNorm
from cs336_basics.nerual_net.RoPE import RoPE
from cs336_basics.nerual_net.SwiGLU import SwiGLU


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    model = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    state_dict = {"W": weights}
    model.load_state_dict(state_dict)
    return model(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    state_dict = {"W": weights}
    embedding.load_state_dict(state_dict)
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    weights_dict = {"w1": w1_weight, "w2": w2_weight, "w3": w3_weight}
    swi = SwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    swi.load_state_dict(weights_dict)

    return swi(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scores = (Q @ einops.rearrange(K, "... k q -> ... q k")) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    softmax = run_softmax(scores, -1)
    return softmax @ V
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    casual_mask = torch.tril(torch.ones(in_features.size(-2), in_features.size(-2), device=in_features.device)).bool()
    in_q = in_features @ q_proj_weight.T
    in_k = in_features @ k_proj_weight.T
    in_v = in_features @ v_proj_weight.T
    in_q_head = einops.rearrange(in_q, "... seq (head dim) -> ... head seq dim", head=num_heads)
    in_k_head = einops.rearrange(in_k, "... seq (head dim) -> ... head seq dim", head=num_heads)
    in_v_head = einops.rearrange(in_v, "... seq (head dim) -> ... head seq dim", head=num_heads)
    self_attn_head = run_scaled_dot_product_attention(in_q_head, in_k_head, in_v_head, casual_mask)
    self_attn = einops.rearrange(self_attn_head, "... head seq dim -> ... seq (head dim)")
    return self_attn @ o_proj_weight.T
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    device = in_features.device
    seq_len = in_features.size(-2)
    # --- 新增：处理 token_positions 为 None 的情况 ---
    if token_positions is None:
        # 1. 生成 0 到 seq_len - 1 的基础一维位置张量
        base_positions = torch.arange(seq_len, device=device)
        # 2. 动态提取前置的 Batch 维度，并使用 expand 扩展形状
        batch_shape = in_features.shape[:-2]
        token_positions = base_positions.expand(*batch_shape, seq_len)
    # ------------------------------------------------
    rope = RoPE(theta, d_model // num_heads, max_seq_len, device=in_features.device)
    casual_mask = torch.tril(torch.ones(in_features.size(-2), in_features.size(-2), device=in_features.device)).bool()
    in_q = in_features @ q_proj_weight.T
    in_k = in_features @ k_proj_weight.T
    in_v = in_features @ v_proj_weight.T
    in_q_head = einops.rearrange(in_q, "... seq (head dim) -> ... head seq dim", head=num_heads)
    in_k_head = einops.rearrange(in_k, "... seq (head dim) -> ... head seq dim", head=num_heads)
    in_v_head = einops.rearrange(in_v, "... seq (head dim) -> ... head seq dim", head=num_heads)
    self_attn_head = run_scaled_dot_product_attention(
        rope(in_q_head, einops.rearrange(token_positions, "... sequence_length -> ... 1 sequence_length")),
        rope(in_k_head, einops.rearrange(token_positions, "... sequence_length -> ... 1 sequence_length")),
        in_v_head,
        casual_mask,
    )
    self_attn = einops.rearrange(self_attn_head, "... head seq dim -> ... seq (head dim)")
    return self_attn @ o_proj_weight.T
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = RoPE(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    ln1_weight = {"W": weights["ln1.weight"]}
    ln2_weight = {"W": weights["ln2.weight"]}
    rms1 = RMSNorm(d_model, device=in_features.device)
    rms1.load_state_dict(ln1_weight)
    rms2 = RMSNorm(d_model, device=in_features.device)
    rms2.load_state_dict(ln2_weight)
    ffn_weight = {"w1": weights["ffn.w1.weight"], "w2": weights["ffn.w2.weight"], "w3": weights["ffn.w3.weight"]}
    swiglu = SwiGLU(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    swiglu.load_state_dict(ffn_weight)
    attn = run_multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        rms1(in_features),
    )
    residual_attn = in_features + attn
    ffn = swiglu(rms2(residual_attn))
    return ffn + residual_attn
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    device = in_indices.device
    # 从权重中获取正确的浮点精度（例如 float32 或 float16），避免误用 int64
    weight_dtype = weights["token_embeddings.weight"].dtype

    # 1. Token Embedding
    # 注意：这里将第一个参数修正为 vocab_size，防止 Token ID 越界
    embedd = Embedding(vocab_size, d_model, device=device, dtype=weight_dtype)
    embedd.load_state_dict({"W": weights["token_embeddings.weight"]})
    x = embedd(in_indices)  # 输出形状: (batch_size, sequence_length, d_model)

    # 2. 循环计算每一层 Transformer Block（这里使用 for 循环是标准且正确的）
    for l in range(num_layers):
        # 2.1 动态提取当前层 weights（过滤并去掉 "layers.{l}." 前缀）
        # 例如将 "layers.0.ln1.weight" 转化为 "ln1.weight" 传给 run_transformer_block
        prefix = f"layers.{l}."
        layer_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}

        # 2.2 将数据送入当前 Transformer 块计算
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    # 3. Final RMSNorm (对应 ln_final.weight)
    ln_final = RMSNorm(d_model, device=device, dtype=weight_dtype)
    ln_final.load_state_dict({"W": weights["ln_final.weight"]})
    x = ln_final(x)

    # 4. Language Model Head (对应 lm_head.weight)
    # lm_head.weight 形状为 (vocab_size, d_model)
    # 通过矩阵乘法将维度从 d_model 映射到 vocab_size 维度
    logits = x @ weights["lm_head.weight"].T

    return logits
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    rms = RMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    dict_w = {"W": weights}
    rms.load_state_dict(dict_w)
    return rms(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x_max = torch.max(in_features, dim=dim, keepdim=True)
    x_max_reduced = in_features - x_max[0]
    x_max_reduced_exp = torch.exp(x_max_reduced)
    x_sum = x_max_reduced_exp.sum(dim=dim, keepdim=True)
    return x_max_reduced_exp / x_sum


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # 1. 找到每行的最大值，用于数值稳定（防止 exp 溢出）
    x_max = torch.max(inputs, dim=1, keepdim=True)[0]
    x_stable = inputs - x_max

    # 2. 计算分母的 log-sum-exp: log(sum(exp(x_stable)))
    # 对应公式中的 log(sum(e^(x_i - x_max)))
    log_sum_exp = torch.log(torch.exp(x_stable).sum(dim=1, keepdim=True))

    # 3. 计算 log_softmax: log(exp(x_i)/sum(exp(x))) = x_stable - log_sum_exp
    log_softmax = x_stable - log_sum_exp

    # 4. 根据 targets 提取对应真实类别的 log 概率
    # 使用高级索引（Advanced Indexing）从每一行取出对应 target 列的值
    batch_size = inputs.shape[0]
    target_log_probs = log_softmax[torch.arange(batch_size), targets]

    # 5. 计算负对数似然损失（Negative Log Likelihood），并在 batch 上求平均
    loss = -target_log_probs.mean()

    return loss
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    # 实例化并返回 Tokenizer 对象
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 读文件和初始化词表
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    vocab = init_bpe(vocab_size, special_tokens)

    # 2. 拿到初始的词频。此时 key 还是 tuple[bytes, ...] 格式
    raw_word_freqs = get_freq_counts(text, special_tokens)

    # 3. 将单词转化为 Token ID 列表 (例如 [100, 111, 103])，以提速哈希与对比
    # words: dict[int, list[int]]  ->  word_id -> [token_id, token_id, ...]
    # word_freqs: dict[int, int]   ->  word_id -> 词频频次
    words = {}
    word_freqs = {}

    for wid, (word_tuple, freq) in enumerate(raw_word_freqs.items()):
        # 由于单字节的 ID 就是其自身的字节数值，所以直接用整数表示
        words[wid] = [b[0] for b in word_tuple]
        word_freqs[wid] = freq

    # 4. 构建数据结构
    # pair_frequencies: dict[tuple[int, int], int]  ->  统计当前所有相邻 (ID1, ID2) 的总频次
    # pair_to_words: dict[tuple[int, int], set[int]] ->  倒排索引，记录某个 pair 出现在哪些 word_id 中
    pair_frequencies = defaultdict(int)
    pair_to_words = defaultdict(set)

    for wid, tokens in words.items():
        freq = word_freqs[wid]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_frequencies[pair] += freq
            pair_to_words[pair].add(wid)

    merges = []
    num_merges = vocab_size - len(vocab)

    # 5. BPE 增量更新核心合并循环
    for _ in range(num_merges):
        if not pair_frequencies:
            break

        # 5.1 选出频次最高且满足 tie-breaking 规则的 pair
        # 由于 pair 此时是 (int, int) 类型的 ID，我们需要查 vocab[ID] 得到对应的 bytes 进行字典序比较
        max_freq = max(pair_frequencies.values())
        candidates = [p for p, freq in pair_frequencies.items() if freq == max_freq]

        if len(candidates) == 1:
            best_pair = candidates[0]
        else:
            best_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 5.2 倒排索引魔法：我们只获取包含了 best_pair 的那些 word_id
        # 完全不需要扫描那些不包含该 pair 的词！
        affected_word_ids = list(pair_to_words.get(best_pair, set()))

        for wid in affected_word_ids:
            tokens = words[wid]
            freq = word_freqs[wid]

            # 步骤 A：从全局频次表和倒排索引中，扣除这个单词“原本包含的所有 pair”贡献
            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                pair_frequencies[p] -= freq
                if pair_frequencies[p] <= 0:
                    del pair_frequencies[p]
                pair_to_words[p].discard(wid)

            # 步骤 B：在这个单词内执行局部合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            words[wid] = new_tokens

            # 步骤 C：将合并后新单词产生的 pair 频次重新加回到全局频次表，并更新倒排索引
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                pair_frequencies[p] += freq
                pair_to_words[p].add(wid)

    return vocab, merges

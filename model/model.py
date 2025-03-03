import torch
import torch.nn.functional as F
from torch import nn
import torchtune.modules as tt


class CausalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: nn.Module,
        kv_cache: any = None,
        max_seq_len: int = 128,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= attn_dropout <= 1.0):
            raise ValueError("attn_dropout must be between 0.0 and 1.0")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.attn_dropout = attn_dropout

        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_value_tensor: torch.Tensor,
        *,
        mask: torch.Tensor = None,
        input_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query_tensor (Tensor): shape [batch, seq_len, embed_dim]
            key_value_tensor (Tensor): shape [batch, seq_len, embed_dim] (same as query_tensor)
            mask (Optional[Tensor]): shape [batch, seq_len, seq_len]
            input_pos (Optional[Tensor]): position ids for tokens

        Returns:
            Tensor: output after attention, shape [batch, seq_len, embed_dim]
        """
        if query_tensor.shape != key_value_tensor.shape:
            raise ValueError("query_tensor and key_value_tensor must have the same shape")

        bsz, seq_len, _ = query_tensor.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")

        # Compute projections
        q = self.q_proj(query_tensor)
        k = self.k_proj(key_value_tensor)
        v = self.v_proj(key_value_tensor)

        # Determine how many query heads per key/value head
        q_per_kv = self.num_heads // self.num_kv_heads

        # Reshape to separate kv-heads and queries per key-value:
        # q: [bsz, seq_len, num_kv_heads, q_per_kv, head_dim]
        # k, v: [bsz, seq_len, num_kv_heads, 1, head_dim]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # Merge kv-head dims to apply positional embeddings (expects [bsz, seq_len, num_heads, head_dim])
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        q = self.pos_embeddings(q, input_pos=input_pos)
        k = self.pos_embeddings(k, input_pos=input_pos)

        # Rearrange to [bsz, num_heads, seq_len, head_dim]
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if mask is not None:
            mask = mask[:, None, :, :]

        # Use PyTorch's scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout,
            is_causal=(self.kv_cache is None and mask is None)
        )

        # Reshape back to [bsz, seq_len, embed_dim] and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(attn_output)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_batch_size: int,
        *,
        num_kv_heads: int = None,
        kv_caching: bool = False,
        attn_dropout: float = 0.0,
        dim_mult: int = 1,
        max_seq_len: int = 128,
        rope_theta_base: int = 10000,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if kv_caching:
            print("[WARNING] KV-cache usage may break causal mask")

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        head_dim = dim // num_heads

        num_kv_heads = num_kv_heads or num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        rotary_embed = tt.RotaryPositionalEmbeddings(head_dim, max_seq_len, rope_theta_base)
        kv_cache = tt.KVCache(max_batch_size, max_seq_len, num_heads, head_dim) if kv_caching else None

        self.self_attn = CausalAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(dim, dim, bias=False),
            pos_embeddings=rotary_embed,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        hidden_dim = dim * dim_mult
        self.ff = tt.FeedForward(
            gate_proj=nn.Linear(dim, hidden_dim, bias=False),
            down_proj=nn.Linear(hidden_dim, dim, bias=False),
            up_proj=nn.Linear(dim, hidden_dim, bias=False),
        )
        self.self_norm = tt.RMSNorm(dim, norm_eps)
        self.ff_norm = tt.RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection.
        normed_x = self.self_norm(x)
        attn_out = self.self_attn(normed_x, normed_x)
        residual = attn_out + x

        # Feedforward with pre-norm and residual connection.
        ff_out = self.ff(self.ff_norm(residual))
        return residual + ff_out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_batch_size: int,
        *,
        num_kv_heads: int = None,
        kv_caching: bool = False,
        attn_dropout: float = 0.0,
        dim_mult: int = 1,
        max_seq_len: int = 128,
        rope_theta_base: int = 10000,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if kv_caching:
            print("[WARNING] KV-cache usage may break causal mask")

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        head_dim = dim // num_heads

        num_kv_heads = num_kv_heads or num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self_embed = tt.RotaryPositionalEmbeddings(head_dim, max_seq_len, rope_theta_base)
        cross_embed = tt.RotaryPositionalEmbeddings(head_dim, max_seq_len, rope_theta_base)

        self_kv_cache = tt.KVCache(max_batch_size, max_seq_len, num_heads, head_dim) if kv_caching else None
        cross_kv_cache = tt.KVCache(max_batch_size, max_seq_len, num_heads, head_dim) if kv_caching else None

        self.self_attn = CausalAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(dim, dim, bias=False),
            pos_embeddings=self_embed,
            kv_cache=self_kv_cache,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        self.cross_attn = CausalAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(dim, dim, bias=False),
            pos_embeddings=cross_embed,
            kv_cache=cross_kv_cache,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        hidden_dim = dim * dim_mult
        self.ff = tt.FeedForward(
            gate_proj=nn.Linear(dim, hidden_dim, bias=False),
            down_proj=nn.Linear(hidden_dim, dim, bias=False),
            up_proj=nn.Linear(dim, hidden_dim, bias=False),
        )

        self.self_norm = tt.RMSNorm(dim, norm_eps)
        self.cross_query_norm = tt.RMSNorm(dim, norm_eps)
        self.cross_key_norm = tt.RMSNorm(dim, norm_eps)
        self.ff_norm = tt.RMSNorm(dim, norm_eps)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention block with residual connection.
        normed_x = self.self_norm(tgt)
        self_attn_out = self.self_attn(normed_x, normed_x)
        h = tgt + self_attn_out

        # Cross-attention block with residual connection.
        cross_attn_out = self.cross_attn(
            self.cross_query_norm(h), self.cross_key_norm(memory)
        )
        h = h + cross_attn_out

        # Feedforward block with residual connection.
        ff_out = self.ff(self.ff_norm(h))
        return h + ff_out


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        max_batch_size: int,
        *,
        num_kv_heads: int = None,
        kv_caching: bool = False,
        attn_dropout: float = 0.0,
        dim_mult: int = 1,
        max_seq_len: int = 128,
        rope_theta_base: int = 10000,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                dim=dim,
                num_heads=num_heads,
                max_batch_size=max_batch_size,
                num_kv_heads=num_kv_heads,
                kv_caching=kv_caching,
                attn_dropout=attn_dropout,
                dim_mult=dim_mult,
                max_seq_len=max_seq_len,
                rope_theta_base=rope_theta_base,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                dim=dim,
                num_heads=num_heads,
                max_batch_size=max_batch_size,
                num_kv_heads=num_kv_heads,
                kv_caching=kv_caching,
                attn_dropout=attn_dropout,
                dim_mult=dim_mult,
                max_seq_len=max_seq_len,
                rope_theta_base=rope_theta_base,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        ])

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = src
        output = tgt
        for encoder, decoder in zip(self.encoder_layers, self.decoder_layers):
            memory = encoder(memory)
            output = decoder(output, memory)
        return output

    def reset_caches(self) -> None:
        for layer in self.encoder_layers:
            if layer.self_attn.kv_cache is not None:
                layer.self_attn.kv_cache.reset()
        for layer in self.decoder_layers:
            if layer.self_attn.kv_cache is not None:
                layer.self_attn.kv_cache.reset()
            if layer.cross_attn.kv_cache is not None:
                layer.cross_attn.kv_cache.reset()


class ContinuousEmbedding(nn.Module):
    def __init__(self, num_continuous: int, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_continuous, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SeqFTTransformer(nn.Module):
    def __init__(
        self,
        *,
        comb_category_sizes: list,
        comb_category_emb_dim: int,
        sep_category_sizes: list,
        sep_category_emb_dims: list,
        pad_idx: int,
        num_continuous: int,
        dim: int,
        depth: int,
        num_heads: int,
        tgt_categories: int,
        max_seq_len: int,
        max_batch_size: int,
        out_categories: int,
        num_kv_heads: int = None,
        kv_caching: bool = False,
        attn_dropout: float = 0.0,
        hidden_mult: int = 1,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        total_input_size = len(comb_category_sizes) + len(sep_category_emb_dims) + num_continuous
        if total_input_size <= 0:
            raise ValueError("Total input size must be at least 1")
        if any(n <= 0 for n in comb_category_sizes):
            raise ValueError("All combined category sizes must be positive")
        if len(sep_category_sizes) != len(sep_category_emb_dims):
            raise ValueError("sep_category_sizes and sep_category_emb_dims must have the same length")
        if depth <= 0:
            raise ValueError("depth must be at least 1")
        if out_categories <= 0:
            raise ValueError("out_categories must be at least 1")

        self.pad_idx = pad_idx

        total_emb_size = 0

        # Combined categorical embeddings
        self.num_comb_categories = len(comb_category_sizes)
        self.total_comb_categories = sum(comb_category_sizes)
        if self.num_comb_categories > 0:
            # Calculate offsets for each category index
            comb_offsets = F.pad(torch.tensor(comb_category_sizes), (1, 0), value=0).cumsum(dim=-1)[:-1]
            self.register_buffer('comb_cat_offset', comb_offsets)
            self.comb_cat_embed = nn.Embedding(self.total_comb_categories, comb_category_emb_dim, padding_idx=0)
            total_emb_size += comb_category_emb_dim * self.num_comb_categories

        # Separate categorical embeddings
        self.num_sep_categories = len(sep_category_emb_dims)
        if self.num_sep_categories > 0:
            self.sep_cat_embed = nn.ModuleList([
                nn.Embedding(e_size, e_dim, padding_idx=self.pad_idx)
                for e_size, e_dim in zip(sep_category_sizes, sep_category_emb_dims)
            ])
            total_emb_size += sum(sep_category_emb_dims)

        # Continuous features embedding
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.cont_embed = ContinuousEmbedding(self.num_continuous, dim)
            total_emb_size += dim

        # Target token embedding
        self.tgt_embed = nn.Embedding(tgt_categories, dim, padding_idx=0)
        self.tgt_norm = tt.RMSNorm(dim, norm_eps)

        # Projection of concatenated embeddings to model dimension
        self.proj = nn.Linear(total_emb_size, dim, bias=False)
        self.proj_norm = tt.RMSNorm(dim, norm_eps)

        # Transformer
        self.transformer = Transformer(
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            max_batch_size=max_batch_size,
            num_kv_heads=num_kv_heads,
            kv_caching=kv_caching,
            attn_dropout=attn_dropout,
            dim_mult=hidden_mult,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
        )

        self.output_norm = tt.RMSNorm(dim, norm_eps)
        self.output = nn.Linear(dim, out_categories, bias=False)

    def reset_caches(self) -> None:
        self.transformer.reset_caches()

    def forward(
        self,
        x_comb_cat: torch.Tensor,
        x_sep_cat: torch.Tensor,
        x_cont: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        # Validate input dimensions
        if x_comb_cat.shape[-1] != self.num_comb_categories:
            raise ValueError(f"Expected {self.num_comb_categories} combined categories, got {x_comb_cat.shape[-1]}")
        if x_sep_cat.shape[-1] != self.num_sep_categories:
            raise ValueError(f"Expected {self.num_sep_categories} separate categories, got {x_sep_cat.shape[-1]}")
        if x_cont.shape[-1] != self.num_continuous:
            raise ValueError(f"Expected {self.num_continuous} continuous values, got {x_cont.shape[-1]}")

        embeddings = []

        # Combined categorical embeddings
        if self.num_comb_categories > 0:
            comb_indices = x_comb_cat + self.comb_cat_offset
            comb_emb = self.comb_cat_embed(comb_indices)
            # Concatenate along the embedding dimension (collapse category axis)
            comb_emb = torch.cat(torch.unbind(comb_emb, dim=2), dim=-1)
            embeddings.append(comb_emb)

        # Separate categorical embeddings
        if self.num_sep_categories > 0:
            sep_emb = torch.cat([
                embed(x.squeeze(-1))
                for embed, x in zip(self.sep_cat_embed, torch.split(x_sep_cat, 1, dim=-1))
            ], dim=-1)
            embeddings.append(sep_emb)

        # Continuous embeddings
        if self.num_continuous > 0:
            embeddings.append(self.cont_embed(x_cont))

        # Concatenate all parts and project
        x = torch.cat(embeddings, dim=-1)
        x = self.proj_norm(self.proj(x))

        # Process target tokens
        tgt_emb = self.tgt_norm(self.tgt_embed(tgt))

        # Run through transformer
        x = self.transformer(x, tgt_emb)
        return self.output(self.output_norm(x))

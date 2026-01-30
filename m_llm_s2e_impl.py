import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from langchain_openai import OpenAIEmbeddings
import math
from typing import Optional, List, Union


class SemanticProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegulatoryAlignmentLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        if self.head_dim * num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gate_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, seq_hidden: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = seq_hidden.shape

        residual = seq_hidden
        seq_hidden = self.layer_norm(seq_hidden)

        query = self.q_proj(seq_hidden)
        key = self.k_proj(text_emb)
        value = self.v_proj(text_emb)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if text_emb.dim() == 2:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(context)
        return residual + (self.gate_alpha * output)


class NTv2BackboneWrapper(nn.Module):
    def __init__(self, model_name: str, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def get_hidden_dim(self) -> int:
        return self.config.hidden_size

    def get_layers(self):
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise AttributeError("Could not locate transformer layers in backbone")

    def get_embeddings(self, input_ids):
        if hasattr(self.model, 'embeddings'):
            return self.model.embeddings(input_ids)
        elif hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'wte'):
            return self.model.wte(input_ids)
        else:
            return self.model.get_input_embeddings()(input_ids)


class TextEmbeddingAPI:
    def __init__(
            self,
            model: str = "text-embedding-3-small",
            base_url: Optional[str] = None,
            api_key: Optional[str] = None
    ):
        self.client = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key
        )

    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            vectors = [self.client.embed_query(texts)]
        else:
            vectors = self.client.embed_documents(texts)
        return torch.tensor(vectors, dtype=torch.float32)


class M_LLM_S2E(nn.Module):
    def __init__(
            self,
            genomic_model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
            text_emb_dim: int = 1536,
            cross_attn_heads: int = 8,
            freeze_backbone: bool = True,
            text_embedder: Optional[TextEmbeddingAPI] = None,
            alignment_dropout: float = 0.1
    ):
        super().__init__()

        self.backbone = NTv2BackboneWrapper(genomic_model_name)
        self.hidden_dim = self.backbone.get_hidden_dim()

        self.text_emb_dim = text_emb_dim
        self.text_projector = SemanticProjector(text_emb_dim, self.hidden_dim)

        self.transformer_layers = self.backbone.get_layers()
        self.num_layers = len(self.transformer_layers)

        self.alignment_layers = nn.ModuleList(
            [
                RegulatoryAlignmentLayer(self.hidden_dim, cross_attn_heads, dropout=alignment_dropout)
                for _ in range(self.num_layers)
            ]
        )

        if freeze_backbone:
            for param in self.backbone.model.parameters():
                param.requires_grad = False

        self.text_embedder = text_embedder
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(
            self,
            input_ids: torch.Tensor,
            text_embeddings: Optional[torch.Tensor] = None,
            attention_mask: Optional = None,
            gene_summaries: Optional[Union[str, List[str]]] = None
    ):
        batch_size, seq_len = input_ids.shape

        hidden_states = self.backbone.get_embeddings(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if text_embeddings is None:
            if gene_summaries is None or self.text_embedder is None:
                raise ValueError("text_embeddings or (gene_summaries and text_embedder) must be provided")
            text_embeddings = self.text_embedder.embed(gene_summaries).to(input_ids.device)
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.size(0) == 1 and batch_size > 1:
            text_embeddings = text_embeddings.expand(batch_size, -1)
        if text_embeddings.size(-1) != self.text_emb_dim:
            raise ValueError(f"text_embeddings must have last dimension {self.text_emb_dim}")

        projected_text = self.text_projector(text_embeddings)

        for i, (layer_module, alignment_module) in enumerate(zip(self.transformer_layers, self.alignment_layers)):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            hidden_states = alignment_module(
                seq_hidden=hidden_states,
                text_emb=projected_text
            )

        final_embedding = self.final_norm(hidden_states)

        cls_token_embedding = final_embedding[:, 0, :]
        logits = self.classifier(cls_token_embedding)

        return {
            "logits": logits,
            "sequence_embeddings": final_embedding
        }


def run_implementation_check():
    model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    try:
        print("Initializing M-LLM-S2E with NTv2-2.5B backbone...")
        model = M_LLM_S2E(genomic_model_name=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        dummy_dna = "ATTCCG" * 50
        dummy_text = torch.randn(1, 1536)

        inputs = tokenizer(dummy_dna, return_tensors="pt", max_length=100, truncation=True)

        print("Executing forward pass...")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                text_embeddings=dummy_text,
                attention_mask=inputs["attention_mask"]
            )

        print("Success.")
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Sequence Embeddings shape: {outputs['sequence_embeddings'].shape}")

    except Exception as e:
        print(f"Execution Error: {e}")


if __name__ == "__main__":
    run_implementation_check()

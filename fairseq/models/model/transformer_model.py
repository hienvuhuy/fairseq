from fairseq.models import register_model
from fairseq.models.transformer import TransformerModelBase, TransformerConfig

@register_model("transformer_mybase", dataclass=TransformerConfig)
class TransformerBase(TransformerModelBase):
    pass
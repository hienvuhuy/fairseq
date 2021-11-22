# wrap-up for transformer to deal with hydra 

from fairseq.models import register_model
from fairseq.models.transformer import TransformerModelBase, TransformerConfig

@register_model("transformer_origin", dataclass=TransformerConfig)
class TransformerBase(TransformerModelBase):
    pass
import unittest

import torch
# from fairseq.modules.multihead_attention import MultiheadAttention, MyMultiheadAttention, MyLocalMultiheadAttention
# from fairseq.modules import LayerNorm, MultiheadAttention, MyMultiheadAttention
from fairseq.modules.my_multihead_attention import MyLocalMultiheadAttention

class TestMyLocalMultiheadAttention(unittest.TestCase):
    def test_freeze_weight(self):
        my_att = MyLocalMultiheadAttention(embed_dim=16, num_heads=4)
        from pudb import set_trace; set_trace()
        print("my_att")
        print(my_att)


if __name__ == "__main__":
    unittest.main()
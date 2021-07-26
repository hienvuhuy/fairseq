import torch

def generate_masking(inputs, sentence_sep_id):
    """
        inputs: batch x length
        sentence_sep_id: the id of the sentence separation token
    """
    
    """
        Steps:
            - get shape; length
            - Create a matrix with size of length of inputs

    """

    shape = inputs.shape
    length = shape[1]
    sentence_sep_id_matrix = sentence_sep_id * torch.ones(shape, dtype=inputs.dtype)
    sentence_end = torch.eq(inputs, sentence_sep_id).to(torch.float)
    sentence_end_mask = torch.cumsum(sentence_end, axis=-1)

    sentence_end_mask_expand_row = sentence_end_mask.expand(sentence_end_mask.dtype)
    return 

_inputs = torch.tensor([[1, 3, 5, 2, 4, 5, 1, 4 ],
                            [1, 5, 4, 2, 5, 6, 1, 1 ],\
                            [1, 3, 5, 2, 4, 5, 1, 4 ],\
                            [1, 5, 4, 2, 5, 6, 1, 1 ]], dtype=torch.int)
_sentence_sep_id = 5
from pudb import set_trace; set_trace()
generate_masking(_inputs, _sentence_sep_id)
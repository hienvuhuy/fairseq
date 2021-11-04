import torch

# def generate_masking(inputs, sentence_sep_id):
#     """
#         inputs: batch x length
#         sentence_sep_id: the id of the sentence separation token
#     """
    
#     """
#         Steps:
#             - get shape; length
#             - Create a matrix with size of length of inputs

#     """

#     shape = inputs.shape
#     length = shape[1]
#     sentence_sep_id_matrix = sentence_sep_id * torch.ones(shape, dtype=inputs.dtype)
#     sentence_end = torch.eq(inputs, sentence_sep_id).to(torch.float)
#     sentence_end_mask = torch.cumsum(sentence_end, axis=-1)

#     sentence_end_mask_expand_row = sentence_end_mask.expand(sentence_end_mask.dtype)
#     return 

def generate_masking(inputs, sentence_sep_id, bool_type=False):
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
    batch_size = shape[0]
    length = shape[1]
    
    # sentence_sep_id_matrix = sentence_sep_id * torch.ones(shape, dtype=inputs.dtype)
    sentence_end = torch.eq(inputs, sentence_sep_id).to(torch.float)
    sentence_end_mask = torch.cumsum(sentence_end, axis=-1)

    # add extra shape into inputs
    # create colunm
    sentence_end_mask_col_expand = sentence_end_mask.unsqueeze(dim=-1)
    sentence_end_mask_col_expand = sentence_end_mask_col_expand.expand(batch_size, -1, length)
    
    ### create row
    sentence_end_mask_row_expand = sentence_end_mask.unsqueeze(dim=-2)
    sentence_end_mask_row_expand = sentence_end_mask_row_expand.expand(batch_size, length, -1)

    # compare to create mask
    
    mask = torch.eq(sentence_end_mask_col_expand, sentence_end_mask_row_expand)
    if bool_type:
        return ~mask
    mask = mask.to(torch.float)
    mask = -1e9 * (1.0 - mask)
    return mask

_inputs = torch.tensor([[1, 3, 5, 2, 4, 5, 1, 4 ],
                            [1, 5, 4, 2, 5, 6, 1, 1 ],\
                            [1, 3, 5, 2, 4, 5, 1, 4 ],\
                            [1, 5, 4, 2, 5, 6, 1, 1 ]], dtype=torch.int)
_sentence_sep_id = 5
from pudb import set_trace; set_trace()
mask = generate_masking(_inputs, _sentence_sep_id, True)

print(mask)

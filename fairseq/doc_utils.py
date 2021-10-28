import torch
# Initialize value of ID for _sep. These values will be change after loading data
# These values basically depends on dictionary.of source and target language
SOURCE_SEPARATION_ID = -1 # initialize value 
TARGET_SEPARATION_ID = -1 # initialize value

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
    mask = torch.eq(sentence_end_mask_col_expand, sentence_end_mask_row_expand).to(torch.float)
    mask = -1e9 * (1.0 - mask)
    return mask
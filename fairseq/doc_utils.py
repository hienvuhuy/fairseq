import torch
# Initialize value of ID for _sep. These values will be change after loading data
# These values basically depends on dictionary.of source and target language
SOURCE_SEPARATION_ID = -1 # initialize value 
TARGET_SEPARATION_ID = -1 # initialize value

# def generate_masking(inputs, sentence_sep_id, bool_type=False):
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
#     batch_size = shape[0]
#     length = shape[1]
    
#     # sentence_sep_id_matrix = sentence_sep_id * torch.ones(shape, dtype=inputs.dtype)
#     sentence_end = torch.eq(inputs, sentence_sep_id).to(torch.float)
#     sentence_end_mask = torch.cumsum(sentence_end, axis=-1)

#     # add extra shape into inputs
#     # create colunm
#     sentence_end_mask_col_expand = sentence_end_mask.unsqueeze(dim=-1)
#     sentence_end_mask_col_expand = sentence_end_mask_col_expand.expand(batch_size, -1, length)
    
#     ### create row
#     sentence_end_mask_row_expand = sentence_end_mask.unsqueeze(dim=-2)
#     sentence_end_mask_row_expand = sentence_end_mask_row_expand.expand(batch_size, length, -1)

#     # compare to create mask
    
#     mask = torch.eq(sentence_end_mask_col_expand, sentence_end_mask_row_expand)
#     if bool_type:
#         # note that value of mask is `True` means it block attention at position of `True`
#         # `False` means allow to attend
#         return ~mask
#     mask = mask.to(torch.float)
#     mask = -1e9 * (1.0 - mask)
#     return mask

# def generate_cross_masking( query, query_sep_id,
#                             key, key_sep_id, bool_type=False):
#     """
#     Input:
#         query:  batch x target length
#         key:    batch x source length
#         query_sep_id: the id of the sentence separation token in query
#         key_sep_id: the id of the sentence separation token in key
#         bool_type: the type of returned mask. True means boolean masking; false means inf masking
#     """
    
#     """
#         Steps:
#             - get shape; length
#             - Create a matrix with size of length of inputs


#     """
#     key_shape = key.shape
#     key_batch_size = key_shape[0]
#     query_shape = query.shape
#     query_batch_size = query_shape[0]
#     assert key_batch_size == query_batch_size
#     batch_size  = key_batch_size
#     key_length = key_shape[1]
#     query_length = query_shape[1]

#     query_end = torch.eq(query, query_sep_id).to(torch.float)
#     query_end_mask = torch.cumsum(query_end, axis=-1)
#     query_end_mask_col_expand = query_end_mask.unsqueeze(dim=-1)
#     query_end_mask_col_expand = query_end_mask_col_expand.expand(batch_size,  -1, key_length)



#     key_end = torch.eq(key, key_sep_id).to(torch.float)
#     key_end_mask = torch.cumsum(key_end, axis=-1)
#     key_end_mask_row_expand = key_end_mask.unsqueeze(dim=-2)
#     key_end_mask_row_expand = key_end_mask_row_expand.expand(batch_size, query_length, -1)


    
#     mask = torch.eq(query_end_mask_col_expand, key_end_mask_row_expand)

#     if bool_type:
#         return ~mask
#     # t_mask = mask.clone()
#     mask = (~mask).to(torch.float)
#     mask[mask.bool()] = -float('inf')
#     # mask = -1e9 * (1.0 - mask)
#     return mask

def generate_cross_masking( query, query_sep_id,
                            key, key_sep_id, mask_type='Binary'):
    """
    Input:
        query:  batch x target length
        key:    batch x source length
        query_sep_id: the id of the sentence separation token in query
        key_sep_id: the id of the sentence separation token in key
        bool_type: the type of returned mask. True means boolean masking; false means inf masking
            'Binary': boolean masking 
            'Value': value of -1e-9 for blocked positions
            'Inf': value of -inf for blocked positions
    """
    
    """
        Steps:
            - get shape; length
            - Create a matrix with size of length of inputs


    """
    key_shape = key.shape
    key_batch_size = key_shape[0]
    query_shape = query.shape
    query_batch_size = query_shape[0]
    assert key_batch_size == query_batch_size
    batch_size  = key_batch_size
    key_length = key_shape[1]
    query_length = query_shape[1]

    query_end = torch.eq(query, query_sep_id).to(torch.float)
    query_end_mask = torch.cumsum(query_end, axis=-1)
    query_end_mask_col_expand = query_end_mask.unsqueeze(dim=-1)
    query_end_mask_col_expand = query_end_mask_col_expand.expand(batch_size,  -1, key_length)



    key_end = torch.eq(key, key_sep_id).to(torch.float)
    key_end_mask = torch.cumsum(key_end, axis=-1)
    key_end_mask_row_expand = key_end_mask.unsqueeze(dim=-2)
    key_end_mask_row_expand = key_end_mask_row_expand.expand(batch_size, query_length, -1)


    
    mask = torch.eq(query_end_mask_col_expand, key_end_mask_row_expand)


    if mask_type=='Binary':
        return ~mask
    elif mask_type=='Inf': 
        mask = (~mask).to(torch.float)
        mask[mask.bool()] = -float('inf')
        return mask
    mask = mask.to(torch.float)
    mask = -1e9 * (1.0 - mask)
    return mask


def generate_masking(inputs, sentence_sep_id, mask_type="Binary"):
    """
    Input:
        inputs:  batch x length
        sentence_sep_id: the id of the sentence separation token in query
        bool_type: the type of returned mask. True means boolean masking; false means inf masking
            'Binary': boolean masking 
            'Value': value of -1e-9 for blocked positions
            'Inf': value of -inf for blocked positions
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
    if mask_type=='Binary':
        return ~mask
    elif mask_type=='Inf': 
        mask = (~mask).to(torch.float)
        mask[mask.bool()] = -float('inf')
        return mask
    mask = mask.to(torch.float)
    mask = -1e9 * (1.0 - mask)
    return mask
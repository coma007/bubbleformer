hparams = {
    'device' : 'cpu',
    'batch_size': 1,
    'lr': 5e-4,
    'model': {
        'dct_size': 'auto',
        'nhead': 10,
        'embed_size': 786,
        # 'self_attn_size': 400,
        'encoder_size': 14,
        'v_size': 200
    },
    'data': {
        'pos_k': 50,
        'neg_k': 4,
        'maxlen': 15
    },
    'max_title_len': 15,
}

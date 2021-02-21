from .gru import *



def get_mdrnn_layer(
    configs, emb_dim=200, 
    direction='Q', norm='', rnn_type='GRU'):
    '''
    direction = ['', 'B', 'Q']
    norm = ['', 'LN']
    Md = ['2d', '25d']
    first_layer  [True, False]
    '''
    
    _direction = direction
    _norm = norm
    _rnn_type = rnn_type
    _Md = '2d'
    
        
    cell_class = eval(f"{_norm}{rnn_type}{_Md}Cell")
    layer_class = eval(f"{_direction}{rnn_type}{_Md}Layer")
    
    layer = layer_class(configs=configs, emb_dim=emb_dim, _Cell=cell_class)
    
    return layer
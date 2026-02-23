import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    
    scores = np.matmul(Q, K.T)
    
    scaled_scores = scores / np.sqrt(d_k)
    
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.matmul(weights, V)
    
    return output, weights
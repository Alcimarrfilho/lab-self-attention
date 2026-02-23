import numpy as np
from attention import scaled_dot_product_attention

Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[10, 20], [30, 40]])

output, weights = scaled_dot_product_attention(Q, K, V)

print("Matriz de Pesos (Softmax):")
print(weights)
print("\nSaída Final:")
print(output)

assert np.allclose(weights.sum(axis=-1), 1.0)
print("\n✅ Sucesso: O mecanismo de atenção está funcionando!")
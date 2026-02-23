Lab P1-01: Mecanismo de Self-Attention

Este repositório contém a implementação do Scaled Dot-Product Attention usando NumPy.

Como Executar:
No terminal, dentro desta pasta, digite:
`python test_attention.py`
Conforme apresentado na Aula 2 (Slide 8), utilizamos o fator de escala $\sqrt{d_k}$ para evitar que o produto escalar entre $Q$ e $K$ resulte em valores de magnitude muito alta. Sem esse ajuste, a função Softmax atingiria regiões de saturação onde o gradiente é quase zero, impedindo o aprendizado eficiente do modelo.
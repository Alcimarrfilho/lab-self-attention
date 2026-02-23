Lab P1-01: Mecanismo de Self-Attention

Este repositório contém a implementação do mecanismo de **Scaled Dot-Product Attention** utilizando a biblioteca NumPy.

## Como Executar:
1. Baixar o ProjetoNo topo desta página, clique no botão verde "<> Code".Selecione "Download ZIP" e extraia os arquivos no seu computador.
2. Instalar Dependências em qualquer sistema, você precisará da biblioteca NumPy. Abra o terminal do seu sistema e digite:              pip install numpy
3. Execução por Sistema Operacional:
### Execução por Sistema Operacional

| Sistema Operacional | Como Abrir o Terminal | Comando para Rodar |
| :--- | :--- | :--- |
| **Windows** | Entre na pasta, segure o `Shift`, clique com o botão direito e escolha **"Abrir janela de PowerShell aqui"**. | `python test_attention.py` |
| **Linux (Ubuntu/Debian)** | Clique com o botão direito na pasta e selecione **"Abrir no terminal"** ou use `Ctrl + Alt + T`. | `python3 test_attention.py` |
| **macOS** | Clique com o botão direito na pasta, vá em **"Serviços"** e selecione **"Novo Terminal na Pasta"**. | `python3 test_attention.py` |

A função de atenção foi implementada seguindo a fórmula oficial:                                                               $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Por que usar o fator de escala $\sqrt{d_k}$? Precisamos utilizae a normalização pela raiz quadrada da dimensão das chaves ($d_k$). Sem esse ajuste, o produto escalar $QK^T$ resultaria em valores de magnitude muito alta. Isso faria com que a função Softmax atingisse regiões de saturação onde os gradientes são extremamente pequenos, dificultando ou impedindo o treinamento do modelo.

Exemplo de Input e OutputO script test_attention.py realiza uma validação com matrizes de teste.Configuração do Teste:Dimensão ($d_k$): 64Input (Q, K, V): Matrizes de exemplo (2x64) com valores unitários e aleatórios para verificação de fluxo.Output obtido no terminal:PlaintextMatriz de Pesos (Softmax):
[[0.73105858 0.26894142]
 [0.26894142 0.73105858]]

Saída Final (Context Vector):
[[15.37882843 25.37882843]
 [24.62117157 34.62117157]]

Sucesso: O (significa que o mecanismo está funcionando)

Nota sobre o desenvolvimento:
Este projeto foi desenvolvido com o suporte da IA Gemini. A ferramenta foi utilizada como par colaborativo para:

-Estruturação da lógica matemática no NumPy.

-Resolução de erros de ambiente (instalação de dependências).

-Auxílio na documentação e comandos de versionamento via Git.

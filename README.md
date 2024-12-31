# DIO - Tipos de Redes Deep Learning

## Introdução as Redes de Deep Learning

A introdução às redes de deep learning (aprendizado profundo) aborda os fundamentos de um ramo do aprendizado de máquina (machine learning) que utiliza redes neurais artificiais profundas para resolver problemas complexos. O aprendizado profundo é amplamente utilizado em diversas áreas, como visão computacional, processamento de linguagem natural, sistemas de recomendação e muito mais.

### **1. O que são Redes Neurais?**

Redes neurais artificiais são modelos computacionais inspirados no funcionamento do cérebro humano. Elas são compostas por unidades básicas chamadas **neurônios**, organizados em camadas:

- **Camada de entrada**: Recebe os dados brutos.

- **Camadas ocultas**: Executam cálculos intermediários e aprendem representações mais abstratas.

- **Camada de saída**: Gera a previsão ou o resultado do modelo.

Em uma rede de deep learning, existem várias camadas ocultas, o que a diferencia de redes neurais mais simples (shallow networks).

### **2. Características do Deep Learning**

- **Estrutura em várias camadas**: Permite a extração de características em diferentes níveis de abstração.

- **Treinamento supervisionado ou não supervisionado**: Pode ser aplicado para tarefas onde há ou não rótulos disponíveis.

- **Grandes volumes de dados**: As redes profundas geralmente requerem grandes conjuntos de dados para alcançar bons resultados.

- **Uso de hardware especializado**: GPUs e TPUs são frequentemente utilizados para acelerar o treinamento.

### **3. Componentes Chave**

1. **Funções de ativação**: Introduzem não-linearidades na rede, permitindo a modelagem de relações complexas. Exemplos:

   - ReLU (Rectified Linear Unit)

   - Sigmoid

   - Tanh

2. **Perda e otimização**:

   - **Função de perda**: Mede o quão bem a rede está se saindo (exemplo: erro quadrático médio, entropia cruzada).

   - **Otimizadores**: Ajustam os pesos da rede para minimizar a função de perda (exemplo: SGD, Adam).

3. **Regularização**: Técnicas como Dropout ou L2 Regularization ajudam a prevenir o overfitting.

### **4. Arquiteturas Comuns**

- **MLP (Multi-Layer Perceptron)**: A forma mais básica de redes profundas.

- **CNN (Convolutional Neural Networks)**: Especializadas em visão computacional.

- **RNN (Recurrent Neural Networks)**: Adequadas para dados sequenciais, como texto e séries temporais.

- **Transformers**: Arquiteturas modernas, amplamente usadas em processamento de linguagem natural.

### **5. Aplicações Práticas**

- **Visão computacional**: Reconhecimento facial, detecção de objetos.

- **Processamento de linguagem natural**: Tradução automática, análise de sentimentos.

- **Jogos e robótica**: Sistemas de tomada de decisão em tempo real.

- **Saúde**: Diagnóstico assistido por IA, análise de imagens médicas.

### **6. Desafios**

- **Treinamento intensivo**: Treinar redes profundas é computacionalmente caro.

- **Interpretação dos resultados**: Redes profundas são frequentemente consideradas caixas-pretas.

- **Dependência de dados**: Resultados eficazes exigem grandes e diversos conjuntos de dados.


## Algoritmos Convolucionais

Os algoritmos convolucionais são uma classe de técnicas amplamente usadas em redes neurais convolucionais (CNNs) para processar dados estruturados, como imagens e sinais, capturando características importantes em níveis crescentes de abstração. Esses algoritmos utilizam a **operação de convolução**, que combina dois conjuntos de informações, permitindo que padrões locais sejam identificados.

### **1. O que é Convolução?**

A convolução é uma operação matemática que aplica um **filtro (kernel)** a uma matriz de dados, como uma imagem. O filtro percorre a matriz em pequenas regiões chamadas de **receptive fields** (janelas de recepção), realizando um cálculo ponto a ponto e gerando uma nova matriz chamada de **mapa de características**.

#### Fórmula Matemática

Para uma entrada \(I(x,y)\) e um filtro \(K(m,n)\), a convolução é dada por:
\[
O(x,y) = \sum_{m}\sum_{n} I(x+m, y+n) \cdot K(m,n)
\]
Onde:
- \(O(x, y)\) é o valor no mapa de saída.
- \(I(x+m, y+n)\) é a região da entrada sobreposta ao filtro.
- \(K(m, n)\) são os pesos do filtro.

---

### **2. Estrutura de um Algoritmo Convolucional**

Um algoritmo convolucional típico opera em etapas sequenciais:

1. **Entrada**:
   - Recebe uma matriz (como uma imagem 2D) que pode ter múltiplas camadas (como RGB para imagens coloridas).
   
2. **Convolução**:
   - Aplica filtros que extraem características específicas, como bordas, texturas ou formas.

3. **Ativação**:
   - Utiliza funções de ativação (como ReLU) para introduzir não-linearidades.

4. **Pooling** (ou Subamostragem):
   - Reduz as dimensões do mapa de características, preservando as informações mais relevantes e melhorando a eficiência.

5. **Camadas Totalmente Conectadas**:
   - Conectam os neurônios para tomar decisões com base nas características extraídas.

### **3. Filtros e Detecção de Características**

Os filtros são pequenos arrays (por exemplo, 3x3 ou 5x5) que percorrem a entrada para detectar padrões locais. Exemplos de padrões detectados por filtros:

- Bordas horizontais ou verticais.

- Texturas.

- Formas complexas em camadas mais profundas.

Os pesos desses filtros são aprendidos durante o treinamento por meio do **gradiente descendente**.

### **4. Vantagens dos Algoritmos Convolucionais**

- **Localidade**: Focam em regiões específicas dos dados de entrada.

- **Compartilhamento de pesos**: Reduzem a complexidade do modelo ao usar o mesmo filtro em diferentes regiões.

- **Eficiência computacional**: Mais rápidos e escaláveis para grandes volumes de dados, como imagens de alta resolução.

### **5. Aplicações**

Os algoritmos convolucionais são usados em diversas áreas, incluindo:

- **Visão Computacional**: Reconhecimento de objetos, segmentação de imagens, rastreamento facial.

- **Processamento de Sinais**: Análise de áudio, detecção de padrões em ECGs.

- **Ciências Médicas**: Identificação de tumores em imagens médicas.

- **Sistemas de Vigilância**: Reconhecimento facial e de placas.

### **6. Desafios**

- **Interpretação**: As características extraídas são difíceis de interpretar diretamente.

- **Dados**: Algoritmos convolucionais requerem muitos dados rotulados para bom desempenho.

- **Hardware**: Exigem GPUs ou TPUs para treinamento eficiente.

## Propriedades Matemáticas para Algoritmos Convolucionais

Os algoritmos convolucionais, amplamente usados em redes neurais convolucionais (CNNs), fundamentam-se em propriedades matemáticas que garantem a eficiência e a capacidade de extração de características. Essas propriedades estão diretamente relacionadas à operação de convolução, à forma como os filtros interagem com os dados de entrada e aos ajustes durante o treinamento.


### **1. Linearidade**

A operação de convolução é linear. Para duas funções \( f(x) \) e \( g(x) \) e uma constante \( c \), temos:

\[
\text{Conv}(cf_1 + f_2, g) = c \cdot \text{Conv}(f_1, g) + \text{Conv}(f_2, g)
\]

Essa propriedade é importante porque permite que os filtros sejam combinados para representar múltiplas características lineares.


### **2. Associatividade**

A convolução é associativa:

\[
\text{Conv}(f, \text{Conv}(g, h)) = \text{Conv}(\text{Conv}(f, g), h)
\]

Isso significa que a ordem de aplicação de convoluções em diferentes filtros não altera o resultado final, o que pode ser explorado em implementações otimizadas.

### **3. Comutatividade**

A operação de convolução é comutativa:

\[
\text{Conv}(f, g) = \text{Conv}(g, f)
\]

Embora em redes neurais convolucionais essa propriedade seja raramente usada, ela demonstra a simetria entre o filtro e os dados de entrada.

### **4. Separabilidade de Filtros**

Alguns filtros podem ser decompostos em operações menores (separáveis), o que reduz a complexidade computacional. Por exemplo, um filtro 2D \( K(x, y) \) pode ser separado em duas convoluções 1D:

\[
K(x, y) = K_1(x) \cdot K_2(y)
\]

Isso é usado em algoritmos como convoluções separáveis em profundidade (depthwise separable convolutions), que são mais eficientes.

### **5. Invariância Espacial**

A convolução é translacionalmente invariante, o que significa que deslocamentos nos dados de entrada resultam em deslocamentos proporcionais no mapa de características, sem alterar a forma das características detectadas. Isso é crucial para a detecção de padrões independentes de posição, como bordas em imagens.

### **6. Propriedades Relativas ao Espaço**

#### a) **Dimensionalidade**

A convolução reduz as dimensões da entrada com base nos tamanhos do filtro (\(F\)), do **stride** (\(S\)) e do **padding** (\(P\)). A fórmula para calcular a dimensão de saída (\(O\)) é:

\[
O = \frac{(I - F + 2P)}{S} + 1
\]

Onde \(I\) é a dimensão da entrada.

#### b) **Overlapping Receptive Fields**

Os filtros permitem que áreas da entrada se sobreponham, garantindo uma extração mais detalhada de características locais.

### **7. Convolução Discreta vs. Convolução Contínua**

Nos algoritmos convolucionais, utiliza-se a convolução discreta, que é uma soma ponderada de valores discretos. Matematicamente, é descrita por:

\[
(f * g)[n] = \sum_{m} f[m] \cdot g[n-m]
\]

Essa diferença simplifica a implementação computacional em relação à convolução contínua, que requer integrais.

### **8. Propriedades de Derivação**

A convolução possui propriedades úteis para o treinamento de redes neurais, como:

- **Derivada da Convolução**:

\[
\frac{\partial}{\partial x} \text{Conv}(f(x), g(x)) = \text{Conv}\left(\frac{\partial f(x)}{\partial x}, g(x)\right) + \text{Conv}\left(f(x), \frac{\partial g(x)}{\partial x}\right)
\]

Isso facilita a aplicação do backpropagation no ajuste dos pesos dos filtros.

### **9. Propriedades de Frequência (Teorema da Convolução)**

A convolução no domínio do tempo/espacial equivale à multiplicação no domínio da frequência:

\[
\text{Conv}(f, g) \leftrightarrow F(u) \cdot G(u)
\]

Onde \(F(u)\) e \(G(u)\) são as transformadas de Fourier de \(f(x)\) e \(g(x)\), respectivamente. Isso é fundamental para compreender como os filtros atuam no espectro de frequências da entrada.

### **10. Regularização e Robustez**

Os algoritmos convolucionais incorporam propriedades que ajudam a regularizar e a evitar overfitting, como:

- **Compartilhamento de pesos**: O mesmo filtro é aplicado em toda a entrada, reduzindo o número de parâmetros.

- **Pooling**: Complementa as propriedades da convolução, mantendo apenas os valores mais representativos e aumentando a robustez.

Essas propriedades matemáticas tornam os algoritmos convolucionais poderosos e flexíveis, permitindo sua aplicação em problemas variados, como visão computacional, processamento de sinais e análise de séries temporais.

## Programando uma Rede de Deep Learning

Aqui está um exemplo simples de uma rede de deep learning usando a biblioteca **Keras**, integrada ao **TensorFlow**, 
para realizar classificação em um conjunto de dados muito conhecido: o **MNIST**, que contém imagens de dígitos manuscritos.

### **Exemplo de Rede Neural para Classificação MNIST**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Carregar os dados do MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (escala os valores entre 0 e 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Codificar as saídas em categorias (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Criar o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Transforma a entrada 2D em 1D
    Dense(128, activation='relu'), # Camada totalmente conectada com 128 neurônios e ativação ReLU
    Dense(64, activation='relu'),  # Camada intermediária com 64 neurônios
    Dense(10, activation='softmax') # Saída com 10 classes (0-9), usando Softmax
])

# Compilar o modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Avaliar o modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Acurácia no teste: {test_accuracy:.2f}")
```

### **Passo a Passo**

1. **Importação de bibliotecas**:
   - `tensorflow.keras` fornece uma interface de alto nível para criar e treinar redes neurais.
   
2. **Carregamento dos dados**:
   - O MNIST é carregado e dividido em conjunto de treinamento e teste.
   
3. **Pré-processamento**:
   - Os valores dos pixels são normalizados para o intervalo `[0, 1]`.
   - As classes de saída são transformadas em vetores one-hot encoding.

4. **Criação do modelo**:
   - `Flatten`: Transforma imagens 28x28 em um vetor unidimensional.
   - `Dense`: Cria camadas densamente conectadas.
   - Funções de ativação como `ReLU` e `Softmax` são usadas.

5. **Compilação**:
   - Otimizador **Adam** ajusta os pesos para minimizar a perda.
   - Função de perda **categorical_crossentropy** para classificação.

6. **Treinamento**:
   - O modelo é treinado por 5 épocas com lotes de tamanho 32 e uma validação de 20%.

7. **Avaliação**:
   - O desempenho é avaliado com o conjunto de teste, fornecendo a acurácia final.

### **Saída Esperada**

Durante o treinamento, você verá a perda e a acurácia do modelo em cada época. No final, será exibida a acurácia no conjunto de teste, algo como:

```
Acurácia no teste: 0.98
```

Esse exemplo demonstra um fluxo básico de aprendizado profundo e pode ser adaptado para tarefas mais complexas com arquiteturas de redes como CNNs ou RNNs.

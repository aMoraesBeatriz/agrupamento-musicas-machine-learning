# agrupamento-musicas-machine-learning
Projeto de Machine Learning não supervisionado para agrupamento de músicas a partir de características, como BPM, energia, dançabilidade e duração. Utiliza K-Means e DBSCAN para identificar perfis e sugerir playlists automáticas. Inclui ETL, visualizações, clusterização, comparação e análise dos resultados. Disciplina de Mineração de Dados.

Foram utilizados dois modelos supervisionados:

- K-Means
- DBSCAN

Tecnologias Utilizadas:

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Estrutura do Código
1. ETL e Preparação dos Dados
- Criação de dataset fictício com 15 músicas
- Normalização e organização dos dados
- Separação das features para clusterização

2. Visualização dos Dados
- Inclui gráficos exploratórios:
- Scatterplot BPM × Energia
- Representação 2D dos clusters formados
- Visualização de exemplos de gêneros musicais atribuídos aos clusters

3. Modelos de Machine Learning

Modelo 1 – K-Means
- Define K=3 clusters
- Agrupa músicas automaticamente
- Permite visualizar os centros dos clusters
- Identifica padrões como “músicas rápidas e energéticas” ou “músicas lentas e suaves”

Modelo 2 – DBSCAN
- Descobre clusters baseados em densidade
- Pode identificar ruídos e outliers
- Agrupamento mais flexível, sem escolher K previamente

4. Métricas e Avaliação

Para ambos os modelos foram analisados:
- Clusterização final
- Comparação visual dos agrupamentos
- Diferenças entre K-Means e DBSCAN
- Interpretação dos clusters gerados

5. Interpretação dos Resultados

Modelo 1 – K-Means

O K-Means formou três perfis musicais bem definidos.
As combinações típicas foram:
- Cluster 0 → Músicas rápidas, energéticas e altamente dançantes
- Cluster 1 → Músicas lentas e pouco energéticas
- Cluster 2 → Músicas moderadas em BPM, mas boas para dançar
Esses grupos poderiam ser usados para criar playlists automáticas, como:
- “Energia Máxima”
- “Relax & Chill”
- “Dançando no Meio-Termo”

Modelo 2 – DBSCAN

O DBSCAN identificou grupos com densidades diferentes e marcou alguns pontos como ruído.
Ele é útil para:
- Detectar músicas fora do padrão geral
- Encontrar gêneros muito distintos do conjunto principal
- Evitar clusters forçados, como ocorre às vezes no K-Means

Como Executar

Instale as dependências:

- pip install pandas numpy scikit-learn matplotlib seaborn

Execute o script:

- python pi2_machine_learning.py

Arquivos do Projeto

- pi2_machine_learning.py – Código completo do projeto

- README.md – Documentação explicativa

Licença

Este projeto foi desenvolvido para fins acadêmicos, como parte da Disciplina de Mineração de Dados do Curso de Sistemas de Informação da Universidade do Estado de Minas Gerais.

# Análise Comparativa de Ferramentas e Técnicas de Anonimização de Textos

Este repositório contém os conjuntos de dados, modelos e resultados e códigos do Trabalho de Conclusão de Curso (TCC).

🔗 [Acesse o TCC](https://github.com/seu-usuario/seu-repositorio)

---

## 📂 Estrutura do Repositório

### 1. Datasets
Os conjuntos de dados utilizados na pesquisa estão disponíveis na pasta [`/datasets`](./datasets). Os arquivos foram preservados em duas formas:
* Em sua estrutura e formato original.
* Convertidos e estruturados em formato `.json`.

### 2. Resultados
Os dados gerados pelas avaliações do TCC estão localizados na pasta [`/resultados`](./resultados), organizados individualmente por dataset. Cada modelo testado possui exatamente 3 arquivos de mapeamento:

* **`attack_dataset_(nome_modelo).json`**: Contém o texto original, o texto anonimizado correspondente e as *labels* que servem de entrada para o modelo atacante.
* **`geral_info_(nome_modelo).json`**: Consolida todas as métricas de desempenho, taxas de acerto e resultados consolidados obtidos pelo modelo.
* **`log_info_(nome_modelo).json`**: Armazena o texto original acompanhado das *labels* verdadeiras (*ground truth*) e das *labels* preditas pelo modelo. Este arquivo pode ser utilizado para realizar uma analise na qualidade da anonimização.

---

## ✉️ Contato

Se tiver qualquer dúvida ou precisar de mais informações sobre os experimentos, sinta-se à vontade para entrar em contato:

📩 [seu-email@instituicao.edu.br](mailto:seu-email@instituicao.edu.br)
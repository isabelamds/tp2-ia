# TP2 – Busca Local e Algoritmo Genético

Sequenciamento de manutenção de ativos: comparação entre greedy descent com
random restart, simulated annealing e algoritmo genético.

## Requisitos

- Python 3.10 ou superior 
- Bibliotecas para a análise: `matplotlib`, `numpy`, `scipy`

Instalação sugerida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `main.py` | executa o tudo na ordem correta |
| `problema.py` | leitura, função h, vizinhanças, avaliação incremental, cruzamento OX e mutação |
| `sa.py` | simulated annealing |
| `ga.py` | algoritmo genético |
| `experimentos.py` | roda e grava os resultados |
| `analise.py` | tabelas, testes estatísticos e gráficos |
| `calibrar_sa.py`, `calibrar_ga.py` | calibração dos parâmetros do SA e do GA |
| `medir_incremental.py` | ganho de tempo da avaliação incremental |
| `testes.py` | verificações automáticas |
| `instancias/` | as três instâncias |
| `resultados/` | todos os dados gerados |

Todos os comandos abaixo são executados na pasta principal do projeto.

## Reproduzindo tudo com um comando

```bash
python3 main.py              # testes, ganho incremental, experimentos e análise
python3 main.py --calibrar   # o mesmo, refazendo antes as calibrações (+10 a 15 min)
```


## Reproduzindo cada resultado

Todas as execuções usam sementes fixas, então os números se repetem exatamente.

### T1 – validação da função de avaliação

```bash
python3 testes.py
```

Confere h = 4366 para a permutação identidade e h = 1875 para a rota ótima da
instância pequena, imprime o cronograma, e testa vizinhanças, avaliação
incremental, orçamento, cruzamento e mutação.

### Ganho da avaliação incremental

```bash
python3 medir_incremental.py
```

Gera `resultados/tempo_incremental.csv`.

### Calibração do simulated annealing

```bash
python3 calibrar_sa.py
```

Gera `resultados/calibracao_sa_deltas.csv` e `resultados/calibracao_sa_grade.csv`
As temperaturas escolhidas estão em `experimentos.py`.

`python3 calibrar_sa.py swap` repete a calibração com a vizinhança swap e gera
`calibracao_sa_deltas_swap.csv` e `calibracao_sa_grade_swap.csv`.

### Calibração do algoritmo genético

```bash
python3 calibrar_ga.py media
```

Gera `resultados/calibracao_ga_media.csv` e `resultados/calibracao_ga_media_etapa2.csv`.


### Experimentos principais e experimento adicional do GA

```bash
python3 experimentos.py todos
```

Roda todas as configurações nas três instâncias (30/30/10 execuções, orçamentos
50.000/150.000/300.000 avaliações).
Grava `resultados/execucoes.csv` e `resultados/execucoes/*.json`.
Se for interrompido, rodar de novo continua de onde parou.

O experimento adicional é a configuração `ga_sem_cruzamento`.

### Tabelas e figuras

```bash
python3 analise.py
```

Gera, em `resultados/analise/`:

| Arquivo | Item do relatório |
|---|---|
| `tabela_principal.md` / `.csv` | tabela principal (média ± DP, mínimo, mediana, gap, tempo, avaliações) |
| `testes_mann_whitney.md` / `.csv` | testes estatísticos par a par |
| `boxplots.png` | boxplots por instância |
| `convergencia_pequena.png`, `_media.png`, `_grande.png` | curvas de convergência (com a média da população do GA) |
| `sa_aceitacao.png` | taxa de aceitação de pioras do SA |
| `ga_diversidade.png` | diversidade da população do GA |
| `ga_operadores.png` / `.csv` | experimento adicional: sucesso do cruzamento × mutação |

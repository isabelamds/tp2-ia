"""
ga.py - Algoritmo Genetico.
    1. elitismo: os 'elite' melhores passam direto para a proxima geracao
    2. ate completar N individuos:
       - seleciona dois pais por torneio
       - com probabilidade Pc, cruza (OX) e gera dois filhos; senao, copia os pais
       - cada filho sofre mutacao com probabilidade Pm
       - avalia os filhos que mudaram (copias identicas herdam o custo do pai)
    3. a nova populacao substitui a antiga (substituicao geracional)
"""

from problema import OrcamentoEsgotado


def algoritmo_genetico(prob, N=100, Pc=0.9, Pm=0.2, elite=2, k_torneio=3,
                       pares_diversidade=50):
    """Parametros:
    N          tamanho da populacao
    Pc         probabilidade de cruzamento de um par de pais
    Pm         probabilidade de mutacao de cada filho
    elite      quantos dos melhores sobrevivem sem alteracao
    k_torneio  quantos individuos disputam cada torneio
    pares_diversidade  quantos pares sortear para estimar a distancia media
    """
    rng = prob.rng
    n = prob.n

    extras = {
        "N": N, "Pc": Pc, "Pm": Pm, "elite": elite, "k_torneio": k_torneio,
        "geracoes": 0,
        # uma entrada por geracao:
        "ger_avaliacao": [],    # avaliacoes gastas ate o inicio da geracao
        "ger_melhor": [],       # melhor h da populacao
        "ger_media": [],        # h medio da populacao
        "ger_distintos": [],    # numero de cromossomos diferentes
        "ger_distancia": [],    # distancia media entre pares (0 = iguais, 1 = nada em comum)
        # sucesso dos operadores (para testar a hipotese do cruzamento), por geracao:
        "ger_cruz_filhos": [],    # filhos gerados por cruzamento (com ou sem mutacao)
        "ger_cruz_sucessos": [],  # ... que ficaram melhores que o MELHOR dos dois pais
        "ger_mut_filhos": [],     # filhos gerados so por mutacao (copia do pai + mutacao)
        "ger_mut_sucessos": [],   # ... que ficaram melhores que o pai
    }

    def registrar(pop, custos):
        extras["ger_avaliacao"].append(prob.avaliacoes)
        extras["ger_melhor"].append(min(custos))
        extras["ger_media"].append(sum(custos) / len(custos))
        extras["ger_distintos"].append(len({tuple(s) for s in pop}))
        # distancia de Hamming normalizada: fracao das posicoes com ativos diferentes
        soma = 0.0
        for _ in range(pares_diversidade):
            a, b = rng.sample(range(len(pop)), 2)
            sa, sb = pop[a], pop[b]
            soma += sum(1 for k in range(n) if sa[k] != sb[k]) / n
        extras["ger_distancia"].append(soma / pares_diversidade)

    def torneio(custos):
        """Sorteia k individuos e devolve o indice do de MENOR custo."""
        candidatos = [rng.randrange(N) for _ in range(k_torneio)]
        return min(candidatos, key=lambda i: custos[i])

    try:
        # Inicializacao
        pop = [prob.solucao_aleatoria() for _ in range(N)]
        custos = [prob.h(s) for s in pop]

        while True:
            registrar(pop, custos)
            extras["geracoes"] += 1

            # 1. elitismo
            ordem = sorted(range(N), key=lambda i: custos[i])
            nova = [pop[i] for i in ordem[:elite]]
            novos_custos = [custos[i] for i in ordem[:elite]]

            # 2. reproducao
            cruz_n = cruz_ok = mut_n = mut_ok = 0
            extras["ger_cruz_filhos"].append(0)
            extras["ger_cruz_sucessos"].append(0)
            extras["ger_mut_filhos"].append(0)
            extras["ger_mut_sucessos"].append(0)
            while len(nova) < N:
                i, j = torneio(custos), torneio(custos)
                cruzou = rng.random() < Pc
                if cruzou:
                    filhos = [prob.cruzamento(pop[i], pop[j]),
                              prob.cruzamento(pop[j], pop[i])]
                    herdado = [None, None]                 # filhos novos: avaliar
                    referencia = [min(custos[i], custos[j])] * 2   # melhor dos pais
                else:
                    filhos = [pop[i], pop[j]]
                    herdado = [custos[i], custos[j]]       # copias: herdam o custo
                    referencia = [custos[i], custos[j]]    # o proprio pai

                for f, c, ref in zip(filhos, herdado, referencia):
                    if len(nova) >= N:
                        break
                    mutou = rng.random() < Pm
                    if mutou:
                        f = prob.mutacao(f)
                        c = None
                    if c is None:
                        c = prob.h(f)                      
                        if cruzou:
                            cruz_n += 1
                            cruz_ok += c < ref
                        else:
                            mut_n += 1
                            mut_ok += c < ref
                        extras["ger_cruz_filhos"][-1] = cruz_n
                        extras["ger_cruz_sucessos"][-1] = cruz_ok
                        extras["ger_mut_filhos"][-1] = mut_n
                        extras["ger_mut_sucessos"][-1] = mut_ok
                    nova.append(f)
                    novos_custos.append(c)

            # 3. substituicao geracional
            pop, custos = nova, novos_custos

    except OrcamentoEsgotado:
        pass

    return extras

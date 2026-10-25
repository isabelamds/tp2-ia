import math

from problema import OrcamentoEsgotado


def simulated_annealing(prob, T0, Tf, janelas=100):
    #Simulated annealing.
    B = prob.orcamento
    passos = B - 1                       # 1 avaliacao vai para a solucao inicial
    alfa = (Tf / T0) ** (1.0 / passos)
    tamanho_janela = max(1, passos // janelas)
    rng = prob.rng

    extras = {
        "T0": T0, "Tf": Tf, "alfa": alfa,
        "aceitos_total": 0,
        # uma entrada por janela:
        "janela_avaliacao": [],     # avaliacao em que a janela terminou
        "janela_temperatura": [],   # temperatura no fim da janela
        "janela_taxa_piores": [],   # aceitos piores / propostos piores
        "janela_h_corrente": [],    # custo da solucao corrente no fim da janela
    }

    piores_propostos = 0
    piores_aceitos = 0
    iteracao = 0

    try:
        s = prob.solucao_aleatoria()
        hs = prob.h(s)
        T = T0

        while True:
            mov, hv = prob.sucessor_aleatorio(s)
            delta = hv - hs

            if delta <= 0:
                aceita = True                          # melhor ou igual: aceita
            else:
                piores_propostos += 1
                # distribuicao de Gibbs: e^(-delta/T)
                aceita = rng.random() < math.exp(-delta / T)
                if aceita:
                    piores_aceitos += 1

            if aceita:
                s = prob.aplicar(s, mov)
                hs = hv
                extras["aceitos_total"] += 1

            T *= alfa
            iteracao += 1

            if iteracao % tamanho_janela == 0:
                extras["janela_avaliacao"].append(prob.avaliacoes)
                extras["janela_temperatura"].append(T)
                extras["janela_taxa_piores"].append(
                    piores_aceitos / piores_propostos if piores_propostos else 0.0)
                extras["janela_h_corrente"].append(hs)
                piores_propostos = piores_aceitos = 0

    except OrcamentoEsgotado:
        pass

    return extras

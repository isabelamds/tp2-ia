import numbers

from problema import OrcamentoEsgotado

ESTRATEGIAS_GREEDY = ("best", "first")


def greedy_descent(problema, estrategia="best", semente=0, orcamento=50_000):
    """
    Parametros
    ---------
    problema : ProblemaManutencao
        Instancia do problema ja configurada com uma vizinhanca.
    estrategia : {"best", "first"}
        "best": avalia toda a vizinhanca e aplica a melhor melhora encontrada.
        "first": percorre os movimentos em ordem aleatoria e aplica a primeira
                 melhora encontrada.
    semente : int
        Semente dos sorteios. Padrao 0, entao duas chamadas com os mesmos
        argumentos produzem exatamente a mesma execucao. Para repeticoes
        independentes, passe uma semente diferente em cada uma e grave-as.
    orcamento : int
        Numero maximo de avaliacoes da funcao objetivo. Deve ser positivo.
        Aceita int do Python e inteiros do numpy.

    Retorno
    -------
    dict
        "estrategia"
        "vizinhanca"
        "semente"
        "melhor_s"
        "melhor_h"
        "avaliacoes"
        "orcamento"
        "reinicios"                 
        "passos_melhoria"
        "otimos_locais_distintos"
        "quantidade_otimos_locais_distintos"
        "historico"
    """
    if estrategia not in ESTRATEGIAS_GREEDY:
        raise ValueError(
            f"Estrategia desconhecida: {estrategia!r}. "
            f"Use uma destas: {ESTRATEGIAS_GREEDY}."
        )
    # numbers.Integral cobre int e inteiros do numpy (np.int64 etc.).
    if (not isinstance(orcamento, numbers.Integral)
            or isinstance(orcamento, bool) or orcamento <= 0):
        raise ValueError("orcamento deve ser um inteiro positivo")
    orcamento = int(orcamento)
    if (not isinstance(semente, numbers.Integral) or isinstance(semente, bool)):
        raise ValueError("semente deve ser um inteiro (None nao e aceito, "
                         "pois tornaria a execucao nao reproduzivel)")
    semente = int(semente)

    problema.reiniciar(semente=semente, orcamento=orcamento)

    reinicios = 0
    passos_melhoria = 0
    # Chave: permutacao (tupla). Assim contamos otimos locais distintos.
    otimos_locais_distintos = {}

    try:
        while True:
            # Random restart: comeca de uma nova permutacao aleatoria.
            s = problema.solucao_aleatoria()
            h_atual = problema.h(s)  # conta uma avaliacao e registra a solucao
            reinicios += 1

            while True:
                movimento_escolhido = None
                melhor_valor = h_atual

                # movimentos() devolve os movimentos em ordem aleatoria.
                for mov in problema.movimentos():
                    # So interessa um vizinho que supere o melhor valor ja
                    # encontrado nesta varredura (inicialmente h_atual). Se a
                    # avaliacao incremental passar desse limite, ela pode parar
                    # cedo, pois o vizinho nao seria escolhido de qualquer forma.
                    h_vizinho = problema.avaliar_movimento(
                        s, mov, limite=melhor_valor
                    )

                    if h_vizinho < melhor_valor:
                        melhor_valor = h_vizinho
                        movimento_escolhido = mov

                        if estrategia == "first":
                            break

                if movimento_escolhido is None:
                    # Toda a vizinhanca foi examinada sem melhora: otimo local.
                    otimos_locais_distintos.setdefault(tuple(s), h_atual)
                    break  # inicia outro random restart

                # Aplica a melhora selecionada; o custo ja foi calculado acima.
                s = problema.aplicar(s, movimento_escolhido)
                h_atual = melhor_valor
                passos_melhoria += 1

    except OrcamentoEsgotado:
        # A classe lancou a excecao ao tentar ultrapassar o limite. Nenhuma
        # avaliacao extra foi contabilizada: problema.avaliacoes == orcamento.
        pass

    otimos = [
        {"solucao": list(solucao), "h": valor}
        for solucao, valor in otimos_locais_distintos.items()
    ]

    return {
        "estrategia": estrategia,
        "vizinhanca": getattr(problema, "vizinhanca", None),
        "semente": semente,
        "melhor_s": list(problema.melhor_s) if problema.melhor_s is not None else None,
        "melhor_h": problema.melhor_h,
        "avaliacoes": problema.avaliacoes,
        "orcamento": orcamento,
        "reinicios": reinicios,
        "passos_melhoria": passos_melhoria,
        "otimos_locais_distintos": otimos,
        "quantidade_otimos_locais_distintos": len(otimos),
        "historico": list(problema.historico),
    }


# ----------------------------------------------------------------------------
# T1 - Validacao da funcao de avaliacao (Secao 1.4 do enunciado)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from problema import ProblemaManutencao

    #CAMINHO = sys.argv[1] if len(sys.argv) > 1 else "manutencao_pequena.txt"
    CAMINHO = "/home/itamar/Documentos/GitHub/tp2-ia/instancias/manutencao_pequena.txt"
    H_IDENTIDADE = 4366    # h(1, 2, ..., 12) segundo o enunciado
    H_OTIMO = 1875         # otimo global da instancia pequena (busca exata)

    prob = ProblemaManutencao(CAMINHO)
    print(f"Instancia: {CAMINHO} (n = {prob.n})")

    # Verificacao 1: permutacao identidade
    identidade = list(range(1, prob.n + 1))
    h_id = prob.custo(identidade)
    print("\n[T1.1] h(identidade)")
    print(prob.cronograma(identidade))
    print(f"obtido = {h_id} | esperado = {H_IDENTIDADE} | "
          f"{'OK' if h_id == H_IDENTIDADE else 'FALHOU'}")

    # Verificacao 2: o otimo 1875 nao tem permutacao conhecida, entao nao da
    # para avalia-lo diretamente. Rodamos o greedy: nenhuma permutacao pode
    # ficar abaixo de 1875; se ficar, a funcao de avaliacao esta errada.
    print("\n[T1.2] Consistencia com o otimo global (h* = 1875)")
    consistente = True
    for estrategia in ESTRATEGIAS_GREEDY:
        r = greedy_descent(prob, estrategia=estrategia, semente=0, orcamento=50_000)
        gap = (r["melhor_h"] - H_OTIMO) / H_OTIMO
        ok = r["melhor_h"] >= H_OTIMO and r["avaliacoes"] == r["orcamento"]
        consistente = consistente and ok
        print(f"  {estrategia:>5}: melhor_h = {r['melhor_h']} | gap = {gap:.2%} | "
              f"avaliacoes = {r['avaliacoes']}/{r['orcamento']} | "
              f"{'OK' if ok else 'FALHOU'}")
        # O melhor_h reportado deve bater com a recomputacao completa.
        assert prob.custo(r["melhor_s"]) == r["melhor_h"], "melhor_h inconsistente"

    print("\nResultado da validacao:",
          "APROVADA" if h_id == H_IDENTIDADE and consistente else "REPROVADA")

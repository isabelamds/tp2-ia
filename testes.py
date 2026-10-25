import math
import random

from problema import ProblemaManutencao, OrcamentoEsgotado, arredondar, VIZINHANCAS

PEQUENA = "instancias/manutencao_pequena.txt"
INSTANCIAS = [
    "instancias/manutencao_pequena.txt",
    "instancias/manutencao_media.txt",
    "instancias/manutencao_grande.txt",
]


# Funcao de avaliacao

def teste_arredondamento():
    assert arredondar(2.5) == 3, "2.5 deveria virar 3"
    assert arredondar(3.5) == 4
    assert arredondar(2.49) == 2
    assert arredondar(0) == 0
    print("OK  arredondamento (2.5 -> 3, e nao 2 como no round do Python)")


def teste_otimo_pequena():
    # Rota otima da instancia pequena
    prob = ProblemaManutencao(PEQUENA)
    otima = [5, 4, 1, 6, 7, 2, 8, 12, 3, 10, 9, 11]
    valor = prob.custo(otima)
    assert valor == 1875, f"rota otima deveria dar 1875, deu {valor}"
    print("OK  rota otima da instancia pequena: h = 1875")


def teste_identidade_pequena():
    prob = ProblemaManutencao(PEQUENA)
    pi = list(range(1, prob.n + 1))       
    valor = prob.custo(pi)
    assert valor == 4366, f"identidade deveria dar 4366, deu {valor}"
    print("OK  identidade na instancia pequena: h = 4366")
    print(prob.cronograma(pi))             


def teste_contador():
    prob = ProblemaManutencao(PEQUENA)
    prob.reiniciar(semente=1)
    pi = prob.solucao_aleatoria()
    prob.custo(pi)                    # nao deve contar
    prob.h(pi)                        # deve contar
    prob.h(pi)                        # deve contar
    assert prob.avaliacoes == 2, f"contador deveria ser 2, e {prob.avaliacoes}"
    print("OK  contador: custo() nao conta, h() conta")


def teste_semente():
    prob = ProblemaManutencao(PEQUENA)
    prob.reiniciar(semente=42)
    a = prob.solucao_aleatoria()
    prob.reiniciar(semente=42)
    b = prob.solucao_aleatoria()
    assert a == b, "mesma semente deveria gerar a mesma permutacao"
    assert sorted(a) == list(range(1, prob.n + 1)), "nao e uma permutacao valida"
    print("OK  semente: mesma semente -> mesma solucao aleatoria")


# Vizinhancas, avaliacao incremental, orcamento
def teste_figura3():
    # Exemplo da Figura 3 
    prob = ProblemaManutencao(PEQUENA)
    s = [3, 7, 1, 5, 2, 8]
    esperado = {
        "swap":      [3, 2, 1, 5, 7, 8],
        "insertion": [3, 1, 5, 2, 7, 8],
        "2opt":      [3, 2, 5, 1, 7, 8],
    }
    for nome, alvo in esperado.items():
        prob.definir_vizinhanca(nome)
        obtido = prob.aplicar(s, (1, 4))
        assert obtido == alvo, f"{nome}: esperava {alvo}, obtive {obtido}"
        assert s == [3, 7, 1, 5, 2, 8], "aplicar() nao pode modificar s"
    print("OK  vizinhancas reproduzem a Figura 3 do enunciado")


def teste_tamanhos():
    # Confere a formula do tamanho e que todos os sucessores sao distintos.
    prob = ProblemaManutencao(PEQUENA)
    n = prob.n
    formula = {"swap": n * (n - 1) // 2,
               "insertion": (n - 1) ** 2,
               "2opt": n * (n - 1) // 2}
    s = list(range(1, n + 1))
    for nome in VIZINHANCAS:
        prob.definir_vizinhanca(nome)
        assert prob.tamanho_vizinhanca() == formula[nome], nome
        distintos = {tuple(prob.aplicar(s, m)) for m in prob.movimentos()}
        assert len(distintos) == formula[nome], f"{nome}: sucessores repetidos"
        assert tuple(s) not in distintos, f"{nome}: sucessor igual a s"
    print(f"OK  tamanhos (n={n}): swap={formula['swap']}, "
          f"insertion={formula['insertion']}, 2opt={formula['2opt']}")


def teste_incremental():
    # Para movimentos sorteados, a avaliacao incremental tem que dar
    # exatamente o mesmo que calcular o custo do sucessor do zero.
    rng = random.Random(0)
    total = 0
    for caminho in INSTANCIAS:
        for nome in VIZINHANCAS:
            prob = ProblemaManutencao(caminho, vizinhanca=nome)
            prob.reiniciar(semente=1)
            # Solucoes variadas: aleatorias (quase tudo atrasado) e ordenadas
            # por prazo com pequenas perturbacoes (mistura de no prazo e atrasado),
            # para exercitar todos os atalhos da avaliacao incremental.
            por_prazo = sorted(range(1, prob.n + 1), key=lambda j: prob.d[j])
            solucoes = [prob.solucao_aleatoria() for _ in range(10)]
            for _ in range(10):
                s = list(por_prazo)
                for _ in range(rng.randint(0, 5)):
                    a, b = rng.sample(range(prob.n), 2)
                    s[a], s[b] = s[b], s[a]
                solucoes.append(s)
            for s in solucoes:
                prob.melhor_h = -math.inf      # desliga o registro do melhor
                for mov in rng.sample(prob.movimentos(), 50):
                    inc = prob.avaliar_movimento(s, mov)
                    completo = prob.custo(prob.aplicar(s, mov))
                    assert inc == completo, (
                        f"{caminho} {nome} {mov}: incremental={inc}, completo={completo}")
                    total += 1
    print(f"OK  avaliacao incremental = avaliacao completa ({total} movimentos testados)")


def teste_poda():
    # Com limite: se o valor devolvido e < limite, ele e exato;
    # se e >= limite, o valor verdadeiro tambem e >= limite.
    rng = random.Random(1)
    prob = ProblemaManutencao(INSTANCIAS[1], vizinhanca="insertion")
    prob.reiniciar(semente=2)
    for _ in range(500):
        s = prob.solucao_aleatoria()
        mov = rng.choice(prob.movimentos())
        verdadeiro = prob.custo(prob.aplicar(s, mov))
        limite = verdadeiro + rng.randint(-2000, 2000)
        prob.melhor_h = -math.inf
        v = prob.avaliar_movimento(s, mov, limite=limite)
        if v < limite:
            assert v == verdadeiro
        else:
            assert verdadeiro >= limite
    print("OK  poda com limite e segura")


def teste_orcamento_e_registro():
    prob = ProblemaManutencao(INSTANCIAS[1], vizinhanca="swap")
    prob.reiniciar(semente=3, orcamento=1000)
    s = prob.solucao_aleatoria()
    prob.h(s)
    try:
        while True:
            mov, v = prob.sucessor_aleatorio(s)
            if v < prob.custo(s):
                s = prob.aplicar(s, mov)
    except OrcamentoEsgotado:
        pass
    assert prob.avaliacoes == 1000, f"parou em {prob.avaliacoes}, e nao em 1000"
    assert prob.custo(prob.melhor_s) == prob.melhor_h, "melhor_s nao bate com melhor_h"
    valores = [v for _, v in prob.historico]
    assert valores == sorted(valores, reverse=True), "historico deveria so diminuir"
    print("OK  orcamento para exatamente no limite; melhor solucao registrada corretamente")


def teste_cruzamento_mutacao():
    prob = ProblemaManutencao(INSTANCIAS[2])           # n = 150
    prob.reiniciar(semente=4)
    filho = prob._ox([1, 2, 3, 4, 5], [5, 4, 2, 1, 3], 1, 3)
    assert filho == [5, 2, 3, 4, 1], f"OX deu {filho}"
    # muitos cruzamentos e mutacoes aleatorios: sempre permutacoes validas
    validos = list(range(1, prob.n + 1))
    for _ in range(500):
        p1, p2 = prob.solucao_aleatoria(), prob.solucao_aleatoria()
        f = prob.cruzamento(p1, p2)
        assert sorted(f) == validos, "cruzamento gerou permutacao invalida"
        m = prob.mutacao(f)
        assert sorted(m) == validos and m != f, "mutacao invalida"
    print("OK  cruzamento OX e mutacao geram permutacoes validas")


if __name__ == "__main__":
    teste_arredondamento()
    teste_contador()
    teste_semente()
    teste_otimo_pequena()
    teste_identidade_pequena()
    teste_figura3()
    teste_tamanhos()
    teste_incremental()
    teste_poda()
    teste_orcamento_e_registro()
    teste_cruzamento_mutacao()
    print("\nTodos os testes passaram.")

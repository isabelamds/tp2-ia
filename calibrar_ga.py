"""
calibrar_ga.py - Ajuste dos parametros do Algoritmo Genetico.

Etapa 1: testa a grade inteira (operador de mutacao, N, k do torneio, Pm)
         com 3 execucoes por combinacao.
Etapa 2: refaz as 4 melhores da etapa 1 com 10 execucoes cada, porque 3
         execucoes deixam muito ruido; a escolhida e a de menor mediana aqui.

Usa sementes 1000+, diferentes das dos experimentos finais. Mostra tambem a
diversidade final (cromossomos distintos), para diagnosticar convergencia prematura.
"""

import csv
import itertools
import os
import statistics
import sys

from problema import ProblemaManutencao
from ga import algoritmo_genetico

INSTANCIAS = {
    "pequena": ("instancias/manutencao_pequena.txt", 50_000),
    "media": ("instancias/manutencao_media.txt", 150_000),
    "grande": ("instancias/manutencao_grande.txt", 300_000),
}
GRADE = {
    "mutacao": ["swap", "insertion"],   # operador de mutacao (swap = visto em aula)
    "N": [50, 100],
    "k_torneio": [3, 5],
    "Pm": [0.2, 0.4],
}
FIXOS = {"Pc": 0.9, "elite": 2}
EXECUCOES = 3              # etapa 1: grade inteira, poucas execucoes
TOP = 4                    # etapa 2: as TOP melhores da etapa 1...
EXECUCOES_CONFIRMACAO = 10 # ...sao refeitas com mais execucoes


def avaliar(caminho, B, mutacao, params, sementes):
    """Roda o GA com os parametros dados e devolve (custos, distintos no fim)."""
    prob = ProblemaManutencao(caminho, vizinhanca=mutacao, incremental=False)
    hs, distintos = [], []
    for semente in sementes:
        prob.reiniciar(semente=semente, orcamento=B)
        extras = algoritmo_genetico(prob, **params)
        hs.append(prob.melhor_h)
        distintos.append(extras["ger_distintos"][-1])
    return hs, distintos


def main(nome="media", so_confirmacao=False):
    caminho, B = INSTANCIAS[nome]
    os.makedirs("resultados", exist_ok=True)
    chaves = list(GRADE)
    arquivo = f"resultados/calibracao_ga_{nome}.csv"
    if so_confirmacao:
        with open(arquivo) as f:
            linhas = [dict(l, mediana_h=float(l["mediana_h"])) for l in csv.DictReader(f)]
        confirmar(nome, caminho, B, linhas, chaves)
        return
    linhas = []
    for valores in itertools.product(*GRADE.values()):
        params = dict(zip(chaves, valores), **FIXOS)
        mutacao = params.pop("mutacao")
        # a mutacao aplica um movimento aleatorio da vizinhanca do problema
        prob = ProblemaManutencao(caminho, vizinhanca=mutacao, incremental=False)
        hs, distintos = [], []
        for r in range(EXECUCOES):
            prob.reiniciar(semente=1000 + r, orcamento=B)
            extras = algoritmo_genetico(prob, **params)
            hs.append(prob.melhor_h)
            distintos.append(extras["ger_distintos"][-1])
        linha = {"mutacao": mutacao, **params, "mediana_h": statistics.median(hs), "valores": hs,
                 "distintos_final": statistics.median(distintos)}
        linhas.append(linha)
        print(f"{nome}: " + " ".join(f"{k}={linha[k]}" for k in chaves) +
              f"  mediana h = {linha['mediana_h']}  distintos no fim = {linha['distintos_final']}")

    with open(arquivo, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)
    melhor = min(linhas, key=lambda l: l["mediana_h"])
    print("Melhor na etapa 1:", {k: melhor[k] for k in chaves}, "mediana", melhor["mediana_h"])
    confirmar(nome, caminho, B, linhas, chaves)


def confirmar(nome, caminho, B, linhas, chaves):
    """Etapa 2: refaz as TOP melhores combinacoes com mais execucoes."""
    print(f"\nEtapa 2: {TOP} melhores com {EXECUCOES_CONFIRMACAO} execucoes cada")
    melhores = sorted(linhas, key=lambda l: l["mediana_h"])[:TOP]
    sementes = [1000 + r for r in range(EXECUCOES_CONFIRMACAO)]
    resultado = []
    for l in melhores:
        params = {k: int(l[k]) if k in ("N", "k_torneio") else float(l[k])
                  for k in chaves if k != "mutacao"}
        params.update(FIXOS)
        hs, dist = avaliar(caminho, B, l["mutacao"], params, sementes)
        linha = {"mutacao": l["mutacao"], **params,
                 "mediana_h": statistics.median(hs), "media_h": round(statistics.mean(hs)),
                 "valores": hs, "distintos_final": statistics.median(dist)}
        resultado.append(linha)
        print(f"{nome}: " + " ".join(f"{k}={linha[k]}" for k in chaves) +
              f"  mediana h = {linha['mediana_h']}  media = {linha['media_h']}")
    with open(f"resultados/calibracao_ga_{nome}_etapa2.csv", "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=resultado[0].keys())
        escritor.writeheader()
        escritor.writerows(resultado)
    melhor = min(resultado, key=lambda l: l["mediana_h"])
    print("Escolhida:", {k: melhor[k] for k in chaves}, "mediana", melhor["mediana_h"])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "media", so_confirmacao="--so-confirmacao" in sys.argv)

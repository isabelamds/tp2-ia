"""
calibrar_sa.py - Calibracao das temperaturas do Simulated Annealing.
1. Medimos pioras tipicas em dois momentos:
   - no COMECO da busca (solucoes aleatorias): delta_inicio = mediana das pioras
   - no FIM da busca (otimos locais):          delta_fim    = percentil 10 das pioras
2. Escolha das temperaturas pela probabilidade de aceitar essas pioras:
   - T0 tal que uma piora tipica do comeco seja aceita com probabilidade chi0
       e^(-delta_inicio / T0) = chi0   =>   T0 = -delta_inicio / ln(chi0)
   - Tf tal que ate uma piora pequena do fim seja aceita so com probabilidade chif
       e^(-delta_fim / Tf) = chif      =>   Tf = -delta_fim / ln(chif)

A calibracao usa sementes diferentes das dos experimentos finais (1000+),
para nao "treinar" nos mesmos sorteios em que o algoritmo sera avaliado.
"""

import csv
import math
import os
import statistics
import sys

from problema import ProblemaManutencao
from sa import simulated_annealing

INSTANCIAS = {
    "pequena": ("instancias/manutencao_pequena.txt", 50_000),
    "media": ("instancias/manutencao_media.txt", 150_000),
    "grande": ("instancias/manutencao_grande.txt", 300_000),
}
CHI0 = [0.3, 0.6, 0.9]
CHIF = [0.01, 0.001, 1e-4, 1e-6, 1e-9, 1e-12]
EXECUCOES = 3


def medir_deltas(caminho, vizinhanca, amostras=3, semente=999):
    prob = ProblemaManutencao(caminho, vizinhanca=vizinhanca)
    prob.reiniciar(semente)
    inicio, fim = [], []
    for _ in range(amostras):
        s = prob.solucao_aleatoria()
        hs = prob.custo(s)
        for mov in prob.movimentos()[:300]:
            v = prob.custo(prob.aplicar(s, mov))
            if v > hs:
                inicio.append(v - hs)
        # desce ate um otimo local (first improvement)
        melhorou = True
        while melhorou:
            melhorou = False
            for mov in prob.movimentos():
                v = prob.avaliar_movimento(s, mov, limite=hs)
                if v < hs:
                    s, hs, melhorou = prob.aplicar(s, mov), v, True
                    break
        for mov in prob.movimentos()[:300]:
            v = prob.custo(prob.aplicar(s, mov))
            if v > hs:
                fim.append(v - hs)
    fim.sort()
    return statistics.median(inicio), fim[len(fim) // 10]


def main(vizinhanca="insertion"):
    os.makedirs("resultados", exist_ok=True)
    linhas_deltas, linhas_grade = [], []

    for nome, (caminho, B) in INSTANCIAS.items():
        d_ini, d_fim = medir_deltas(caminho, vizinhanca)
        linhas_deltas.append({"instancia": nome, "vizinhanca": vizinhanca,
                              "delta_inicio": d_ini, "delta_fim": d_fim})
        print(f"\n{nome}: piora tipica no comeco = {d_ini:.0f}, no fim = {d_fim:.0f}")

        prob = ProblemaManutencao(caminho, vizinhanca=vizinhanca, incremental=False)
        for chi0 in CHI0:
            for chif in CHIF:
                T0 = -d_ini / math.log(chi0)
                Tf = -d_fim / math.log(chif)
                valores = []
                for r in range(EXECUCOES):
                    prob.reiniciar(semente=1000 + r, orcamento=B)
                    simulated_annealing(prob, T0, Tf)
                    valores.append(prob.melhor_h)
                med = statistics.median(valores)
                linhas_grade.append({"instancia": nome, "vizinhanca": vizinhanca,
                                     "chi0": chi0, "chif": chif,
                                     "T0": round(T0, 3), "Tf": round(Tf, 3),
                                     "mediana_h": med, "valores": valores})
                print(f"  chi0={chi0:<4} chif={chif:<6} T0={T0:9.1f} Tf={Tf:8.2f}"
                      f"  mediana h = {med}")

    sufixo = "" if vizinhanca == "insertion" else f"_{vizinhanca}"
    for arquivo, linhas in [(f"resultados/calibracao_sa_deltas{sufixo}.csv", linhas_deltas),
                            (f"resultados/calibracao_sa_grade{sufixo}.csv", linhas_grade)]:
        with open(arquivo, "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
            escritor.writeheader()
            escritor.writerows(linhas)

    print("\nMelhor combinacao por instancia:")
    for nome in INSTANCIAS:
        melhor = min((l for l in linhas_grade if l["instancia"] == nome),
                     key=lambda l: l["mediana_h"])
        print(f"  {nome:8} chi0={melhor['chi0']} chif={melhor['chif']} "
              f"T0={melhor['T0']} Tf={melhor['Tf']} mediana={melhor['mediana_h']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "insertion")

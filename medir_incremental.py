"""
medir_incremental.py - Mede o ganho de tempo da avaliacao incremental.
Para cada instancia e vizinhanca, varre a vizinhanca COMPLETA de algumas
solucoes aleatorias de tres formas:
  completa     -> monta cada sucessor e calcula h do zero
  incremental  -> reaproveita o prefixo que nao muda (sem poda)
  incr.+poda   -> incremental, parando quando o sucessor ja nao melhora
O numero de avaliacoes e o mesmo nos tres casos; so o tempo muda.
"""

import csv
import math
import os
import time

from problema import ProblemaManutencao, VIZINHANCAS

INSTANCIAS = {
    "pequena": "instancias/manutencao_pequena.txt",
    "media": "instancias/manutencao_media.txt",
    "grande": "instancias/manutencao_grande.txt",
}
SOLUCOES = 5   # quantas solucoes aleatorias varrer 


def varrer(prob, solucoes, com_poda):
    inicio = time.perf_counter()
    for s in solucoes:
        hs = prob.custo(s)
        prob.melhor_h = -math.inf        
        limite = hs if com_poda else math.inf
        for mov in prob.movimentos():
            prob.avaliar_movimento(s, mov, limite)
    return time.perf_counter() - inicio


def main():
    os.makedirs("resultados", exist_ok=True)
    linhas = []
    print(f"{'instancia':>9} {'vizinhanca':>10} {'avaliacoes':>10} "
          f"{'completa(s)':>11} {'incr(s)':>8} {'incr+poda(s)':>12} "
          f"{'ganho':>6} {'ganho c/ poda':>13}")
    for nome_inst, caminho in INSTANCIAS.items():
        for viz in VIZINHANCAS:
            prob = ProblemaManutencao(caminho, vizinhanca=viz)
            prob.reiniciar(semente=123)
            solucoes = [prob.solucao_aleatoria() for _ in range(SOLUCOES)]
            avals = SOLUCOES * prob.tamanho_vizinhanca()

            prob.incremental = False
            t_comp = varrer(prob, solucoes, com_poda=False)
            prob.incremental = True
            t_inc = varrer(prob, solucoes, com_poda=False)
            t_poda = varrer(prob, solucoes, com_poda=True)

            linhas.append({
                "instancia": nome_inst, "vizinhanca": viz, "avaliacoes": avals,
                "tempo_completa_s": round(t_comp, 4),
                "tempo_incremental_s": round(t_inc, 4),
                "tempo_incremental_poda_s": round(t_poda, 4),
                "ganho": round(t_comp / t_inc, 2),
                "ganho_poda": round(t_comp / t_poda, 2),
            })
            print(f"{nome_inst:>9} {viz:>10} {avals:>10} {t_comp:>11.3f} "
                  f"{t_inc:>8.3f} {t_poda:>12.3f} {t_comp / t_inc:>5.1f}x "
                  f"{t_comp / t_poda:>12.1f}x")

    with open("resultados/tempo_incremental.csv", "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)
    print("\nSalvo em resultados/tempo_incremental.csv")


if __name__ == "__main__":
    main()

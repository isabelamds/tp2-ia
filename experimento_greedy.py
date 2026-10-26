"""
Tarefa T2: comparacao das vizinhancas do Greedy Descent.

Para cada instancia, roda o greedy descent com random restart nas tres
vizinhancas (swap, insertion, 2opt) com o mesmo orcamento e as mesmas sementes,
e mede o custo por movimento da avaliacao incremental. Responde as perguntas do T2:
  - qual vizinhanca conduz a melhores otimos locais?
  - qual e a mais barata por movimento com avaliacao incremental?
  - quanto a avaliacao incremental ganha sobre a avaliacao completa?

Entrada : instancias/manutencao_pequena.txt, manutencao_media.txt, manutencao_grande.txt
Saida   : resultados/
    t2_execucoes.csv       uma linha por (instancia, estrategia, vizinhanca, semente)
    t2_resumo.csv          media, desvio, mediana, minimo, gap e contagens por grupo
    t2_custo_movimento.csv ms por movimento (completa x incremental) e ganho de tempo
    t2_testes.csv          testes estatisticos par a par entre as vizinhancas
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from problema import ProblemaManutencao, VIZINHANCAS
from greedy import ESTRATEGIAS_GREEDY, greedy_descent

try:
    from scipy.stats import mannwhitneyu, wilcoxon
except ImportError:
    raise SystemExit("Este script precisa do scipy: python -m pip install scipy")

ALPHA = 0.05
DIR_INSTANCIAS = Path("instancias")
NOMES_INSTANCIAS = ("manutencao_pequena.txt", "manutencao_media.txt", "manutencao_grande.txt")

# n -> (orcamento de avaliacoes, execucoes independentes): Secao 3 do enunciado.
PROTOCOLO = {12: (50_000, 30), 50: (150_000, 30), 150: (300_000, 10)}
# n -> valor de referencia de h (Tabela 1): n=12 e o otimo global; os demais, melhor conhecido.
REFERENCIA = {12: 1875, 50: 16332, 150: 60417}

# Modos de avaliar um movimento no benchmark de custo.
MODOS = ("completa", "incremental_sem_poda", "incremental_com_poda")


# ----------------------------------------------------------------------------
# Utilitarios
# ----------------------------------------------------------------------------
def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(c for linha in linhas for c in linha))
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def finitos(valores):
    return [v for v in valores if isinstance(v, (int, float)) and math.isfinite(v)]


def media(valores):
    return statistics.mean(valores) if valores else math.nan


def mediana(valores):
    return statistics.median(valores) if valores else math.nan


def desvio(valores):
    return statistics.stdev(valores) if len(valores) > 1 else 0.0


def holm(p_valores: list[float]) -> list[float]:
    """p-valores ajustados pelo procedimento step-down de Holm."""
    m = len(p_valores)
    ajustados = [0.0] * m
    anterior = 0.0
    for pos, (idx, p) in enumerate(sorted(enumerate(p_valores), key=lambda it: it[1])):
        anterior = max(anterior, min(1.0, (m - pos) * p))
        ajustados[idx] = anterior
    return ajustados


# ----------------------------------------------------------------------------
# Execucoes do Greedy Descent
# ----------------------------------------------------------------------------
def rodar_execucoes(problema, nome, estrategias, orcamento, repeticoes, seed_base):
    n = problema.n
    ref = REFERENCIA.get(n)
    problema.incremental = True
    linhas = []
    for estrategia in estrategias:
        for i in range(repeticoes):
            seed = seed_base + i  # mesma semente para as tres vizinhancas
            for viz in VIZINHANCAS:
                problema.definir_vizinhanca(viz)
                inicio = time.perf_counter()
                r = greedy_descent(problema, estrategia=estrategia,
                                   semente=seed, orcamento=orcamento)
                tempo = time.perf_counter() - inicio

                melhor = r["melhor_h"]
                hs_locais = [o["h"] for o in r["otimos_locais_distintos"]]
                linhas.append({
                    "instancia": nome,
                    "n": n,
                    "estrategia": estrategia,
                    "vizinhanca": viz,
                    "seed": seed,
                    "orcamento": orcamento,
                    "avaliacoes": r["avaliacoes"],
                    "melhor_h": melhor,
                    "h_referencia": ref if ref else "",
                    "gap_relativo": (melhor - ref) / ref if ref else "",
                    "atingiu_referencia": melhor <= ref if ref else "",
                    "tempo_s": tempo,
                    "reinicios": r["reinicios"],
                    "passos_melhoria": r["passos_melhoria"],
                    "otimos_locais_distintos": r["quantidade_otimos_locais_distintos"],
                    # media de h sobre os otimos locais distintos desta execucao (nan se
                    # o orcamento acabou antes de qualquer descida terminar)
                    "media_h_otimos_locais": media(hs_locais),
                    "melhor_solucao": " ".join(map(str, r["melhor_s"])),
                })
    return linhas


# ----------------------------------------------------------------------------
# Benchmark do custo por movimento (+ teste automatico incremental x completa)
# ----------------------------------------------------------------------------
def _preparar(problema, seed, s):
    """Zera contador e incumbente e registra h(s), como o greedy faz ao iniciar."""
    problema.reiniciar(semente=seed, orcamento=None)
    return problema.h(s)


def benchmark_movimentos(problema, nome, repeticoes, seed_base):
    n = problema.n
    linhas = []
    for viz in VIZINHANCAS:
        problema.definir_vizinhanca(viz)
        for r in range(repeticoes):
            seed = seed_base + r  # mesma permutacao para as tres vizinhancas
            problema.reiniciar(semente=seed, orcamento=None)
            s = problema.solucao_aleatoria()
            movs = problema.movimentos()  # mesma ordem em todos os modos

            # Teste automatico: a avaliacao incremental (sem poda) deve dar o mesmo
            # valor que a avaliacao completa para TODO movimento.
            problema.incremental = True
            _preparar(problema, seed, s)
            for mov in movs:
                esperado = problema.custo(problema.aplicar(s, mov))
                obtido = problema.avaliar_movimento(s, mov)
                if obtido != esperado:
                    raise AssertionError(
                        f"Incremental diverge da completa: instancia={nome} viz={viz} "
                        f"seed={seed} mov={mov} incremental={obtido} completa={esperado}"
                    )

            for modo in MODOS:
                problema.incremental = modo != "completa"
                h_base = _preparar(problema, seed, s)
                poda = modo == "incremental_com_poda"
                limite = h_base if poda else math.inf
                inicio = time.perf_counter()
                for mov in movs:
                    v = problema.avaliar_movimento(s, mov, limite=limite)
                    if poda and v < limite:  # o best improvement tambem aperta o limite
                        limite = v
                duracao = time.perf_counter() - inicio
                linhas.append({
                    "instancia": nome, "n": n, "vizinhanca": viz, "modo": modo,
                    "seed": seed, "movimentos": len(movs),
                    "tempo_total_s": duracao,
                    "tempo_ms_por_movimento": 1000.0 * duracao / len(movs),
                })
    problema.incremental = True
    return linhas


# ----------------------------------------------------------------------------
# Resumos
# ----------------------------------------------------------------------------
def resumir_execucoes(execucoes):
    grupos = defaultdict(list)
    for r in execucoes:
        grupos[(r["instancia"], r["n"], r["estrategia"], r["vizinhanca"])].append(r)
    resumo = []
    for (inst, n, est, viz), rows in grupos.items():
        hs = [r["melhor_h"] for r in rows]
        ref = REFERENCIA.get(n)
        locais = finitos([r["media_h_otimos_locais"] for r in rows])
        resumo.append({
            "instancia": inst, "n": n, "estrategia": est, "vizinhanca": viz,
            "execucoes": len(rows),
            "media_h": statistics.mean(hs), "desvio_h": desvio(hs),
            "mediana_h": statistics.median(hs), "minimo_h": min(hs), "maximo_h": max(hs),
            "h_referencia": ref if ref else "",
            "gap_medio": (statistics.mean(hs) - ref) / ref if ref else "",
            "execucoes_que_atingiram_referencia":
                sum(r["atingiu_referencia"] for r in rows) if ref else "",
            "media_tempo_s": statistics.mean(r["tempo_s"] for r in rows),
            "media_avaliacoes": statistics.mean(r["avaliacoes"] for r in rows),
            "media_reinicios": statistics.mean(r["reinicios"] for r in rows),
            "media_passos_melhoria": statistics.mean(r["passos_melhoria"] for r in rows),
            "media_otimos_locais_distintos":
                statistics.mean(r["otimos_locais_distintos"] for r in rows),
            "execucoes_com_otimo_local": len(locais),
            "media_h_otimos_locais": media(locais),
        })
    return resumo


def resumir_benchmark(bench):
    grupos = defaultdict(list)
    for b in bench:
        grupos[(b["instancia"], b["n"], b["vizinhanca"], b["modo"])].append(b)
    medias = {k: statistics.mean(x["tempo_ms_por_movimento"] for x in v)
              for k, v in grupos.items()}
    resumo = []
    for (inst, n, viz, modo), rows in grupos.items():
        ts = [x["tempo_ms_por_movimento"] for x in rows]
        base = medias[(inst, n, viz, "completa")]
        resumo.append({
            "instancia": inst, "n": n, "vizinhanca": viz, "modo": modo,
            "repeticoes": len(rows), "movimentos_por_varredura": rows[0]["movimentos"],
            "media_ms_por_movimento": statistics.mean(ts),
            "desvio_ms_por_movimento": desvio(ts),
            "mediana_ms_por_movimento": statistics.median(ts),
            "ganho_vs_completa": base / statistics.mean(ts),  # >1 = mais rapido que a completa
            "concordancia_incremental_completa": True,  # um desvio levanta AssertionError
        })
    return resumo


# ----------------------------------------------------------------------------
# Testes estatisticos (comparacao par a par entre vizinhancas)
# ----------------------------------------------------------------------------
def _testar(va, vb, pareado):
    if len(set(va) | set(vb)) == 1 or (pareado and all(x == y for x, y in zip(va, vb))):
        return 0.0, 1.0, "todos os valores iguais"
    try:
        if pareado:
            stat, p = wilcoxon(va, vb, alternative="two-sided")
        else:
            stat, p = mannwhitneyu(va, vb, alternative="two-sided")
    except ValueError as exc:
        return math.nan, 1.0, f"teste nao calculavel; p=1 por conservadorismo ({exc})"
    if not math.isfinite(p):
        return float(stat), 1.0, "p-valor nao finito; tratado como 1"
    return float(stat), float(p), ""


def comparar_vizinhancas(inst, n, estrategia, metrica, grupos, pareado):
    """Testa as 3 duplas de vizinhancas e aplica Holm aos 3 p-valores.

    pareado=False -> Mann-Whitney U (execucoes independentes).
    pareado=True  -> Wilcoxon pareado (mesma permutacao medida nas tres vizinhancas).
    """
    linhas = []
    for a, b in combinations(VIZINHANCAS, 2):
        va, vb = grupos[a], grupos[b]
        linha = {
            "instancia": inst, "n": n, "estrategia": estrategia, "metrica": metrica,
            "teste": "Wilcoxon pareado" if pareado else "Mann-Whitney U",
            "comparacao": f"{a} vs {b}",
            "mediana_A": mediana(va), "mediana_B": mediana(vb),
            "estatistica": "", "p_valor": "", "p_valor_holm": "",
            "significativo": "", "observacao": "",
        }
        if len(va) < 3 or len(vb) < 3:
            linha["observacao"] = f"amostra insuficiente (n_A={len(va)}, n_B={len(vb)})"
        else:
            linha["estatistica"], linha["p_valor"], linha["observacao"] = _testar(va, vb, pareado)
        linhas.append(linha)

    calculados = [l for l in linhas if l["p_valor"] != ""]
    for linha, p_adj in zip(calculados, holm([l["p_valor"] for l in calculados])):
        linha["p_valor_holm"] = p_adj
        linha["significativo"] = p_adj < ALPHA
    return linhas


def montar_testes(execucoes, bench):
    testes = []
    for inst, n, est in sorted({(r["instancia"], r["n"], r["estrategia"]) for r in execucoes}):
        rows = [r for r in execucoes
                if (r["instancia"], r["n"], r["estrategia"]) == (inst, n, est)]
        for metrica in ("melhor_h", "media_h_otimos_locais"):
            grupos = {v: finitos([r[metrica] for r in rows if r["vizinhanca"] == v])
                      for v in VIZINHANCAS}
            testes += comparar_vizinhancas(inst, n, est, metrica, grupos, pareado=False)

    for inst, n in sorted({(b["instancia"], b["n"]) for b in bench}):
        for modo in ("incremental_sem_poda", "incremental_com_poda"):
            grupos = {v: [b["tempo_ms_por_movimento"] for b in bench
                          if (b["instancia"], b["vizinhanca"], b["modo"]) == (inst, v, modo)]
                      for v in VIZINHANCAS}
            testes += comparar_vizinhancas(inst, n, "", f"tempo_ms_por_movimento[{modo}]",
                                           grupos, pareado=True)
    return testes


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instancias", nargs="*",
                    help=f"Arquivos de instancia. Padrao: {DIR_INSTANCIAS}/{{pequena,media,grande}}.")
    ap.add_argument("--estrategia", nargs="+", choices=ESTRATEGIAS_GREEDY, default=["best"],
                    help="Uma ou mais estrategias do greedy (padrao: best).")
    ap.add_argument("--orcamento", type=int, default=None,
                    help="Substitui o orcamento do protocolo em todas as instancias.")
    ap.add_argument("--repeticoes", type=int, default=None,
                    help="Substitui o numero de execucoes do protocolo (minimo 3).")
    ap.add_argument("--bench-repeticoes", type=int, default=10,
                    help="Permutacoes aleatorias usadas no benchmark por movimento (padrao: 10).")
    ap.add_argument("--seed-base", type=int, default=2026,
                    help="Primeira semente; execucao i usa seed-base + i.")
    ap.add_argument("--saida", default="resultados", help="Pasta dos CSVs (padrao: resultados).")
    args = ap.parse_args()

    if args.orcamento is not None and args.orcamento <= 0:
        ap.error("--orcamento deve ser positivo")
    if args.repeticoes is not None and args.repeticoes < 3:
        ap.error("--repeticoes deve ser pelo menos 3 para os testes estatisticos")
    if args.bench_repeticoes < 3:
        ap.error("--bench-repeticoes deve ser pelo menos 3 para os testes estatisticos")

    caminhos = [Path(c) for c in args.instancias] or [DIR_INSTANCIAS / nome
                                                      for nome in NOMES_INSTANCIAS]
    for caminho in caminhos:
        if not caminho.is_file():
            raise FileNotFoundError(f"Nao encontrei '{caminho}'.")
    estrategias = list(dict.fromkeys(args.estrategia))

    execucoes, bench = [], []
    for caminho in caminhos:
        nome = caminho.stem
        problema = ProblemaManutencao(str(caminho), incremental=True)
        orc_padrao, rep_padrao = PROTOCOLO.get(problema.n, (None, None))
        orcamento = args.orcamento or orc_padrao
        repeticoes = args.repeticoes or rep_padrao
        if orcamento is None or repeticoes is None:
            ap.error(f"n={problema.n} ({nome}) esta fora do protocolo; "
                     "informe --orcamento e --repeticoes")
        print(f"\n{nome}: n={problema.n}, orcamento={orcamento:,}, execucoes={repeticoes}, "
              f"estrategias={estrategias}")

        execucoes += rodar_execucoes(problema, nome, estrategias, orcamento,
                                     repeticoes, args.seed_base)
        print(f"  benchmark por movimento ({args.bench_repeticoes} permutacoes)...")
        bench += benchmark_movimentos(problema, nome, args.bench_repeticoes,
                                      args.seed_base + 100_000)

    resumo = resumir_execucoes(execucoes)
    saida = Path(args.saida)
    gravar_csv(saida / "t2_execucoes.csv", execucoes)
    gravar_csv(saida / "t2_resumo.csv", resumo)
    gravar_csv(saida / "t2_custo_movimento.csv", resumir_benchmark(bench))
    gravar_csv(saida / "t2_testes.csv", montar_testes(execucoes, bench))

    print("\nResumo (media de melhor_h | passos de melhoria | otimos locais distintos):")
    for r in resumo:
        print(f"  {r['instancia']:<20} {r['estrategia']:<5} {r['vizinhanca']:<9} "
              f"h={r['media_h']:>10.1f} +- {r['desvio_h']:<8.1f} "
              f"passos={r['media_passos_melhoria']:>8.1f} "
              f"otimos={r['media_otimos_locais_distintos']:>7.1f}")
    print(f"\nCSVs gravados em {saida}/")


if __name__ == "__main__":
    main()

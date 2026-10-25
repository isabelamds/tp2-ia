"""
analise.py - Tabelas, testes estatisticos e graficos do relatorio (secoes 3 e 5).

Gera:       resultados/analise/
              tabela_principal.csv / .md    media, desvio, minimo, mediana, gap, tempo
              testes_mann_whitney.csv / .md comparacoes par a par (p-valores)
              boxplots.png                  um painel por instancia
              convergencia_<inst>.png       melhor h x avaliacoes (mediana das execucoes)
              sa_aceitacao.png              taxa de aceitacao de pioras e temperatura
              ga_diversidade.png            cromossomos distintos e distancia media
              ga_operadores.csv / .png      sucesso do cruzamento x da mutacao

"""

import csv
import glob
import json
import os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")                  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import warnings

# Antes da primeira geracao do GA nao ha media da populacao (valores vazios):
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ENTRADA_CSV = "resultados/execucoes.csv"
ENTRADA_JSON = "resultados/execucoes"
SAIDA = "resultados/analise"

INSTANCIAS = ["pequena", "media", "grande"]
ORCAMENTO = {"pequena": 50_000, "media": 150_000, "grande": 300_000}
TAMANHO = {"pequena": 12, "media": 50, "grande": 150}
OTIMO_PEQUENA = 1875
REFERENCIA = {"pequena": 1875, "media": 16332, "grande": 60417}

# Cor fixa por algoritmo (a cor segue o algoritmo, nunca a posicao no grafico).
CORES_FIXAS = {"sa": "#2a78d6", "ga": "#eb6834", "ga_sem_cruzamento": "#1baf7a"}
CORES_EXTRAS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NOMES = {"sa": "SA", "ga": "GA", "ga_sem_cruzamento": "GA sem cruzamento"}

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"
FUNDO = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2, "text.color": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 2,
})


# Leitura
def ler_execucoes():
    with open(ENTRADA_CSV) as f:
        linhas = list(csv.DictReader(f))
    for l in linhas:
        l["melhor_h"] = int(l["melhor_h"])
        l["avaliacoes"] = int(l["avaliacoes"])
        l["tempo_s"] = float(l["tempo_s"])
    return linhas


def ler_detalhes(config, inst):
    arquivos = sorted(glob.glob(os.path.join(ENTRADA_JSON, f"{config}_{inst}_*.json")))
    detalhes = []
    for arq in arquivos:
        with open(arq) as f:
            detalhes.append(json.load(f))
    return detalhes


def ordem_configs(linhas):
    """Ordem estavel: SA, GA, GA sem cruzamento, depois o resto em ordem alfabetica."""
    presentes = {l["config"] for l in linhas}
    fixas = [c for c in CORES_FIXAS if c in presentes]
    return fixas + sorted(presentes - set(fixas))


def cor(config, ordem):
    if config in CORES_FIXAS:
        return CORES_FIXAS[config]
    extras = [c for c in ordem if c not in CORES_FIXAS]
    return CORES_EXTRAS[extras.index(config) % len(CORES_EXTRAS)]


def nome(config):
    return NOMES.get(config, config)


# 1. Tabela principal
def tabela_principal(linhas, ordem):
    grupos = defaultdict(list)
    for l in linhas:
        grupos[(l["config"], l["instancia"])].append(l)

    tabela = []
    for inst in INSTANCIAS:
        for config in ordem:
            g = grupos.get((config, inst))
            if not g:
                continue
            hs = np.array([l["melhor_h"] for l in g])
            linha = {
                "instancia": inst, "algoritmo": nome(config), "config": config,
                "execucoes": len(hs),
                "media": round(hs.mean(), 1),
                "desvio": round(hs.std(ddof=1), 1) if len(hs) > 1 else 0.0,
                "minimo": int(hs.min()),
                "mediana": float(np.median(hs)),
                "referencia": REFERENCIA[inst],
                "gap_medio_ref_%": round(100 * (hs.mean() - REFERENCIA[inst]) / REFERENCIA[inst], 2),
                "gap_minimo_ref_%": round(100 * (hs.min() - REFERENCIA[inst]) / REFERENCIA[inst], 2),
                "atingiu_otimo": int((hs == OTIMO_PEQUENA).sum()) if inst == "pequena" else "",
                "avaliacoes": int(np.mean([l["avaliacoes"] for l in g])),
                "tempo_medio_s": round(np.mean([l["tempo_s"] for l in g]), 2),
                "tempo_por_mil_aval_ms": round(
                    1000 * np.mean([l["tempo_s"] for l in g]) / np.mean([l["avaliacoes"] for l in g]) * 1000, 2),
            }
            tabela.append(linha)

    salvar_csv(tabela, "tabela_principal.csv")
    with open(os.path.join(SAIDA, "tabela_principal.md"), "w") as f:
        f.write("| Instância | Algoritmo | Média ± DP | Mínimo | Mediana | Gap médio | "
                "Ótimo atingido | Avaliações | Tempo médio (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for t in tabela:
            otimo = f"{t['atingiu_otimo']}/{t['execucoes']}" if t["instancia"] == "pequena" else "—"
            f.write(f"| {t['instancia']} | {t['algoritmo']} | {t['media']:.0f} ± {t['desvio']:.0f} | "
                    f"{t['minimo']} | {t['mediana']:.0f} | {t['gap_medio_ref_%']:.1f}% | {otimo} | "
                    f"{t['avaliacoes']} | {t['tempo_medio_s']:.2f} |\n")
        f.write("\nGap = (h − referência) / referência. Referência: ótimo global na pequena "
                "(1875); melhor valor conhecido dos professores na média (16332) e na grande (60417).\n")
    return tabela


# 2. Testes de Mann-Whitney
def testes(linhas, ordem):
    por = defaultdict(list)
    for l in linhas:
        por[(l["config"], l["instancia"])].append(l["melhor_h"])
    resultado = []
    for inst in INSTANCIAS:
        configs = [c for c in ordem if (c, inst) in por]
        for a, b in combinations(configs, 2):
            xa, xb = por[(a, inst)], por[(b, inst)]
            if len(set(xa) | set(xb)) == 1:
                p = 1.0          
                u = len(xa) * len(xb) / 2
            else:
                u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            ma, mb = np.median(xa), np.median(xb)
            melhor = nome(a) if ma < mb else nome(b) if mb < ma else "empate"
            resultado.append({
                "instancia": inst, "A": nome(a), "B": nome(b),
                "mediana_A": float(ma), "mediana_B": float(mb),
                "U": float(u), "p_valor": float(p),
                "significativo_5%": "sim" if p < 0.05 else "nao",
                "menor_mediana": melhor,
            })
    salvar_csv(resultado, "testes_mann_whitney.csv")
    with open(os.path.join(SAIDA, "testes_mann_whitney.md"), "w") as f:
        f.write("| Instância | A | B | Mediana A | Mediana B | p-valor | Significativo (5%) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in resultado:
            f.write(f"| {r['instancia']} | {r['A']} | {r['B']} | {r['mediana_A']:.0f} | "
                    f"{r['mediana_B']:.0f} | {formatar_p(r['p_valor'])} | {r['significativo_5%']} |\n")
        f.write("\nTeste de Mann–Whitney bilateral sobre o melhor h de cada execução.\n")
    return resultado


def formatar_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


# 3. Boxplots
def boxplots(linhas, ordem):
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, inst in zip(eixos, INSTANCIAS):
        configs = [c for c in ordem if any(l["config"] == c and l["instancia"] == inst for l in linhas)]
        dados = [[l["melhor_h"] for l in linhas if l["config"] == c and l["instancia"] == inst]
                 for c in configs]
        caixa = ax.boxplot(dados, patch_artist=True, widths=0.5,
                           medianprops={"color": TINTA, "linewidth": 2},
                           whiskerprops={"color": TINTA_2}, capprops={"color": TINTA_2},
                           flierprops={"marker": "o", "markersize": 4,
                                       "markerfacecolor": TINTA_2, "markeredgewidth": 0})
        for patch, c in zip(caixa["boxes"], configs):
            patch.set_facecolor(cor(c, ordem))
            patch.set_alpha(0.85)
            patch.set_edgecolor(FUNDO)
        # pontos individuais (com leve espalhamento horizontal)
        rng = np.random.default_rng(0)
        for k, d in enumerate(dados, 1):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=TINTA,
                       alpha=0.45, zorder=3, linewidths=0)
        rotulo_ref = "ótimo global" if inst == "pequena" else "melhor valor conhecido"
        ax.axhline(REFERENCIA[inst], color=TINTA_2, linestyle="--", linewidth=1,
                   label=f"{rotulo_ref} ({REFERENCIA[inst]})")
        baixo, alto = ax.get_ylim()
        ax.set_ylim(baixo, alto + 0.18 * (alto - baixo))    
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticks(range(1, len(configs) + 1))
        ax.set_xticklabels([nome(c) for c in configs], rotation=15)
        ax.set_title(f"Instância {inst} (n = {TAMANHO[inst]})", loc="left")
        ax.set_ylabel("melhor h por execução")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Melhor h encontrado em cada execução", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "boxplots.png"), dpi=160)
    plt.close(fig)


# 4. Curvas de convergencia
def curva_melhor(historico, grade_x):
    """Transforma o historico [(aval, melhor_h), ...] numa curva em degraus na grade."""
    av = np.array([a for a, _ in historico])
    hv = np.array([h for _, h in historico], dtype=float)
    idx = np.searchsorted(av, grade_x, side="right") - 1
    curva = np.where(idx >= 0, hv[np.clip(idx, 0, None)], np.nan)
    return curva


def convergencia(ordem):
    for inst in INSTANCIAS:
        B = ORCAMENTO[inst]
        grade_x = np.unique(np.geomspace(1, B, 400).astype(int))
        fig, ax = plt.subplots(figsize=(8, 5.6))
        for config in ordem:
            dets = ler_detalhes(config, inst)
            if not dets:
                continue
            curvas = np.array([curva_melhor(d["historico"], grade_x) for d in dets])
            mediana = np.nanmedian(curvas, axis=0)
            c = cor(config, ordem)
            ax.plot(grade_x, mediana, color=c, label=f"{nome(config)}: melhor h (mediana)")
            # GA: tambem a media da populacao (slide 20 da aula)
            if dets[0]["extras"].get("ger_media"):
                medias = []
                for d in dets:
                    xg = np.array(d["extras"]["ger_avaliacao"])
                    yg = np.array(d["extras"]["ger_media"])
                    medias.append(np.interp(grade_x, xg, yg, left=np.nan))
                ax.plot(grade_x, np.nanmedian(np.array(medias), axis=0), color=c,
                        linestyle=":", linewidth=2,
                        label=f"{nome(config)}: média da população (mediana)")
        ax.axhline(REFERENCIA[inst], color=TINTA_2, linestyle="--", linewidth=1,
                   label="ótimo global" if inst == "pequena" else "melhor valor conhecido")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("avaliações de h (escala log)")
        ax.set_ylabel("h (escala log)")
        ax.set_title(f"Convergência – instância {inst} (orçamento {B:,} avaliações)".replace(",", "."),
                     loc="left")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(SAIDA, f"convergencia_{inst}.png"), dpi=160)
        plt.close(fig)


# 5. SA: taxa de aceitacao de pioras e temperatura
def sa_aceitacao():
    fig, eixos = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    estilos = {"pequena": ":", "media": "--", "grande": "-"}
    tinta_inst = {"pequena": "#9fc3ee", "media": "#5d9be0", "grande": "#2a78d6"}
    algum = False
    for inst in INSTANCIAS:
        dets = ler_detalhes("sa", inst)
        if not dets:
            continue
        algum = True
        B = ORCAMENTO[inst]
        x = np.array(dets[0]["extras"]["janela_avaliacao"]) / B * 100
        taxas = np.array([d["extras"]["janela_taxa_piores"] for d in dets])
        temps = np.array(dets[0]["extras"]["janela_temperatura"])
        eixos[0].plot(x, np.median(taxas, axis=0), color=tinta_inst[inst],
                      linestyle=estilos[inst], label=f"{inst}")
        eixos[1].plot(x, temps, color=tinta_inst[inst], linestyle=estilos[inst], label=inst)
    if not algum:
        plt.close(fig)
        return
    eixos[0].set_ylabel("fração de pioras aceitas")
    eixos[0].set_ylim(0, 1)
    eixos[0].set_title("SA – taxa de aceitação de movimentos piores (mediana das execuções)",
                       loc="left")
    eixos[0].annotate("≈ random walk", xy=(1, 0.95), fontsize=8, color=TINTA_2)
    eixos[0].annotate("≈ greedy descent", xy=(80, 0.05), fontsize=8, color=TINTA_2)
    eixos[0].legend(title="instância", fontsize=8)
    eixos[1].set_yscale("log")
    eixos[1].set_ylabel("temperatura T (log)")
    eixos[1].set_xlabel("progresso da execução (% do orçamento)")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "sa_aceitacao.png"), dpi=160)
    plt.close(fig)


# 6. GA: diversidade da populacao
def ga_diversidade(ordem):
    configs = [c for c in ordem if c.startswith("ga")]
    if not configs:
        return
    fig, eixos = plt.subplots(2, 3, figsize=(13, 6), sharex="col")
    for col, inst in enumerate(INSTANCIAS):
        B = ORCAMENTO[inst]
        grade_x = np.linspace(0, B, 300)
        for config in configs:
            dets = ler_detalhes(config, inst)
            if not dets:
                continue
            for linha, chave in enumerate(["ger_distintos", "ger_distancia"]):
                curvas = [np.interp(grade_x, d["extras"]["ger_avaliacao"], d["extras"][chave],
                                    right=np.nan) for d in dets]
                eixos[linha, col].plot(grade_x / 1000, np.nanmedian(np.array(curvas), axis=0),
                                       color=cor(config, ordem), label=nome(config))
        eixos[0, col].set_title(f"Instância {inst}", loc="left")
        eixos[1, col].set_xlabel("avaliações (milhares)")
    eixos[0, 0].set_ylabel("cromossomos distintos")
    eixos[1, 0].set_ylabel("distância média entre pares\n(0 = iguais, 1 = nada em comum)")
    eixos[0, 0].legend(fontsize=8)
    fig.suptitle("GA – diversidade da população (mediana das execuções)", x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "ga_diversidade.png"), dpi=160)
    plt.close(fig)


# 7. Experimento adicional: sucesso do cruzamento x da mutacao
def ga_operadores(ordem):
    """Fracao de filhos melhores que os pais, por operador, ao longo da execucao."""
    if "ga" not in ordem:
        return
    fases = 10
    tabela = []
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, inst in zip(eixos, INSTANCIAS):
        dets = ler_detalhes("ga", inst)
        if not dets or "ger_cruz_filhos" not in dets[0]["extras"]:
            continue
        B = ORCAMENTO[inst]
        soma = {k: np.zeros(fases) for k in ("cn", "cs", "mn", "ms")}
        for d in dets:
            e = d["extras"]
            fase = np.minimum((np.array(e["ger_avaliacao"]) / B * fases).astype(int), fases - 1)
            for k, chave in [("cn", "ger_cruz_filhos"), ("cs", "ger_cruz_sucessos"),
                             ("mn", "ger_mut_filhos"), ("ms", "ger_mut_sucessos")]:
                np.add.at(soma[k], fase, np.array(e[chave]))
        taxa_c = np.divide(soma["cs"], soma["cn"], out=np.full(fases, np.nan), where=soma["cn"] > 0)
        taxa_m = np.divide(soma["ms"], soma["mn"], out=np.full(fases, np.nan), where=soma["mn"] > 0)
        x = (np.arange(fases) + 0.5) * 100 / fases
        ax.plot(x, 100 * taxa_c, color=CORES_FIXAS["ga"], marker="o", markersize=5,
                label="filho de cruzamento melhor que o melhor pai")
        ax.plot(x, 100 * taxa_m, color=TINTA_2, marker="s", markersize=5, linestyle="--",
                label="filho só de mutação melhor que o pai")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"Instância {inst}", loc="left")
        ax.set_xlabel("progresso (% do orçamento)")
        tabela.append({
            "instancia": inst,
            "filhos_cruzamento": int(soma["cn"].sum()),
            "sucesso_cruzamento_%": round(100 * soma["cs"].sum() / max(1, soma["cn"].sum()), 3),
            "sucesso_cruzamento_inicio_%": round(100 * taxa_c[0], 3),
            "sucesso_cruzamento_fim_%": round(100 * np.nanmean(taxa_c[-5:]), 3),
            "filhos_so_mutacao": int(soma["mn"].sum()),
            "sucesso_mutacao_%": round(100 * soma["ms"].sum() / max(1, soma["mn"].sum()), 3),
        })
    eixos[0].set_ylabel("% de filhos melhores (escala symlog)")
    eixos[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("GA – com que frequência cada operador gera um filho melhor que os pais",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "ga_operadores.png"), dpi=160)
    plt.close(fig)
    if tabela:
        salvar_csv(tabela, "ga_operadores.csv")


def salvar_csv(linhas, arquivo):
    if not linhas:
        return
    with open(os.path.join(SAIDA, arquivo), "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    linhas = ler_execucoes()
    ordem = ordem_configs(linhas)
    print("Configuracoes encontradas:", ", ".join(ordem))

    tabela_principal(linhas, ordem)
    testes(linhas, ordem)
    boxplots(linhas, ordem)
    convergencia(ordem)
    sa_aceitacao()
    ga_diversidade(ordem)
    ga_operadores(ordem)

    with open(os.path.join(SAIDA, "tabela_principal.md")) as f:
        print("\n" + f.read())
    with open(os.path.join(SAIDA, "testes_mann_whitney.md")) as f:
        print(f.read())
    print(f"Arquivos gerados em {SAIDA}/")


if __name__ == "__main__":
    main()

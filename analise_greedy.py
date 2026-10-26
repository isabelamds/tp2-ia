"""
Gerar figuras e tabelas do T2 (vizinhancas do Greedy Descent).

Le os CSVs gerados pelo experimento_greedy.py e produz o material do relatorio.

Entrada: resultados/t2_execucoes.csv, t2_resumo.csv, t2_custo_movimento.csv, t2_testes.csv
Saida  : relatorio/figuras/*.png e *.pdf
         relatorio/tabelas/*.csv e *.tex
         relatorio/conclusoes_t2.txt   (resumo em texto do que os dados mostram)

Figuras
    boxplot_melhor_h_<estrategia>   boxplots de h por vizinhanca, um painel por instancia
    boxplot_otimos_locais_<estr.>   h medio dos otimos locais por vizinhanca
    passos_melhoria                 passos de melhoria dentro do orcamento (best x first)
    custo_por_movimento             us por movimento: completa x incremental
    ganho_incremental               quantas vezes o incremental e mais rapido que a completa
Tabelas
    principal, otimos_locais, tempo_avaliacoes, custo_movimento, tamanho_vizinhanca,
    testes_qualidade, testes_tempo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDEM_VIZ = ["swap", "insertion", "2opt"]
COR_VIZ = {"swap": "#0072B2", "insertion": "#E69F00", "2opt": "#009E73"}
ROT_ESTR = {"best": "best improvement", "first": "first improvement"}
COR_ESTR = {"best": "#4C72B0", "first": "#DD8452"}
ROT_MODO = {
    "completa": "completa",
    "incremental_sem_poda": "incremental (sem poda)",
    "incremental_com_poda": "incremental (com poda)",
}
COR_MODO = {"completa": "#999999", "incremental_sem_poda": "#56B4E9",
            "incremental_com_poda": "#0072B2"}
ROT_METRICA = {
    "melhor_h": "melhor h",
    "media_h_otimos_locais": "h médio dos ótimos locais",
    "tempo_ms_por_movimento[incremental_sem_poda]": "tempo/mov. (incr. sem poda)",
    "tempo_ms_por_movimento[incremental_com_poda]": "tempo/mov. (incr. com poda)",
}
COLUNAS_TEXTO = {"Instância", "Estratégia", "Vizinhança", "Comparação", "Métrica", "Teste",
                 "Fórmula", "Significativo"}


# ----------------------------------------------------------------------------
# Leitura e utilitarios
# ----------------------------------------------------------------------------
def carregar(entrada: Path) -> dict[str, pd.DataFrame]:
    dados = {}
    for nome in ("execucoes", "resumo", "custo_movimento", "testes"):
        caminho = entrada / f"t2_{nome}.csv"
        if not caminho.is_file():
            raise SystemExit(f"Nao encontrei {caminho}. Rode antes o experimento_greedy.py.")
        dados[nome] = pd.read_csv(caminho)
    return dados


def rotulo_instancia(nome: str, n: int) -> str:
    return f"{nome.removeprefix('manutencao_')} (n={n})"


def adicionar_ordem(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona 'rot_inst' e ordena por n, estrategia e vizinhanca."""
    df = df.copy()
    df["rot_inst"] = [rotulo_instancia(i, n) for i, n in zip(df["instancia"], df["n"])]
    df["_viz"] = df["vizinhanca"].map({v: k for k, v in enumerate(ORDEM_VIZ)})
    chaves = ["n"] + (["estrategia"] if "estrategia" in df else []) + ["_viz"]
    if "modo" in df:
        chaves.append("modo")
    return df.sort_values(chaves).reset_index(drop=True)


def instancias_ordenadas(df: pd.DataFrame) -> list[tuple[str, int]]:
    pares = df[["instancia", "n"]].drop_duplicates().sort_values("n")
    return list(zip(pares["instancia"], pares["n"]))


def salvar_figura(fig, pasta: Path, nome: str) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(pasta / f"{nome}.png", dpi=200)
    fig.savefig(pasta / f"{nome}.pdf")
    plt.close(fig)
    print(f"  figura: {pasta / (nome + '.png')}")


def como_booleano(serie: pd.Series) -> pd.Series:
    return serie.astype(str).map({"True": 1.0, "False": 0.0})


def f1(x):
    return "—" if pd.isna(x) else f"{x:,.1f}".replace(",", "")


def f0(x):
    return "—" if pd.isna(x) else f"{x:,.0f}".replace(",", "")


def f_p(p):
    if pd.isna(p):
        return "n/d"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------
# Saida de tabelas (CSV + LaTeX)
# ----------------------------------------------------------------------------
def _esc(texto: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("±", r"$\pm$"), ("µ", r"$\mu$"), ("≥", r"$\geq$"),
                 ("×", r"$\times$"), ("<", r"$<$")):
        texto = texto.replace(a, b)
    return texto


def salvar_tabela(df: pd.DataFrame, pasta: Path, nome: str, legenda: str) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta / f"{nome}.csv", index=False, encoding="utf-8")

    alinh = "".join("l" if c in COLUNAS_TEXTO else "r" for c in df.columns)
    linhas = [r"\begin{table}[ht]", r"\centering", r"\small",
              rf"\caption{{{_esc(legenda)}}}", rf"\label{{tab:{nome}}}",
              rf"\begin{{tabular}}{{{alinh}}}", r"\hline",
              " & ".join(_esc(str(c)) for c in df.columns) + r" \\", r"\hline"]
    anterior = None
    for _, linha in df.iterrows():
        atual = linha.iloc[0]
        if anterior is not None and atual != anterior:
            linhas.append(r"\hline")
        anterior = atual
        linhas.append(" & ".join(_esc(str(v)) for v in linha) + r" \\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (pasta / f"{nome}.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"  tabela: {pasta / (nome + '.csv')} (+ .tex)")


# ----------------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------------
def tabela_principal(resumo: pd.DataFrame) -> pd.DataFrame:
    r = adicionar_ordem(resumo)
    ref = r["h_referencia"]
    return pd.DataFrame({
        "Instância": r["rot_inst"],
        "Estratégia": r["estrategia"].map(ROT_ESTR),
        "Vizinhança": r["vizinhanca"],
        "Média ± DP": [f"{m:.1f} ± {s:.1f}" for m, s in zip(r["media_h"], r["desvio_h"])],
        "Mínimo": r["minimo_h"].map(f0),
        "Mediana": r["mediana_h"].map(f1),
        "Gap da média (%)": (100 * r["gap_medio"]).map(f1),
        "Gap do melhor (%)": (100 * (r["minimo_h"] - ref) / ref).map(f1),
    })


def tabela_otimos_locais(resumo: pd.DataFrame, exe: pd.DataFrame) -> pd.DataFrame:
    r = adicionar_ordem(resumo)
    ext = (exe.groupby(["instancia", "estrategia", "vizinhanca"])["otimos_locais_distintos"]
           .agg(["min", "max"]).reset_index())
    r = r.merge(ext, on=["instancia", "estrategia", "vizinhanca"], how="left")
    atingiu = [f"{int(a)}/{int(e)}" if pd.notna(a) else "—"
               for a, e in zip(r["execucoes_que_atingiram_referencia"], r["execucoes"])]
    return pd.DataFrame({
        "Instância": r["rot_inst"],
        "Estratégia": r["estrategia"].map(ROT_ESTR),
        "Vizinhança": r["vizinhanca"],
        "Reinícios (média)": r["media_reinicios"].map(f1),
        "Passos de melhoria (média)": r["media_passos_melhoria"].map(f1),
        "Ótimos locais distintos (média [mín–máx])": [
            f"{m:.1f} [{lo}–{hi}]" for m, lo, hi in
            zip(r["media_otimos_locais_distintos"], r["min"], r["max"])],
        "Execuções com ≥1 ótimo local": [f"{int(a)}/{int(e)}" for a, e in
                                         zip(r["execucoes_com_otimo_local"], r["execucoes"])],
        "h médio dos ótimos locais": r["media_h_otimos_locais"].map(f1),
        "Execuções que atingiram a referência": atingiu,
    })


def tabela_tempo_avaliacoes(resumo: pd.DataFrame, exe: pd.DataFrame) -> pd.DataFrame:
    r = adicionar_ordem(resumo)
    dp = (exe.groupby(["instancia", "estrategia", "vizinhanca"])["tempo_s"]
          .std().fillna(0.0).rename("dp_tempo").reset_index())
    r = r.merge(dp, on=["instancia", "estrategia", "vizinhanca"], how="left")
    return pd.DataFrame({
        "Instância": r["rot_inst"],
        "Estratégia": r["estrategia"].map(ROT_ESTR),
        "Vizinhança": r["vizinhanca"],
        "Avaliações (média)": r["media_avaliacoes"].map(f0),
        "Tempo por execução (s)": [f"{m:.2f} ± {s:.2f}" for m, s in
                                   zip(r["media_tempo_s"], r["dp_tempo"])],
        "ms por avaliação": (1000 * r["media_tempo_s"] / r["media_avaliacoes"]).map(
            lambda x: f"{x:.4f}"),
    })


def tabela_custo_movimento(custo: pd.DataFrame) -> pd.DataFrame:
    c = adicionar_ordem(custo)
    base = c[["instancia", "n", "rot_inst", "vizinhanca", "_viz"]].drop_duplicates()
    us = c.pivot(index=["instancia", "vizinhanca"], columns="modo",
                 values="media_ms_por_movimento").mul(1000)
    ganho = c.pivot(index=["instancia", "vizinhanca"], columns="modo", values="ganho_vs_completa")
    mov = (c[c["modo"] == "completa"].set_index(["instancia", "vizinhanca"])
           ["movimentos_por_varredura"])
    t = base.sort_values(["n", "_viz"]).reset_index(drop=True)
    chave = list(zip(t["instancia"], t["vizinhanca"]))
    return pd.DataFrame({
        "Instância": t["rot_inst"],
        "Vizinhança": t["vizinhanca"],
        "Movimentos por varredura": [int(mov[k]) for k in chave],
        "µs/mov. completa": [f"{us.loc[k, 'completa']:.2f}" for k in chave],
        "µs/mov. incr. sem poda": [f"{us.loc[k, 'incremental_sem_poda']:.2f}" for k in chave],
        "µs/mov. incr. com poda": [f"{us.loc[k, 'incremental_com_poda']:.2f}" for k in chave],
        "Ganho sem poda (×)": [f"{ganho.loc[k, 'incremental_sem_poda']:.2f}" for k in chave],
        "Ganho com poda (×)": [f"{ganho.loc[k, 'incremental_com_poda']:.2f}" for k in chave],
    })


def tabela_tamanho_vizinhanca(custo: pd.DataFrame) -> pd.DataFrame:
    c = custo[custo["modo"] == "completa"]
    piv = c.pivot(index="vizinhanca", columns="n", values="movimentos_por_varredura")
    piv = piv.reindex([v for v in ORDEM_VIZ if v in piv.index])
    formula = {"swap": "n(n-1)/2", "2opt": "n(n-1)/2", "insertion": "(n-1)^2"}
    out = pd.DataFrame({"Vizinhança": piv.index,
                        "Fórmula": [formula.get(v, "") for v in piv.index]})
    for n in piv.columns:
        out[f"n = {n}"] = piv[n].astype(int).to_numpy()
    return out


def _tabela_testes(testes: pd.DataFrame, metricas: list[str]) -> pd.DataFrame:
    t = testes[testes["metrica"].isin(metricas)].copy()
    t["estrategia"] = t["estrategia"].fillna("")
    t["rot_inst"] = [rotulo_instancia(i, n) for i, n in zip(t["instancia"], t["n"])]
    t["_m"] = t["metrica"].map({m: k for k, m in enumerate(metricas)})
    t = t.sort_values(["n", "estrategia", "_m"], kind="stable").reset_index(drop=True)

    def sig(valor, p):
        if pd.isna(p):
            return "—"
        return "sim" if str(valor) == "True" else "não"

    return pd.DataFrame({
        "Instância": t["rot_inst"],
        "Estratégia": t["estrategia"].map(lambda e: ROT_ESTR.get(e, "—")),
        "Métrica": t["metrica"].map(ROT_METRICA),
        "Comparação": t["comparacao"],
        "Teste": t["teste"],
        "Mediana A": t["mediana_A"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "—"),
        "Mediana B": t["mediana_B"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "—"),
        "p-valor": t["p_valor"].map(f_p),
        "p (Holm)": t["p_valor_holm"].map(f_p),
        "Significativo": [sig(v, p) for v, p in zip(t["significativo"], t["p_valor_holm"])],
    })


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def _caixas(ax, dados_por_viz):
    """Boxplot + pontos individuais, uma caixa por vizinhanca (vazias ignoradas)."""
    rng = np.random.default_rng(0)
    for pos, viz in enumerate(ORDEM_VIZ, 1):
        d = np.asarray(dados_por_viz.get(viz, []), dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        bp = ax.boxplot([d], positions=[pos], widths=0.55, patch_artist=True, showfliers=False)
        bp["boxes"][0].set(facecolor=COR_VIZ[viz], alpha=0.55)
        bp["medians"][0].set(color="black")
        ax.scatter(pos + rng.uniform(-0.14, 0.14, d.size), d, s=12, color="black",
                   alpha=0.55, zorder=3)
    ax.set_xticks(range(1, len(ORDEM_VIZ) + 1))
    ax.set_xticklabels(ORDEM_VIZ)
    ax.grid(axis="y", alpha=0.3)


def fig_boxplot(exe, estrategia, metrica, nome, titulo_y, pasta, linha_ref):
    insts = instancias_ordenadas(exe)
    fig, axes = plt.subplots(1, len(insts), figsize=(4.3 * len(insts), 4.3), squeeze=False)
    algum_dado = False
    for ax, (inst, n) in zip(axes[0], insts):
        sub = exe[(exe["instancia"] == inst) & (exe["estrategia"] == estrategia)]
        dados = {v: sub.loc[sub["vizinhanca"] == v, metrica].to_numpy() for v in ORDEM_VIZ}
        tem = any(np.isfinite(d).any() for d in dados.values())
        algum_dado |= tem
        ax.set_title(rotulo_instancia(inst, n), fontsize=10)
        if not tem:
            ax.text(0.5, 0.5, "nenhuma descida terminou\ndentro do orçamento", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            continue
        _caixas(ax, dados)
        if linha_ref:
            ref = sub["h_referencia"].dropna()
            if len(ref):
                rotulo = "ótimo global" if n == 12 else "melhor conhecido"
                ax.axhline(ref.iloc[0], ls="--", color="gray", lw=1, label=f"{rotulo}")
                ax.legend(fontsize=8, loc="best")
    axes[0][0].set_ylabel(titulo_y)
    fig.suptitle(ROT_ESTR.get(estrategia, estrategia), fontsize=11)
    if not algum_dado:
        plt.close(fig)
        print(f"  (sem dados para {nome}_{estrategia}; figura omitida)")
        return
    salvar_figura(fig, pasta, f"{nome}_{estrategia}")


def fig_passos(exe, pasta):
    insts = instancias_ordenadas(exe)
    estrategias = [e for e in ROT_ESTR if e in set(exe["estrategia"])]
    g = (exe.groupby(["instancia", "estrategia", "vizinhanca"])["passos_melhoria"]
         .agg(["mean", "std"]).fillna(0.0))
    fig, axes = plt.subplots(1, len(insts), figsize=(4.3 * len(insts), 4.3), squeeze=False)
    largura = 0.8 / len(estrategias)
    for ax, (inst, n) in zip(axes[0], insts):
        for k, est in enumerate(estrategias):
            x = np.arange(len(ORDEM_VIZ)) + (k - (len(estrategias) - 1) / 2) * largura
            m = [g.loc[(inst, est, v), "mean"] for v in ORDEM_VIZ]
            s = [g.loc[(inst, est, v), "std"] for v in ORDEM_VIZ]
            ax.bar(x, m, largura * 0.95, yerr=s, capsize=3, color=COR_ESTR[est],
                   label=ROT_ESTR[est])
            for xi, mi in zip(x, m):
                ax.annotate(f"{mi:.0f}", (xi, mi), ha="center", va="bottom", fontsize=7,
                            xytext=(0, 2), textcoords="offset points")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(len(ORDEM_VIZ)))
        ax.set_xticklabels(ORDEM_VIZ)
        ax.set_title(rotulo_instancia(inst, n), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("passos de melhoria dentro do orçamento\n(média ± DP, escala symlog)")
    handles, rotulos = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="upper right", ncol=len(rotulos), fontsize=9,
               frameon=False)
    salvar_figura(fig, pasta, "passos_melhoria")


def fig_custo_movimento(custo, pasta):
    insts = instancias_ordenadas(custo)
    fig, axes = plt.subplots(1, len(insts), figsize=(4.3 * len(insts), 4.3), squeeze=False)
    modos = [m for m in ROT_MODO if m in set(custo["modo"])]
    largura = 0.8 / len(modos)
    for ax, (inst, n) in zip(axes[0], insts):
        sub = custo[custo["instancia"] == inst].set_index(["vizinhanca", "modo"])
        for k, modo in enumerate(modos):
            x = np.arange(len(ORDEM_VIZ)) + (k - (len(modos) - 1) / 2) * largura
            m = [1000 * sub.loc[(v, modo), "media_ms_por_movimento"] for v in ORDEM_VIZ]
            s = [1000 * sub.loc[(v, modo), "desvio_ms_por_movimento"] for v in ORDEM_VIZ]
            ax.bar(x, m, largura * 0.95, yerr=s, capsize=3, color=COR_MODO[modo],
                   label=ROT_MODO[modo])
        ax.set_xticks(range(len(ORDEM_VIZ)))
        ax.set_xticklabels(ORDEM_VIZ)
        ax.set_title(rotulo_instancia(inst, n), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("µs por movimento (média ± DP)")
    handles, rotulos = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="upper right", ncol=len(rotulos), fontsize=9,
               frameon=False)
    salvar_figura(fig, pasta, "custo_por_movimento")


def fig_ganho(custo, pasta):
    insts = instancias_ordenadas(custo)
    modos = ["incremental_sem_poda", "incremental_com_poda"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    largura = 0.8 / len(ORDEM_VIZ)
    for ax, modo in zip(axes, modos):
        for k, viz in enumerate(ORDEM_VIZ):
            x = np.arange(len(insts)) + (k - 1) * largura
            g = [custo[(custo["instancia"] == i) & (custo["vizinhanca"] == viz)
                       & (custo["modo"] == modo)]["ganho_vs_completa"].iloc[0]
                 for i, _ in insts]
            ax.bar(x, g, largura * 0.95, color=COR_VIZ[viz], label=viz)
            for xi, gi in zip(x, g):
                ax.annotate(f"{gi:.1f}×", (xi, gi), ha="center", va="bottom", fontsize=7,
                            xytext=(0, 2), textcoords="offset points")
        ax.axhline(1, color="black", lw=0.8)
        ax.set_xticks(range(len(insts)))
        ax.set_xticklabels([rotulo_instancia(i, n) for i, n in insts], fontsize=9)
        ax.set_title(ROT_MODO[modo], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("ganho sobre a avaliação completa (×)")
    axes[0].legend(fontsize=8)
    salvar_figura(fig, pasta, "ganho_incremental")


# ----------------------------------------------------------------------------
# Conclusoes em texto
# ----------------------------------------------------------------------------
def _p_holm(testes, inst, est, metrica, a, b):
    t = testes[(testes["instancia"] == inst) & (testes["metrica"] == metrica)
               & (testes["estrategia"].fillna("") == est)]
    for _, l in t.iterrows():
        if {l["comparacao"].split(" vs ")[0], l["comparacao"].split(" vs ")[1]} == {a, b}:
            return l["p_valor_holm"]
    return np.nan


def conclusoes(exe, custo, testes) -> list[str]:
    linhas = ["CONCLUSÕES DO T2 (geradas automaticamente; confira antes de citar)", ""]
    linhas.append("1) Qualidade: vizinhança com menor mediana de melhor h")
    for inst, n in instancias_ordenadas(exe):
        for est in [e for e in ROT_ESTR if e in set(exe["estrategia"])]:
            sub = exe[(exe["instancia"] == inst) & (exe["estrategia"] == est)]
            med = sub.groupby("vizinhanca")["melhor_h"].median().reindex(ORDEM_VIZ)
            if med.nunique() == 1:
                linhas.append(f"  {rotulo_instancia(inst, n):<18} {est:<5} -> empate "
                              f"(as três vizinhanças têm mediana {med.iloc[0]:.1f})")
                continue
            melhor = med.idxmin()
            partes = []
            for outra in ORDEM_VIZ:
                if outra == melhor:
                    continue
                p = _p_holm(testes, inst, est, "melhor_h", melhor, outra)
                if pd.isna(p):
                    partes.append(f"vs {outra}: n/d")
                else:
                    partes.append(f"vs {outra}: p(Holm)={f_p(p)} "
                                  f"({'significativo' if p < 0.05 else 'não significativo'})")
            linhas.append(f"  {rotulo_instancia(inst, n):<18} {est:<5} -> {melhor} "
                          f"(mediana {med[melhor]:.1f}); " + "; ".join(partes))

    linhas += ["", "2) Custo por movimento com avaliação incremental (menor média)"]
    for inst, n in instancias_ordenadas(custo):
        for modo in ("incremental_sem_poda", "incremental_com_poda"):
            sub = custo[(custo["instancia"] == inst) & (custo["modo"] == modo)]
            m = sub.set_index("vizinhanca")["media_ms_por_movimento"].reindex(ORDEM_VIZ) * 1000
            linhas.append(f"  {rotulo_instancia(inst, n):<18} {ROT_MODO[modo]:<24} -> "
                          f"{m.idxmin()} ({m.min():.2f} µs; mais lenta: {m.idxmax()} "
                          f"{m.max():.2f} µs)")

    linhas += ["", "3) Ganho do incremental sobre a avaliação completa (×)"]
    for inst, n in instancias_ordenadas(custo):
        partes = []
        for modo in ("incremental_sem_poda", "incremental_com_poda"):
            sub = custo[(custo["instancia"] == inst) & (custo["modo"] == modo)]
            g = sub.set_index("vizinhanca")["ganho_vs_completa"].reindex(ORDEM_VIZ)
            partes.append(f"{ROT_MODO[modo]}: " + ", ".join(f"{v} {g[v]:.2f}" for v in ORDEM_VIZ))
        linhas.append(f"  {rotulo_instancia(inst, n):<18} " + " | ".join(partes))

    linhas += ["", "4) Referência atingida (execuções com melhor h <= referência)"]
    for inst, n in instancias_ordenadas(exe):
        for est in [e for e in ROT_ESTR if e in set(exe["estrategia"])]:
            sub = exe[(exe["instancia"] == inst) & (exe["estrategia"] == est)]
            if sub["h_referencia"].isna().all():
                continue
            atingiu = sub.assign(a=como_booleano(sub["atingiu_referencia"])).groupby(
                "vizinhanca")["a"].agg(["sum", "count"]).reindex(ORDEM_VIZ)
            linhas.append(f"  {rotulo_instancia(inst, n):<18} {est:<5} " + ", ".join(
                f"{v} {int(atingiu.loc[v, 'sum'])}/{int(atingiu.loc[v, 'count'])}"
                for v in ORDEM_VIZ))
    return linhas


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="resultados", help="Pasta dos CSVs (padrão: resultados).")
    ap.add_argument("--saida", default="relatorio", help="Pasta de saída (padrão: relatorio).")
    args = ap.parse_args()

    dados = carregar(Path(args.entrada))
    exe, resumo = dados["execucoes"], dados["resumo"]
    custo, testes = dados["custo_movimento"], dados["testes"]
    figuras, tabelas = Path(args.saida) / "figuras", Path(args.saida) / "tabelas"

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    print("Figuras:")
    for est in [e for e in ROT_ESTR if e in set(exe["estrategia"])]:
        fig_boxplot(exe, est, "melhor_h", "boxplot_melhor_h", "melhor h encontrado",
                    figuras, linha_ref=True)
        fig_boxplot(exe, est, "media_h_otimos_locais", "boxplot_otimos_locais",
                    "h médio dos ótimos locais (por execução)", figuras, linha_ref=False)
    fig_passos(exe, figuras)
    fig_custo_movimento(custo, figuras)
    fig_ganho(custo, figuras)

    print("Tabelas:")
    salvar_tabela(
        tabela_principal(resumo), tabelas, "principal",
        "Melhor h encontrado por vizinhança (greedy descent com random restart). Gap relativo "
        "ao ótimo global (n=12) ou ao melhor valor conhecido (n=50 e n=150).")
    salvar_tabela(
        tabela_otimos_locais(resumo, exe), tabelas, "otimos_locais",
        "Reinícios, passos de melhoria e ótimos locais distintos encontrados dentro do orçamento.")
    salvar_tabela(
        tabela_tempo_avaliacoes(resumo, exe), tabelas, "tempo_avaliacoes",
        "Número de avaliações e tempo de parede por execução.")
    salvar_tabela(
        tabela_custo_movimento(custo), tabelas, "custo_movimento",
        "Custo por movimento: avaliação completa x incremental (ganho = tempo da completa / "
        "tempo da incremental).")
    salvar_tabela(
        tabela_tamanho_vizinhanca(custo), tabelas, "tamanho_vizinhanca",
        "Tamanho da vizinhança induzida por cada operador em função de n.")
    salvar_tabela(
        _tabela_testes(testes, ["melhor_h", "media_h_otimos_locais"]), tabelas, "testes_qualidade",
        "Testes entre vizinhanças (Mann-Whitney U, duas caudas; p ajustado por Holm).")
    salvar_tabela(
        _tabela_testes(testes, ["tempo_ms_por_movimento[incremental_sem_poda]",
                                "tempo_ms_por_movimento[incremental_com_poda]"]),
        tabelas, "testes_tempo",
        "Testes entre vizinhanças para o tempo por movimento (Wilcoxon pareado; p ajustado "
        "por Holm).")

    texto = conclusoes(exe, custo, testes)
    destino = Path(args.saida) / "conclusoes_t2.txt"
    destino.write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n" + "\n".join(texto))
    print(f"\nTudo em {Path(args.saida)}/")


if __name__ == "__main__":
    main()

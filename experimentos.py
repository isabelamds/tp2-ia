"""
experimentos.py - Roda os algoritmos.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from problema import ProblemaManutencao

INSTANCIAS = {
    "pequena": {"arquivo": "instancias/manutencao_pequena.txt", "orcamento": 50_000, "execucoes": 30},
    "media":   {"arquivo": "instancias/manutencao_media.txt",   "orcamento": 150_000, "execucoes": 30},
    "grande":  {"arquivo": "instancias/manutencao_grande.txt",  "orcamento": 300_000, "execucoes": 10},
}

# Sementes: execucao r usa a semente 1 + r (as mesmas para todos os algoritmos).
# A calibracao usa sementes 1000+, para nao coincidir.
def semente_da_execucao(r):
    return 1 + r


# Configuracoes de algoritmos

def _config_sa():
    from sa import simulated_annealing
    # Temperaturas calibradas por calibrar_sa.py.
    # Regra unica para todas as instancias
    temperaturas = {
        "pequena": {"T0": 1505.4,  "Tf": 13.95},
        "media":   {"T0": 5119.2,  "Tf": 38.51},
        "grande":  {"T0": 14193.7, "Tf": 86.81},
    }
    return {
        "sa": {
            "funcao": simulated_annealing,
            "vizinhanca": "insertion",
            # O SA troca de solucao corrente com muita frequencia; a cada troca
            # a avaliacao incremental precisa recalcular os resumos (O(n)).
            # Medido: fica mais lenta que a completa.
            "incremental": False,
            "parametros": temperaturas,
        },
    }


def _config_ga():
    from ga import algoritmo_genetico
    params = {"N": 100, "Pc": 0.9, "Pm": 0.2, "elite": 2, "k_torneio": 5}
    # Experimento adicional: o mesmo GA sem cruzamento (Pc = 0).
    sem_cruz = dict(params, Pc=0.0)
    return {
        "ga": {
            "funcao": algoritmo_genetico,
            "vizinhanca": "insertion",      # operador de mutacao
            "incremental": False,           # o GA so avalia solucoes completas
            "parametros": {inst: params for inst in INSTANCIAS},
        },
        "ga_sem_cruzamento": {
            "funcao": algoritmo_genetico,
            "vizinhanca": "insertion",
            "incremental": False,
            "parametros": {inst: sem_cruz for inst in INSTANCIAS},
        },
    }


def _config_greedy():
    import greedy  # noqa: F401  (se nao existir, a configuracao e ignorada)
    return getattr(greedy, "CONFIGURACOES", {})


def carregar_configuracoes(avisar=False):
    configs = {}
    for fabrica in (_config_sa, _config_ga, _config_greedy):
        try:
            configs.update(fabrica())
        except ImportError as erro:
            if avisar:
                print(f"(aviso: configuracao ignorada, modulo ausente: {erro.name})")
    return configs


# Execucao de uma rodada
def rodar_uma(nome_config, nome_inst, r):
    """Roda UMA execucao e devolve (linha do CSV, detalhes para o JSON)."""
    config = carregar_configuracoes()[nome_config]
    inst = INSTANCIAS[nome_inst]
    semente = semente_da_execucao(r)

    prob = ProblemaManutencao(inst["arquivo"],
                              vizinhanca=config["vizinhanca"],
                              incremental=config["incremental"])
    prob.reiniciar(semente=semente, orcamento=inst["orcamento"])
    parametros = config["parametros"].get(nome_inst, {})

    inicio = time.perf_counter()
    extras = config["funcao"](prob, **parametros)
    tempo = time.perf_counter() - inicio

    linha = {
        "config": nome_config,
        "instancia": nome_inst,
        "execucao": r,
        "semente": semente,
        "vizinhanca": config["vizinhanca"],
        "incremental": config["incremental"],
        "melhor_h": prob.melhor_h,
        "avaliacoes": prob.avaliacoes,
        "tempo_s": round(tempo, 3),
        "melhor_s": " ".join(map(str, prob.melhor_s)),
    }
    detalhes = {
        **linha,
        "parametros": parametros,
        "historico": prob.historico,
        "extras": extras,
    }
    return linha, detalhes


# Programa principal
CSV = "resultados/execucoes.csv"
PASTA = "resultados/execucoes"
CAMPOS = ["config", "instancia", "execucao", "semente", "vizinhanca", "incremental",
          "melhor_h", "avaliacoes", "tempo_s", "melhor_s"]


def ja_feitas():
    if not os.path.exists(CSV):
        return set()
    with open(CSV) as f:
        return {(l["config"], l["instancia"], int(l["execucao"])) for l in csv.DictReader(f)}


def main():
    configs = carregar_configuracoes(avisar=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", help="nomes das configuracoes ou 'todos'")
    parser.add_argument("--instancias", nargs="*", default=list(INSTANCIAS))
    parser.add_argument("--processos", type=int, default=os.cpu_count())
    parser.add_argument("--listar", action="store_true")
    args = parser.parse_args()

    if args.listar or not args.configs:
        print("Configuracoes disponiveis:", ", ".join(configs))
        return
    nomes = list(configs) if "todos" in args.configs else args.configs
    for nome in nomes:
        if nome not in configs:
            raise SystemExit(f"Configuracao desconhecida: {nome}")

    os.makedirs(PASTA, exist_ok=True)
    feitas = ja_feitas()
    tarefas = [(c, i, r) for c in nomes for i in args.instancias
               for r in range(INSTANCIAS[i]["execucoes"]) if (c, i, r) not in feitas]
    print(f"{len(tarefas)} execucoes a fazer ({len(feitas)} ja feitas), "
          f"usando {args.processos} processos.")

    novo = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f, \
         ProcessPoolExecutor(max_workers=args.processos) as pool:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            escritor.writeheader()
        futuros = {pool.submit(rodar_uma, *t): t for t in tarefas}
        for k, futuro in enumerate(as_completed(futuros), 1):
            linha, detalhes = futuro.result()
            escritor.writerow(linha)
            f.flush()                          # grava ja, para nao perder nada
            nome_json = f"{linha['config']}_{linha['instancia']}_{linha['execucao']:02d}.json"
            with open(os.path.join(PASTA, nome_json), "w") as fj:
                json.dump(detalhes, fj)
            print(f"[{k}/{len(tarefas)}] {linha['config']:>10} {linha['instancia']:>7} "
                  f"exec {linha['execucao']:>2}: h = {linha['melhor_h']:>7} "
                  f"({linha['tempo_s']:.1f} s)")


if __name__ == "__main__":
    main()

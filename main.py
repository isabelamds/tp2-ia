import subprocess
import sys
import time


def executar(numero, total, descricao, comando):
    """Executa um script e para tudo se ele falhar."""
    print(f"\n=== Etapa {numero}/{total}: {descricao} ===")
    print("$ python3 " + " ".join(comando), flush=True)
    inicio = time.perf_counter()
    resultado = subprocess.run([sys.executable] + comando)
    duracao = time.perf_counter() - inicio
    if resultado.returncode != 0:
        print(f"\nA etapa '{descricao}' falhou (codigo {resultado.returncode}). "
              "As etapas seguintes nao foram executadas.")
        sys.exit(resultado.returncode)
    print(f"--- concluida em {duracao:.1f} s")


def main():
    calibrar = "--calibrar" in sys.argv[1:]

    etapas = [
        ("verificar o problema.py", ["testes.py"]),
        ("medir o ganho da avaliacao incremental", ["medir_incremental.py"]),
    ]
    if calibrar:
        etapas += [
            ("calibrar o simulated annealing", ["calibrar_sa.py"]),
            ("calibrar o algoritmo genetico", ["calibrar_ga.py", "media"]),
        ]
    etapas += [
        ("executar os experimentos", ["experimentos.py", "todos"]),
        ("gerar tabelas e graficos", ["analise.py"]),
    ]

    inicio = time.perf_counter()
    for numero, (descricao, comando) in enumerate(etapas, 1):
        executar(numero, len(etapas), descricao, comando)
    print(f"\nTudo pronto em {time.perf_counter() - inicio:.0f} s. "
          "Resultados em resultados/ e figuras em resultados/analise/.")


if __name__ == "__main__":
    main()

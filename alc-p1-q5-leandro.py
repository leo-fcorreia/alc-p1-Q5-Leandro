import numpy as np

TOLERANCIA = 1e-12

def resolve_lu(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Validação da entrada
    n = len(A)
    for linha in A:
        if len(linha) != n:
            raise Exception("A deve ser uma matriz quadrada.")
    if len(b) != n:
        raise Exception(f"b deve ter {n} elementos.")

    # 1) Decomposição A = LU pela eliminação de Gauss
    U = np.array(A, dtype=float)   # cópia de A, que vai virar U
    L = np.eye(n)                  # identidade: diagonal de L = 1

    for k in range(n):   # coluna do pivô
        if abs(U[k][k]) < TOLERANCIA:
            raise Exception(
                f"Pivô nulo na posição ({k+1},{k+1}). A decomposição LU sem "
                "pivotamento não se aplica. Sugestão: use a eliminação de Gauss "
                "com pivotamento parcial (PA = LU)."
            )
        for i in range(k + 1, n):            # linhas abaixo do pivô
            m = U[i][k] / U[k][k]            # multiplicador da eliminação
            L[i][k] = m                      # o multiplicador é o elemento de L
            for j in range(k, n):
                U[i][j] = U[i][j] - m * U[k][j]   # linha_i <- linha_i - m*linha_k

    # 2) Substituição progressiva: Ly = b (de cima para baixo)
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = b[i] - soma    # L[i][i] = 1

    # 3) Substituição regressiva: Ux = y (de baixo para cima)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]

    return L, U, x


# ---------------------------------------------------------------------------
# Verificação e exibição (não fazem parte da resolução)
# ---------------------------------------------------------------------------
def confere(A, b, L, U, x, tol=1e-9):
    """Verifica, com laços, se L·U = A e se A·x = b."""
    n = len(b)
    lu_ok, ax_ok = True, True
    for i in range(n):
        for j in range(n):
            valor = 0.0
            for k in range(n):
                valor += L[i][k] * U[k][j]
            if abs(valor - A[i][j]) > tol:
                lu_ok = False
        valor_b = 0.0
        for j in range(n):
            valor_b += A[i][j] * x[j]
        if abs(valor_b - b[i]) > tol:
            ax_ok = False
    return lu_ok, ax_ok


def fmt(v):
    """Inteiro sem casas decimais; demais valores com até 4 casas."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0")


def linhas_matriz(M):
    """Converte a matriz em linhas de texto alinhadas: | a  b  c |"""
    textos = [[fmt(v) for v in linha] for linha in M]
    w = max(len(t) for linha in textos for t in linha)
    return ["| " + "  ".join(t.rjust(w) for t in linha) + " |" for linha in textos]


def mostrar(A, b):
    n = len(A)

    print("\nSistema  [A | b]:")
    la = linhas_matriz(A)
    lb = linhas_matriz([[v] for v in b])
    for i in range(n):
        print("   " + la[i] + "  " + lb[i])

    try:
        L, U, x = resolve_lu(A, b)
    except Exception as erro:
        print("\nERRO:", erro, "\n")
        return

    print("\nDecomposição  A = L · U:")
    ll, lu = linhas_matriz(L), linhas_matriz(U)
    print("   " + "L".center(len(ll[0])) + "     " + "U".center(len(lu[0])))
    for i in range(n):
        print("   " + ll[i] + "     " + lu[i])

    print("\nSolução:  x = (" + ", ".join(fmt(v) for v in x) + ")")

    lu_ok, ax_ok = confere(A, b, L, U, x)
    print(f"Verificação:  A = LU {'OK' if lu_ok else 'FALHOU'}   |   "
          f"Ax = b {'OK' if ax_ok else 'FALHOU'}\n")


def inserir_dados():
    n = int(input("Ordem da matriz n: "))
    A = []
    for i in range(n):
        linha = input(f"Linha {i+1} de A ({n} números separados por espaço): ")
        A.append([float(v) for v in linha.split()])
        if len(A[i]) != n:
            raise ValueError
    b = [float(v) for v in input(f"Vetor b ({n} números): ").split()]
    if len(b) != n:
        raise ValueError
    return A, b


if __name__ == "__main__":
    print("1 - Teste (exemplo pronto)")
    print("2 - Inserir meus dados")
    opcao = input("Escolha: ")

    if opcao == "1":
        A = np.array([[2, 1, 1],
                      [4, -6, 0],
                      [-2, 7, 2]])
        b = np.array([5, -2, 9])       # solução esperada: x = (1, 1, 2)
        mostrar(A, b)
    elif opcao == "2":
        try:
            A, b = inserir_dados()
            mostrar(A, b)
        except ValueError:
            print("\nERRO: digite exatamente n números por linha, separados por espaço.\n")
    else:
        print("Opção inválida.")

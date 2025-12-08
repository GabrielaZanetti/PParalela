import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados_execucao.csv", header=None,
                 names=["np","threads","trials","tempo"])

df.sort_values("np", inplace=True)
t1 = df[df["np"]==1]["tempo"].values[0]
df["speedup"] = t1 / df["tempo"]
df["eficiencia"] = df["speedup"] / df["np"]

# tempo
plt.figure()
plt.plot(df["np"],df["tempo"],marker='o')
plt.xlabel("Processos MPI")
plt.ylabel("Tempo (s)")
plt.title("Tempo de Execução x Processos")
plt.grid(True)
plt.savefig("tempo_execucao.png")

# speedup
plt.figure()
plt.plot(df["np"],df["speedup"],marker='o')
plt.xlabel("Processos MPI")
plt.ylabel("Speedup")
plt.title("Speedup x Processos")
plt.grid(True)
plt.savefig("speedup.png")

# eficiencia
plt.figure()
plt.plot(df["np"],df["eficiencia"],marker='o')
plt.xlabel("Processos MPI")
plt.ylabel("Eficiência paralela")
plt.title("Eficiência x Processos")
plt.grid(True)
plt.savefig("eficiencia.png")

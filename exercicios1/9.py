vet = [34, 12, 5, 67, 23, 89, 1, 45, 78, 90]

print("Vetor original: "+ str(vet))
def ordenar_vetor(vet):
    for i in range(len(vet)):
        for j in range(i + 1, len(vet)):
            if vet[i] > vet[j]:
                vet[i], vet[j] = vet[j], vet[i]
    return vet

vetor_ordenado = ordenar_vetor(vet)
print("Vetor ordenado: "+ str(vetor_ordenado))
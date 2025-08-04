num = int(input("Digite um numero inteiro: "))
fatorial = 1
for i in range(1, num + 1):
    fatorial *= i
print("O fatorial de " + str(num) +  " e " + str(fatorial))
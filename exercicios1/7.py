numero = int(input("Digite um numero inteiro: "))
numero_str = str(abs(numero))

soma = 0
for digito in numero_str:
    soma += int(digito)

print("Soma dos digitos: "+ str(soma))

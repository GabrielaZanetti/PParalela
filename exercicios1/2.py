num1 = int(input("Digite o primeiro numero: "))
num2 = int(input("Digite o segundo numero: "))

soma = num1 + num2
subtracao = num1 - num2
multiplicacao = num1 * num2

if num2 != 0:
    divisao = num1 / num2
else:
    divisao = "Indefinida (divisao por zero)"

print("Soma: "+ str(soma))
print("Subtracao: "+ str(subtracao))
print("Multiplicacao: "+ str(multiplicacao))
print("Divisao: "+ str(divisao))

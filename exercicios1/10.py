texto = input("Digite uma string: ")
texto = texto.replace(" ", "").lower()

if texto == texto[::-1]:
    print("A string e um palindromo.")
else:
    print("A string nao e um palindromo.")

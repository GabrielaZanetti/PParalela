num = int(input("Digite um numero: "))

def eh_primo(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

if eh_primo(num):
    print(str(num) + " e um numero primo.")
else:
    print(str(num) + " nao e um numero primo.")


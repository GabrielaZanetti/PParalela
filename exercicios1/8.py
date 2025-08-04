num = int(input("Digite um numero: "))

def fibonacci(num):
    fib_sequence = []
    a, b = 0, 1
    while a <= num:
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

fib_sequence = fibonacci(num)
print("A sequencia de Fibonacci ate " + str(num) + " e: "+ str(fib_sequence))

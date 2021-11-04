print('hello world')

print('test')
def naiseki(a, b):
    sum = 0
    for i in range(len(a)):
            sum += a[i] * b[i]

    return sum

a = [2,3,4,4]
b = [5,2,2,7]
print(naiseki(a, b))
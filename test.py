from multiprocessing import Pool
import time

def f(x, bias):
    sum = x + bias
    return sum

xs = [1,2,3,4]
bias = [100,100,100,100]
t = zip(xs,bias)

t0 = time.time()
pool = Pool(12)
results = pool.map(f, xs, bias)
pool.close()
pool.join()
t1 = time.time()
print(t1-t0, results)

results = []
t0 = time.time()
for x in xs:
    results.append(f(x))
t1 = time.time()
print(t1-t0, results)

t0 = time.time()
pool = Pool(4)
results = pool.map(f, xs, bias)
pool.close()
pool.join()
t1 = time.time()
print(t1-t0, results)
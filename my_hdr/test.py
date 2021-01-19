import numpy as np 

a = np.array([[2,1,2],[6,0,-1],[100,3,29]])
fa = a.flatten()
# print(fa)
# print(np.argsort(fa))


def H(a):
    H, W = a.shape
    fa = a.flatten()
    sorted_index = np.argsort(fa)
    min_index= sorted_index[0]
    max_index= sorted_index[-1]
    mid_index= sorted_index[len(fa) // 2]
    min_r = min_index // W
    min_c = min_index % W
    max_r = max_index // W
    max_c = max_index % W
    mid_r = mid_index // W
    mid_c = mid_index % W
    return [[min_r, min_c, a[min_r,min_c]],[mid_r, mid_c, a[mid_r, mid_c]],[max_r, max_c, a[max_r, max_c]]]
# print(a)
# print(H(a))

a = np.array([[2,1,2],[6,0,-1],[100,3,29]])
print(a)
index = np.array([[0,0],[1,1]])
values = [a[i[0], i[1]] for i in index]
print(values)

g = [1,3,5,6,7,8,0,1]
def G(value):
    return g[value]



a = np.array([1,2,3])

print(np.vectorize(G)(a))

B = np.array([1,2,3,4])
B = B.reshape(4,1,1)
print(B)
C = np.ones((4,5,5))
C = C* B
print(C)
# BB = np.expand_dims(B, axis=)
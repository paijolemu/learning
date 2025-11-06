# %%
import itertools

data = ['A', 'B', 'C', 'D']
permutasi = itertools.permutations(data)

for i in permutasi:
    print(i)

# %%
data = ['A', 'B', 'C', 'D']
permutasi = itertools.permutations(data, 2)

for i in permutasi:
    print(i)

# %%
data = ['A', 'B', 'C', 'D']
permutasi = itertools.permutations(data, 1)

for i in permutasi:
    print(i)

# %%
# KALKULASI PERMUTASI
permutasi = itertools.permutations(range(4), 2)
print('total panjangnya adalah:', {len(tuple(permutasi))})
      
permutasi = itertools.permutations(range(10), 2)
print('total panjangnya adalah:', {len(tuple(permutasi))})
      
permutasi = itertools.permutations(range(43), 3)
print('total panjangnya adalah:', {len(tuple(permutasi))})


# %%
# KOMBINASI SEDERHANA
data = ['A', 'B', 'C', 'D']
combinasi = itertools.combinations(data, 3)

for i in combinasi:
    print(i)

# %%
# KALKULASI KOMBINASI
combinasi = itertools.combinations(range(4), 3)
print('total panjangnya adalah:', {len(tuple(combinasi))})

combinasi = itertools.combinations(range(16), 4)
print('total panjangnya adalah:', {len(tuple(combinasi))})

print(f'hasil range 5: {list(range(5))}')
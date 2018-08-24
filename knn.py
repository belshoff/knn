import math

'''
	Implementação do kNN
	dataset: https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
	Sobrevivência de pacientes submetidos a cirurgia de câncer de mama
'''

# a lista amostras é uma lista de listas
# cada lista corresponde a uma amostra
amostras = []

with open('dataset.data', 'r') as f:
	for linha in f.readlines():
		atrib = linha.replace('\n','').split(',')
		amostras.append([int(atrib[0]), int(atrib[1]),
						int(atrib[2]), int(atrib[3])])


# função que imprime algumas informações sobre os dados
def info_dataset(amostras, verbose=True):
	if verbose:
		print('Total de amostras: %d' % len(amostras))
	rotulo1, rotulo2 = 0, 0
	for amostra in amostras:
		if amostra[-1] == 1:
			rotulo1 += 1
		else:
			rotulo2 += 1
	if verbose:
		print('Total rotulo 1: %d' % rotulo1)
		print('Total rotulo 2: %d' % rotulo2)
	return [len(amostras), rotulo1, rotulo2]

p = 0.6
_, rotulo1, rotulo2 = info_dataset(amostras, verbose=False)
treinamento, teste = [], []
max_rotulo1, max_rotulo2 = int(p * rotulo1), int(p * rotulo2)
total_rotulo1, total_rotulo2 = 0, 0
for amostra in amostras:
	if (total_rotulo1 + total_rotulo2) < (max_rotulo1 + max_rotulo2):
		treinamento.append(amostra)
		if amostra[-1] == 1 and total_rotulo1 < max_rotulo1:
			total_rotulo1 += 1
		else:
			total_rotulo2 += 1
	else:
		teste.append(amostra)


def dist_euclidiana(v1, v2):
	dim, soma = len(v1), 0
	# dim - 1 para não pegar o atributo de saída
	for i in range(dim - 1):
		soma += math.pow(v1[i] - v2[i], 2)
	return math.sqrt(soma)

def knn(treinamento, nova_amostra, K):
	dists, tam_treino = {}, len(treinamento)
	for i in range(tam_treino):
		d = dist_euclidiana(treinamento[i], nova_amostra)
		dists[i] = d
	k_vizinhos = sorted(dists, key=dists.get)[:K]

	qtd_rotulo1, qtd_rotulo2 = 0, 0
	for indice in k_vizinhos:
		if treinamento[indice][-1] == 1:
			qtd_rotulo1 += 1
		else:
			qtd_rotulo2 += 1

	if qtd_rotulo1 > qtd_rotulo2:
		return 1
	else:
		return 2

# testando com uma amostra
#knn(treinamento, teste[0], K=13)

# testando com todas as amostras do conjunto de teste
acertos, K = 0, 15
for amostra in teste:
	classe = knn(treinamento, amostra, K)
	if amostra[-1] == classe:
		acertos += 1

print('Total de treinamento: %d' % len(treinamento))
print('Total de testes: %d' % len(teste))
print('Total de acertos: %d' % acertos)
print('Porcentagem de acertos: %.2f%%' % (100 * acertos / len(teste)))
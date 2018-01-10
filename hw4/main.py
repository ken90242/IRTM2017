from Vectorize import Vectorize
import pickle as pkl
import heapq
import random
import copy
import numpy as np
from tqdm import tqdm

def lazyDel(arr, del_docIdsArr):
	# arr = [(cosine, id), (cosine, id), ...]
	# 可加上 heapq._heapify_max
	return [(cosine, docId) for cosine, docId in arr if docId not in del_docIdsArr]

def getNewSim(docIdSetA, docIdSetB, cosine_dict):
	# Complete-Link
	smallest_sim = 99999
	for Aid in docIdSetA:
		for Bid in docIdSetB:
			if (cosine_dict[Aid][Bid] < smallest_sim): smallest_sim = cosine_dict[Aid][Bid]
	return smallest_sim

def selectHighestCosine(P, available):
	# P = { 1: [(cosine, id), (cosine, id), ...], 2: [(cosine, id), (cosine, id), ...], ... }
	# P[docId][0]: Heap最大的tuple:(cosine, id)，P[docId][0][0]：cosine
	idMaxCosList = [ P[docId][0][0] for docId in P ]
	MaxDocId = -1
	CurrentMax = -9999
	for i, val in enumerate(idMaxCosList):
		docId = i + 1 # 因為docId沒有0
		if (val > CurrentMax and available[docId] == True):
			CurrentMax = val
			MaxDocId = docId
	return MaxDocId


def EfficientHAC(N, k_max, cosine_dict):
	P = {}
	Clusters = { i: set({ i }) for i in range(1, N + 1) }
	Cluster_progress = { N: len(Clusters) }

	available = [True] * (N + 1) # 因為doc沒有0的id]
	for docId in tqdm(range(1, N + 1), desc='[4/5] Constructing the Heap of all docs\' similarity'):
		heap_list = []
		for targetId in range(1, N + 1):
			if (docId == targetId): continue
			heap_list.append((cosine_dict[docId][targetId], targetId))
		heapq._heapify_max(heap_list)
		P[docId] = heap_list

	for _ in tqdm(range(1, N - k_max + 1), desc='[5/5] Start Hierarchical Clustering (Complete-Link)'):
		docId = selectHighestCosine(P, available)

		sim, targetId = heapq.heappop(P[docId])

		Clusters[docId] = Clusters[docId].union(Clusters[targetId])
		del Clusters[targetId]
		Cluster_progress[len(Clusters)] = copy.deepcopy(Clusters) # 因為object是address-pointer

		available[targetId] = False

		P[docId] = []

		for i, existBool in enumerate(available):
			if (i == docId or i == 0 or existBool == False): continue
			P[i] = lazyDel(P[i], [docId, targetId])

			newMutualSim = getNewSim(Clusters[i], Clusters[docId], cosine_dict)

			cosine_dict[i][docId] = newMutualSim
			cosine_dict[docId][i] = newMutualSim

			P[i].append((cosine_dict[i][docId], docId))
			P[docId].append((cosine_dict[docId][i], i))
			# heapq._heapify_max(P[i])
			# heapq._heapify_max(P[docId])

	return Cluster_progress

if __name__ == "__main__":
	print('\nPreparing...\n')
	
	N = 1095
	cosine_dict = Vectorize(N=N).process()
	Cluster_progress = EfficientHAC(N=N, k_max=8, cosine_dict=cosine_dict)

	for idx, k in enumerate([8, 13, 20]):
		res = Cluster_progress[k]
		# print({x:len(res[x]) for x in res})
		content = ''
		for _, ids in res.items():
			for i in sorted(ids):
				content += '{}\n'.format(i)
			content += '\n'
		with open(str(k) + '.txt', 'w') as f:
			f.write(content.strip())

	print('\nDone.\n')

import pickle


file_path = 'chunks.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(len(data))
for i, item in enumerate(data):
    
    if (len(item) == 0):
        print(f"Chunk {i+1}:")
        print(len(item))

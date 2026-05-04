import pickle

file_path = './data/S10/S10.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Print top-level keys
for key in data.keys():
    print(f" - {key}")

# Print detailed structure of 'signal'
signal = data['signal']
print("\nChest channels:")
for ch, arr in signal['chest'].items():
    print(f"  {ch}: shape {arr.shape}, dtype {arr.dtype}")

print("\nWrist channels:")
for ch, arr in signal['wrist'].items():
    print(f"  {ch}: shape {arr.shape}, dtype {arr.dtype}")


label = data['label']
print("\nLabel array shape:", label.shape)
print("First 10 labels:", label[:10])

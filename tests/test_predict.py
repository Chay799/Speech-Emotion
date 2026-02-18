from src.inference.predict import predict_emotion, predict_folder

print("\n==== SINGLE FILE TEST ====\n")

# 🔹 Change this to any audio file you want
single_audio = "data/test/c8.2.anger.wav"

result = predict_emotion(single_audio)

print(f"File: {single_audio}")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Probabilities: {result['probabilities']}")


print("\n==== FOLDER TEST ====\n")

# 🔹 Change this to any folder you want
folder = "data/test"

results = predict_folder(folder)

for file, res in results:
    print(f"{file} → {res['emotion']} ({res['confidence']:.2f})")
    print("   probs:", res["probabilities"])

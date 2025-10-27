from vector_model import VectorModel

vm = VectorModel(fake=False)
vecs = vm.encode(["Hello world", "How are you?", "VectorModel works!"])
print(vecs)


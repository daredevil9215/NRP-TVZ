from neural_classes import *

# Ucitaj slike za trening
X = load_dataset("data", "train")
# Postavi zeljene ishode treninga
y = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],  #happy
              [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],  #sad
              [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],  #exploding
              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],  #skull
              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0],  #alien
              [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0],  #nerd
              [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]) #ghost

# Ucitaj slike za test
X_test = load_dataset("data", "test")
# Postavi zeljene ishode testa
y_test = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]])

# Incijalizacija modela
model = Model()
neurons = 512

model.add(Layer_Dense(X.shape[1], neurons))
model.add(Activation_ReLU())
model.add(Layer_Dense(neurons, neurons))
model.add(Activation_ReLU())
model.add(Layer_Dense(neurons, y.shape[1]))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(decay=1e-3), accuracy=Accuracy_Categorical())

# Potvrdi postavke mreze
model.finalize()

# Trening
model.train(X, y, epochs=300, print_every=1000)

#Ucitaj prvi set emojija
image = cv2.imread("data/img1.png" ,cv2.IMREAD_GRAYSCALE)
slika1 = slice_image(image)
y1 = np.array([[0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0]])

#Ucitaj drugi set emojija
image = cv2.imread("data/img2.png" ,cv2.IMREAD_GRAYSCALE)
slika2 = slice_image(image)
y2 = np.array([[1, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0], 
               [0, 0, 1, 0, 0, 0, 0], 
               [0, 0, 0, 1, 0, 0, 0], 
               [0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 1, 0], 
               [0, 0, 0, 0, 0, 0, 1]])

#Evaluiraj na setu slika
model.evaluate(slika1, y1)

# Ispis testiranja
for i in range(len(model.layers[5].output)):

    print(f'Predvidio sam: {kategorije[np.argmax(np.round(model.layers[5].output[i], 1))]}, a treba biti: {kategorije[np.argmax(y1[i])]}')

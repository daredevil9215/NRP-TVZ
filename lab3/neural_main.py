from neural_classes import *

#Inicijalizacija rječnika, 6 klasa po 5 rijeci
rjecnik1 = ['hendboll', 'gandbol', 'handbal', 'handbol', 'handbold']  #[1, 0, 0, 0, 0, 0]
rjecnik2 = ['football', 'fussball', 'futboll', 'futbola', 'futbol']   #[0, 1, 0, 0, 0, 0]
rjecnik3 = ['basketboll', 'basketbol', 'basketball', 'basketbal', 'basquetebol']     #[0, 0, 1, 0, 0, 0]
rjecnik4 = ['ball', 'baloia', 'bola', 'ballo', 'ballon']    #[0, 0, 0, 1, 0, 0]
rjecnik5 = ['racket', 'raketka', 'reket', 'raketa', 'raqueta']   #[0, 0, 0, 0, 1, 0]
rjecnik6 = ['puck', 'pak', 'puk', 'palet', 'pucul']    #[0, 0, 0, 0, 0, 1]
ukupno = rjecnik1 + rjecnik2 + rjecnik3 + rjecnik4 + rjecnik5 + rjecnik6

ukupno_test = ['handball', 'rakoomet', 
               'foldbold', 'futbolla', 
               'baloncesto', 'basketbols', 
               'pallo', 'ballun', 
               'raquette', 'rekket',
               'pac', 'pucel']

# Kodiranje rijeci u vektore (training)
X = napravi_matricu(ukupno, 15)

# Postavljanje zeljenih izlaza (training)
y = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], 
     [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
y = np.array(y)

# Kodiranje rijeci u vektore (test)
X_test = napravi_matricu(ukupno_test, 15)

# Postavljanje zeljenih izlaza (test)
y_test = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
y_test = np.array(y_test)

# Incijalizacija modela
model = Model()

model.add(Layer_Dense(X.shape[1], 8))
model.add(Activation_ReLU())
model.add(Layer_Dense(8, 8))
model.add(Activation_ReLU())
model.add(Layer_Dense(8, y.shape[1]))
model.add(Activation_Softmax())

#Optimizer_Adam(learning_rate=0.001, decay=1e-3)
model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(), accuracy=Accuracy_Categorical())

# Potvrdi postavke mreze
model.finalize()

model.train(X, y, epochs=10000, print_every=1000)

model.evaluate(X_test, y_test)

# Ispis testiranja
for i in range(len(model.layers[5].output)):

    print(f'Predvidio sam: {kategorije[np.argmax(np.round(model.layers[5].output[i], 1))]}, a treba biti: {kategorije[np.argmax(y_test[i])]}')






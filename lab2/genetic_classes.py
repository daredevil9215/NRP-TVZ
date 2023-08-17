import random
import numpy as np

# NOTAOKTAVA - TRAJANJE
notes = ['C1-1', 'C1#-1', 'D1-1', 'D1#-1', 'E1-1', 'F1-1', 'F1#-1', 'G1-1', 'G1#-1', 'A1-1', 'A1#-1', 'B1-1',
         'C2-1', 'C2#-1', 'D2-1', 'D2#-1', 'E2-1', 'F2-1', 'F2#-1', 'G2-1', 'G2#-1', 'A2-1', 'A2#-1', 'B2-1',
         'C1-2', 'C1#-2', 'D1-2', 'D1#-2', 'E1-2', 'F1-2', 'F1#-2', 'G1-2', 'G1#-2', 'A1-2', 'A1#-2', 'B1-2',
         'C2-2', 'C2#-2', 'D2-2', 'D2#-2', 'E2-2', 'F2-2', 'F2#-2', 'G2-2', 'G2#-2', 'A2-2', 'A2#-2', 'B2-2',]

class Songs:

    def __init__(self, num_of_songs, length):

        self.chromosomes = []

        for i in range(num_of_songs):

            temp_song = []

            for j in range(length):

                temp_song.append(notes[random.randint(0, len(notes) - 1)])

            self.chromosomes.append(temp_song)

        self.chromosomes = np.array(self.chromosomes)

    def set_chromosomes(self, chromos):
        self.chromosomes = chromos

def fitness_function(songs, result):

    rez = []

    for song in songs.chromosomes:

        temp = []

        for i in range(len(song)):

            temp.append((notes.index(song[i]) - notes.index(result[i]))**2)

        temp = np.array(temp)
        rez.append([np.sum(temp)])

    return np.array(rez)

#Nadklasa za selekciju
class Selection:

    def activate(self, chromosomes, fitness_result):
        return self.select(chromosomes, fitness_result)

#Klasa za implementaciju K-turnirske selekcije
class K_Tournament(Selection):

    #Konstruktor s parametrom k
    def __init__(self, *, k):
        self.k = k

    #Metoda koja obavlja selekciju
    def select(self, songs, fitness_result):

        contestants = []

        #Odaberi nasumicno k natjecatelja
        for i in range(self.k):
            random_index = np.random.randint(0, len(songs.chromosomes))
            contestants.append(random_index)

        best_index = 0
        best_value = 999999

        #Pretrazi i vrati natjecatelja s najboljim rezultatom fitness funkcije
        for index in contestants:
            if fitness_result[index] < best_value:
                best_index = index
                best_value = fitness_result[index]

        return songs.chromosomes[best_index]

#Nadklasa za rekombinaciju
class Crossover:

    def activate(self, parent_1, parent_2):
        return self.crossover(parent_1, parent_2)

#Klasa za implementaciju rekombiniranja u jednoj tocki
class Single_Point_Crossover(Crossover):

    def __init__(self, *, crossover_rate):
        self.crossover_rate = crossover_rate

    def crossover(self, parent_1, parent_2):

        prob = np.random.random()
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        if self.crossover_rate > prob:

        #Nasumicno nadi tocku rekombinacije
            crossover_point = np.random.randint(0, len(parent_1) - 1)

            child_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
            child_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
    
        return child_1, child_2

#Klasa za implementaciju rekombiniranja u dvije tocke
class Two_Point_Crossover(Crossover):

    def __init__(self, *, crossover_rate):
        self.crossover_rate = crossover_rate

    def crossover(self, parent_1, parent_2):

        prob = np.random.random()
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        if self.crossover_rate > prob:

        #Nasumicno nadi tocke rekombinacije
            crossover_point_1 = np.random.randint(0, len(parent_1) - 2)
            crossover_point_2 = np.random.randint(crossover_point_1 + 1, len(parent_1) - 1)

            child_1 = np.concatenate((parent_1[:crossover_point_1], parent_2[crossover_point_1:crossover_point_2], parent_1[crossover_point_2:]))
            child_2 = np.concatenate((parent_2[:crossover_point_1], parent_1[crossover_point_1:crossover_point_2], parent_2[crossover_point_2:]))
    
        return child_1, child_2

#Nadklasa za mutaciju
class Mutation:

    def activate(self, child):
        return self.mutate(child)

#Klasa za implementaciju mutacije nasumicnog gena u kromosomu
class Bit_Flip_Mutation(Mutation):

    #Konstruktor s parametrom mutation_rate
    def __init__(self, *, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, chromosome):

        prob = np.random.random()
        chrmsm = chromosome.copy()

        if self.mutation_rate > prob:

                #Promijeni gen u njegov komplement
                i = np.random.randint(0, len(chrmsm))
                chrmsm[i] = notes[(len(notes) - 1) - notes.index(chrmsm[i])]

        return chrmsm

#Klasa za implementaciju mutacije kromosoma u njegov komplement
class Inverse_Mutation(Mutation):

    #Konstruktor s parametrom mutation_rate
    def __init__(self, *, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, chromosome):

        prob = np.random.random()
        chrmsm = chromosome.copy()

        if self.mutation_rate > prob:

            for i in range(len(chrmsm)):
                chrmsm[i] = notes[(len(notes) - 1) - notes.index(chrmsm[i])]

        return chrmsm

#Klasa za implementaciju mutacije kromosoma nasumicno
class Random_Mutation(Mutation):

     #Konstruktor s parametrom mutation_rate
    def __init__(self, *, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, chromosome):

        prob = np.random.random()
        chrmsm = chromosome.copy()

        if self.mutation_rate > prob:

            for i in range(len(chrmsm)):
                chrmsm[i] = notes[np.random.randint(0, len(notes))]

        return chrmsm

#Metoda za obavljanje selekcije, rekombinacije i mutacije
def selection(chromosomes, fitness_result, *, selection_fun, crossover_fun, mutation_fun, elitism=False):

    selekcija = []

    if(elitism):

        elite = 0
        elite_value = 99999

        #Pronadi jedinku s najboljim rezultatom fitness funkcije
        for i in range(len(fitness_result)):
            if(fitness_result[i] < elite_value):
                elite_value = fitness_result[i]
                elite = i

        print(f'Elita je {chromosomes.chromosomes[elite]} s vrijednosti {elite_value}')
        kopija = chromosomes.chromosomes[elite].copy()

        #Pohrani najbolju jedinku u novu generaciju
        selekcija.append(kopija)

        if(chromosomes.chromosomes.shape[0] % 2 == 1):

            #Odaberi, rekombiniraj i mutiraj nove jedinke i pohrani u novu generaciju onoliko puta koliko ih je bilo u prethodnoj generaciji
           for i in range(chromosomes.chromosomes.shape[0] // 2):
            par_1 = kopija.copy()
            par_2 = selection_fun.activate(chromosomes, fitness_result)
            chld_1, chld_2 = crossover_fun.activate(par_1, par_2)
            chld_1 = mutation_fun.activate(chld_1)
            chld_2 = mutation_fun.activate(chld_2)
            selekcija.append(chld_1) 
            selekcija.append(chld_2)

        else:
            for i in range(chromosomes.chromosomes.shape[0] // 2 - 1):
                par_1 = kopija.copy()
                par_2 = selection_fun.activate(chromosomes, fitness_result)
                chld_1, chld_2 = crossover_fun.activate(par_1, par_2)
                chld_1 = mutation_fun.activate(chld_1)
                chld_2 = mutation_fun.activate(chld_2)
                selekcija.append(chld_1) 
                selekcija.append(chld_2)
            


    else:

        for i in range(chromosomes.chromosomes.shape[0] // 2):

            par_1 = selection_fun.activate(chromosomes, fitness_result)
            par_2 = selection_fun.activate(chromosomes, fitness_result)
            chld_1, chld_2 = crossover_fun.activate(par_1, par_2)
            chld_1 = mutation_fun.activate(chld_1)
            chld_2 = mutation_fun.activate(chld_2)
            selekcija.append(chld_1) 
            selekcija.append(chld_2)

        if(chromosomes.chromosomes.shape[0] % 2 == 1):
            par_1 = selection_fun.activate(chromosomes, fitness_result)
            par_2 = selection_fun.activate(chromosomes, fitness_result)
            chld_1, chld_2 = crossover_fun.activate(par_1, par_2)
            chld_1 = mutation_fun.activate(chld_1)
            selekcija.append(chld_1)

    selekcija = np.array(selekcija)
    return selekcija
#https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
import time
from genetic_classes import *

#Parametri
max_item_weight = 10
max_value = 100
max_backpack_weight = 9000
N = 1000
num_of_backpacks = 10

#Nasumicna inicijalizacija predmeta i populacije ruksaka
predmeti = Items(num_of_items=N, max_weight=max_item_weight, max_value=max_value)
ruksaci = Backpacks(num_of_backpacks, N)

start_time = time.time()

print(f'Prvotna populacija:\n {ruksaci.chromosomes}')

#Prvo izracunavanje vrijednosti fitness funkcije
rez = fitness_function(predmeti, ruksaci, max_weight=max_backpack_weight)
print(f'Fitness funkcija prije selekcije:\n {rez}')

generations = 0
value_set = None

while(True):

    print(f'Generacija: {generations}')
    
    #Odaberi jedinke za novu populaciju
    selekcija = selection(ruksaci, rez, selection_fun=K_Tournament(k=2), crossover_fun=Single_Point_Crossover(crossover_rate=0.7), 
                          mutation_fun=Bit_Flip_Mutation(mutation_rate=0.3), elitism=False)
    #Postavi nove jedinke
    ruksaci.set_chromosomes(selekcija)

    print(f'Nova populacija nakon selekcije:\n {selekcija}')

    #Izracunaj fitness vrijednosti za novu populaciju
    rez = fitness_function(predmeti, ruksaci, max_weight=max_backpack_weight)

    print(f'Fitness funkcija poslije selekcije:\n {rez}')

    #Inicijalizacija praznog seta
    value_set = np.unique(rez)

    #Ako je u setu samo jedna vrijednost prekini petlju
    if(len(value_set) == 1):
        break

    generations += 1

print(f'Najbolja vrijednost {value_set[0]} je postignuta ruksakom {ruksaci.chromosomes[0]} nakon {generations} generacija')

print("--- %s seconds ---" % (time.time() - start_time))


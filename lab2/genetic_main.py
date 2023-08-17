from genetic_classes import *
import time
import musicalbeeps

N = 10
M = 10
player = musicalbeeps.Player(volume = 0.3, mute_output = False)
a = []

#Nasumicno inicijaliziraj trazenu melodiju
for i in range(N):
    a.append(notes[np.random.randint(0, len(notes))])

#Inicijaliziraj populaciju
songs = Songs(M, N)

start_time = time.time()

print(f'Prvotna populacija:\n {songs.chromosomes}')

#Prvo izracunavanje vrijednosti fitness funkcije
rez = fitness_function(songs, a)
print(f'Fitness funkcija prije selekcije:\n {rez}')

generations = 0
vrti = True

while(vrti):

    print(f'Generacija: {generations}')
    
    #Odaberi jedinke za novu populaciju
    selekcija = selection(songs, rez, selection_fun=K_Tournament(k=2), crossover_fun=Two_Point_Crossover(crossover_rate=0.6), 
                          mutation_fun=Random_Mutation(mutation_rate=0.9), elitism=True)
    #Postavi nove jedinke
    songs.set_chromosomes(selekcija)

    print(f'Nova populacija nakon selekcije:\n {selekcija}')

    #Izracunaj fitness vrijednosti za novu populaciju
    rez = fitness_function(songs, a)

    print(f'Fitness funkcija poslije selekcije:\n {rez}')

    generations += 1

    if(not generations % 1000):

        # Pronadi i odsviraj najbolju trenutnu melodiju
        best = selekcija[selekcija.argmin()]
        for nota in best:
            a, b = nota.split('-')
            player.play_note(a, 0.5 / int(b))        

    #Ako je u polju rezultata najmanji element 0 prekini petlju
    if(rez.min() == 0):
        vrti = False

    

print(f'Najbolja vrijednost {rez[rez.argmin()]} je postignuta pjesmicom {songs.chromosomes[rez.argmin()]} nakon {generations} generacija')
print(f'Trazena pjesmica je {a}')

print("--- %s seconds ---" % (time.time() - start_time))
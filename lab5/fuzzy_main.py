import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedenti
suigraci = ctrl.Antecedent(np.arange(0, 12, 1), 'suigraci') #max 11
raspolozivo_vrijeme = ctrl.Antecedent(np.arange(0, 91, 1), 'raspolozivo_vrijeme') #max 90 min
intenzitet_aktivnosti = ctrl.Antecedent(np.arange(0, 11, 1), 'intenzitet_aktivnosti') #max 10
sklonost_fiz_kontakt = ctrl.Antecedent(np.arange(0, 11, 1), 'sklonost_fizickom_kontaktu') #max 10
kondicija = ctrl.Antecedent(np.arange(0, 11, 1), 'kondicija') #max 10

# Konzekvent
sport = ctrl.Consequent(np.arange(0, 15, 1), 'sport')

# Postavljanje funkcija pripadnosti
suigraci['malo'] = fuzz.trimf(suigraci.universe, [0, 0, 5])
suigraci['srednje'] = fuzz.trapmf(suigraci.universe, [4, 6, 7, 9])
suigraci['mnogo'] = fuzz.trimf(suigraci.universe, [8, 11, 11])

raspolozivo_vrijeme['kratko'] = fuzz.trimf(raspolozivo_vrijeme.universe, [0, 0, 35])
raspolozivo_vrijeme['srednje'] = fuzz.trapmf(raspolozivo_vrijeme.universe, [25, 45, 55, 70])
raspolozivo_vrijeme['dugo'] = fuzz.trimf(raspolozivo_vrijeme.universe, [60, 90, 90])

intenzitet_aktivnosti['lagan'] = fuzz.trimf(intenzitet_aktivnosti.universe, [0, 0, 4])
intenzitet_aktivnosti['srednji'] = fuzz.trapmf(intenzitet_aktivnosti.universe, [3, 5, 6, 8])
intenzitet_aktivnosti['visok'] = fuzz.trimf(intenzitet_aktivnosti.universe, [7, 10, 10])

sklonost_fiz_kontakt['mala'] = fuzz.trimf(sklonost_fiz_kontakt.universe, [0, 0, 4])
sklonost_fiz_kontakt['srednja'] = fuzz.trapmf(sklonost_fiz_kontakt.universe, [3, 5, 6, 8])
sklonost_fiz_kontakt['velika'] = fuzz.trimf(sklonost_fiz_kontakt.universe, [7, 10, 10])

kondicija['slaba'] = fuzz.trimf(kondicija.universe, [0, 0, 3])
kondicija['srednja'] = fuzz.trapmf(kondicija.universe, [2, 4, 6, 8])
kondicija['velika'] = fuzz.trimf(kondicija.universe, [6, 10, 10])

sport['golf'] = fuzz.trimf(sport.universe, [0, 1, 2])
sport['tenis'] = fuzz.trimf(sport.universe, [2, 3, 4])
sport['nogomet'] = fuzz.trimf(sport.universe, [4, 5, 6])
sport['kosarka'] = fuzz.trimf(sport.universe, [6, 7, 8])
sport['bicikliranje'] = fuzz.trimf(sport.universe, [8, 9, 10])
sport['boks'] = fuzz.trimf(sport.universe, [10, 11, 12])
sport['odbojka'] = fuzz.trimf(sport.universe, [12, 13, 14])

# Postavljanje pravila
rule1 = ctrl.Rule(suigraci['malo'] & raspolozivo_vrijeme['srednje'] & intenzitet_aktivnosti['srednji'] & sklonost_fiz_kontakt['mala'] & kondicija['srednja'], sport['tenis'])
rule2 = ctrl.Rule(suigraci['malo'] & raspolozivo_vrijeme['kratko'] & intenzitet_aktivnosti['lagan'] & sklonost_fiz_kontakt['mala'] & kondicija['slaba'], sport['golf'])
rule3 = ctrl.Rule(suigraci['mnogo'] & raspolozivo_vrijeme['dugo'] & intenzitet_aktivnosti['visok'] & sklonost_fiz_kontakt['srednja'] & kondicija['velika'], sport['nogomet'])
rule4 = ctrl.Rule(suigraci['srednje'] & raspolozivo_vrijeme['srednje'] & intenzitet_aktivnosti['visok'] & sklonost_fiz_kontakt['velika'] & kondicija['velika'], sport['kosarka'])
rule5 = ctrl.Rule(suigraci['malo'] & raspolozivo_vrijeme['dugo'] & intenzitet_aktivnosti['srednji'] & sklonost_fiz_kontakt['mala'] & kondicija['velika'], sport['bicikliranje'])
rule6 = ctrl.Rule(suigraci['malo'] & raspolozivo_vrijeme['kratko'] & intenzitet_aktivnosti['visok'] & sklonost_fiz_kontakt['velika'] & kondicija['velika'], sport['boks'])
rule7 = ctrl.Rule(suigraci['mnogo'] & raspolozivo_vrijeme['srednje'] & intenzitet_aktivnosti['visok'] & sklonost_fiz_kontakt['mala'] & kondicija['slaba'], sport['odbojka'])

#Inicijalizacija fuzzy sustava
sport_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
sport_decision = ctrl.ControlSystemSimulation(sport_ctrl)

#Postavljanje ulaznih varijabli
sport_decision.input['suigraci'] = 9
sport_decision.input['raspolozivo_vrijeme'] = 50
sport_decision.input['intenzitet_aktivnosti'] = 7.9
sport_decision.input['sklonost_fizickom_kontaktu'] = 2.1
sport_decision.input['kondicija'] = 0.6

# Izracun centroid
sport_decision.compute()

# Ispis rezultata
print(sport_decision.output['sport'])
sport.view(sim=sport_decision)
plt.show()

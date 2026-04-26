## Projet Valeurs Extrêmes Zergaoui Kélian

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import os
import csv

## Etape 1: lire les documents csv et les nettoyer

#On va utiliser des documents csv issus de la Banque Centrale Européenne

#Italie

chemin_italie=r"C:\Users\kelia\Desktop\CENTRALE MED 1A\2A\S8\Modélisation maths\Mitra\Projet final\Italie.csv"


df_italie=pd.read_csv(chemin_italie, sep=';') #pd.read_csv créer directement un dataframe
#print(italie.head())

df_italie = df_italie.rename(columns={
    "Observation date": "date",
    "Gross yield of benchmark 10-year BTP": "taux_BTP"
})
#les obligations italiennes se nomment BTP

df_italie["date"]=pd.to_datetime(df_italie["date"],errors="coerce")

df_italie["taux_BTP"]=pd.to_numeric(df_italie["taux_BTP"],errors="coerce")

df_italie=df_italie.dropna()

#print(df_italie)
#print(df_italie.dtypes)

df_italie=df_italie.sort_values(by='date')
df_italie = df_italie.reset_index(drop=True)

#print(df_italie.shape)

#on obtient un dataframe avec 8880 lignes et 2 colonnes

#Allemagne

chemin_allemagne=r"C:\Users\kelia\Desktop\CENTRALE MED 1A\2A\S8\Modélisation maths\Mitra\Projet final\Allemagne.csv"

df_allemagne=pd.read_csv(chemin_allemagne,header=None)
# print(df_allemagne.head(15))
# print(df_allemagne.columns)
# print(df_allemagne[0])
# print(df_allemagne[1])
# print(df_allemagne[2])

df_allemagne=df_allemagne.iloc[9:]
df_allemagne=df_allemagne.drop(columns=[2])
#print(df_allemagne.head(15))

df_allemagne[1]=df_allemagne[1].replace('.',np.nan)
#print(df_allemagne)

df_allemagne=df_allemagne.dropna(subset=[1])

df_allemagne=df_allemagne.rename(columns={
        0: "date",
        1: "taux_BUND"})
#les obligations allemandes se nomment Bund

df_allemagne=df_allemagne.sort_values(by='date')
df_allemagne=df_allemagne.reset_index(drop=True)

df_allemagne["date"] = pd.to_datetime(df_allemagne["date"], errors="coerce")
df_allemagne["taux_BUND"]=pd.to_numeric(df_allemagne["taux_BUND"],
errors="coerce")

print(df_allemagne)

##
# à ce stade, les tableaux sont nettoyés mais pour être comparables ils doivent avoir la même taille, et même plus que ça, les dates doivent correspondre

print(df_allemagne.shape) #(6219, 2)
print(df_italie.shape) # (8880, 2)

##

# print(df_italie.head(15))
# print(df_allemagne.head(15))

#Pour gérer ce problème, on fusionne les 2 tableaux, et la jointure se fait sur les dates

# Or pour cela: il faut que les deux colonnes de jointure aient le même type: ici c'est bien le cas car on a fait to_datetime avant sur les colonnes "date"

#print(df_italie["date"].dtype) #datetime64[ns]
#print(df_allemagne["date"].dtype) #datetime64[ns]

df_taux = pd.merge(df_italie, df_allemagne,on="date", how="inner")
#inner sert à garder seulement les dates communes aux deux dataframes

print(df_taux.head(15))


## Etape 2: Construction du spread Italie-Allemagne et de ses variations

#spread
df_taux["spread"]=df_taux["taux_BTP"]-df_taux["taux_BUND"]

#variation quotidienne du spread
df_taux["dspread"]=df_taux["spread"].diff()

#print(df_taux.head(10))
#la première valeur de la colonne dspread est NaN car elle fait la différence entre "spread" et un flotant...

df_taux = df_taux.dropna(subset=["dspread"])
df_taux=df_taux.reset_index(drop=True)
print(df_taux)

##Etape 3: Analyse exploratoire du spread et des variations du spread

#BTP & BUND

plt.figure(0)
plt.plot(df_taux["date"].to_numpy(),df_taux["taux_BTP"].to_numpy(),label="taux BTP (Ita)")
plt.plot(df_taux["date"].to_numpy(),df_taux["taux_BUND"].to_numpy(), label="taux Bund (All)")
plt.title("Taux obligataires Italiens et Allemands sur plus de 10 ans")
plt.xlabel("dates")
plt.ylabel("taux")
plt.legend()
plt.grid(True)
plt.show()

#spread

plt.figure(1)
plt.plot(df_taux["date"].to_numpy(), df_taux["spread"].to_numpy(), label="Spread Italie-Allemagne")
plt.title("Spread Italie-Allemagne sur plus de 10 ans")
plt.ylabel("Spread")
plt.xlabel("dates")
plt.legend()
plt.grid(True)
plt.show()

#dspread plot

plt.figure(2)
plt.plot(df_taux["date"].to_numpy(), df_taux["dspread"].to_numpy(), label="dspread Italie-Allemagne")
plt.title("Variation quotidienne du spread sur plus de 10 ans")
plt.ylabel("Delta Spread ou dspread")
plt.xlabel("dates")
plt.legend()
plt.grid(True)
plt.show()

#dspread hist

plt.figure(3)
plt.hist(df_taux["dspread"].to_numpy(),bins=100, edgecolor="black")
plt.title("Histogramme de la variation quotidienne du spread sur plus de 10 ans")
plt.xlabel("Delta Spread ou dspread")
plt.ylabel("Nombre d’observations")
plt.grid(True)
plt.show()

#On obtient un histogramme avec un nombre d'observations elevées en 0 ce qui traduit que l'égalité des taux est assez récurente sur plusd de 10 ans, ce à quoi on pouvait s'attendre.

## Etape 4.1: Méthode des blocs maximum sur la série dspread

#On voit apparaître sur l'histogramme et sur le graphique du dspread des valeurs isolées ou impulsionelles qui donnent envie d'aller analyser ceci à l'aide de la théorie des valeurs extrêmes

#on choisit donc la variable extrême à étudier: le dspread

#on découpe la série en blocs temporels:
#par exemple des blocs mensuels

maxima_mensuels=df_taux.set_index("date")["dspread"].resample("M").max()

#resample(M) permet de faire des blocs mensuels

print(maxima_mensuels.head(15))
print(maxima_mensuels.shape) #on obtient 293 maximums sur 293 mois

plt.figure(4)
plt.plot(maxima_mensuels.index.to_numpy(),maxima_mensuels.to_numpy(), label="maxima mensuels")
plt.title("Maxima des blocs mensuels sur plus de 10 ans")
plt.ylabel("Maximum")
plt.xlabel("Mois")
plt.legend()
plt.grid(True)
plt.show()

#on se concentre sur les élargissements extrêmes du spread, plus directement liés aux épisodes de stress ; une étude symétrique des resserrements pourrait être menée en appliquant la même méthode à -dspread

##Étape 4.2.1 : Estimation des paramètres de la GEV par maximum de vraisemblance
import scipy.stats as st

param_gev=st.genextreme.fit(maxima_mensuels)


gamma_hat_ml=-param_gev[0]
mu_hat_ml=param_gev[1]
sigma_hat_ml=param_gev[2]

print("Pour la méthode du maximum de vraisemblance, on obtient:","\n")

print("paramètre de forme ou de type de queu gamma_hat_ml:", gamma_hat_ml)
print("paramètre de localisation mu_hat_ml:", mu_hat_ml)
print("paramètre d'échelle sigma_hat_ml:", sigma_hat_ml)

#on a bien sigma_hat>0

##Étape 4.2.2 : Estimation des paramètres de la GEV par la méthode des moments pondérés

maxima_tries = np.sort(maxima_mensuels.to_numpy())

n=len(maxima_tries)
i=np.arange(1,n+1)

def calcul_moment(r):
    return(1/n*np.sum(maxima_tries*((i-1)/n)**r))

b0=calcul_moment(0)
b1=calcul_moment(1)
b2=calcul_moment(2)

print("b0:", b0)
print("b1:", b1)
print("b2:",b2)

#maitenant on va déterminer lambda, gamma et mu
from scipy.special import gamma as gamma_function
from scipy.optimize import minimize_scalar

# gamma_hat

#On fait les calculs au brouillons avec l'expression donnée dans le cours

R=(3*b2-b0)/(2*b1-b0) #par le calcul on trouve que R=(3^gamma-1)/(2^gamma-1)

#on va donc poser une fonction et chercher à la minimser pour trouver gamma

def rapport_theorique(g):
    epsilon=1e-6
    if abs(g) < epsilon:
        return np.log(3)/np.log(2)
    else:
        return ((3**g-1)/(2**g-1))

def f(g):
    return((rapport_theorique(g)-R)**2)

res=minimize_scalar(f)
gamma_hat_pwm=res.x

print("gamma_hat_pwm:", gamma_hat_pwm)

#sigma_hat

#par le calcul on trouve que si on pose A=2*b1-b0 alors on peut calculer sigma_hat avec A, gamma et la fonction Gamma

A = 2*b1 - b0

sigma_hat_pwm = A * gamma_hat_pwm / (gamma_function(1 - gamma_hat_pwm) * (2**gamma_hat_pwm - 1))

print("sigma_hat_pwm:",sigma_hat_pwm)

#mu_hat

mu_hat_pwm = b0 + (sigma_hat_pwm / gamma_hat_pwm) * (1 - gamma_function(1 - gpamma_hat_pwm))

print("mu_hat_pwm:", mu_hat_pwm)

## Etape 4.3: QQ-plot et diagnostic de l’ajustement GEV

maxima_tries = np.sort(maxima_mensuels.to_numpy())

n=len(maxima_tries)
i=np.arange(1,n+1)

p_i=(i-0.5)/n

q_theo_ml=st.genextreme.ppf(p_i,-gamma_hat_ml,loc=mu_hat_ml,scale=sigma_hat_ml)

mn = min(np.min(q_theo_ml), np.min(maxima_tries))
mx = max(np.max(q_theo_ml), np.max(maxima_tries))

x=np.linspace(mn,mx,10)

plt.figure(5)
plt.scatter(q_theo_ml,maxima_tries)
plt.plot(x,x,color="red")
plt.xlabel("quantiles théoriques")
plt.ylabel("quantiles empiriques")
plt.title("QQ-plot GEV (estimation ML)")
plt.grid(True)
plt.show()

#Un écart apparaît pour les plus grandes valeurs, ce qui suggère une surestimation de la queue droite par l’estimation du maximum de vraisemblance.

##Étape 4.4: Estimation des quantiles extrêmes et des niveaux de retour avec la GEV

#Le cours définit justement le return level RL(T) par:
# 1-F(RL(T))=1/T

#Donc RL(T)=F^{-1}(1-1/T)

T=np.array([12,24,60,120,200,250]) #nombre de mois

p=1-1/T

RL_ml=st.genextreme.ppf(p,-gamma_hat_ml,loc=mu_hat_ml,scale=sigma_hat_ml)

RL_pwm=st.genextreme.ppf(p,-gamma_hat_pwm, loc=mu_hat_pwm, scale=sigma_hat_pwm)

plt.figure(6)
plt.plot(T,RL_ml,color="blue", marker="o", linestyle="--", label="RL_ml")
plt.plot(T,RL_pwm,color="red", marker="o", linestyle="--", label="RL_pwm")
plt.title("Comparaison des niveaux de retour GEV: ML & PWM")
plt.ylabel("Niveau de retour")
plt.xlabel("Période de retour T (mois)")
plt.legend()
plt.grid(True)
plt.show()

#les deux méthodes donnent des niveaux de retours croissants avec la période de retour T, mais l'estimation par maximum de vraisemblance (ML) montre des niveaux plus élevés que la méthode des moments pondérés (pwm), en particulier pour des grandes valeurs de T.

##Etape 4.5: Verification numérique de la normalité asymptotique.

#Le cours dit que les estimateurs GEV sont asymptotiquements gaussiens dans certaines conditions.On va essayer de le vérifier numériquement avec un bootstrap.

## 4.5.1 Bootstrap des estimateurs GEV
B=500

np.random.seed(45)
m=len(maxima_mensuels)

gev_par_ml=[]

for _ in range (B):
    echantillon=np.random.choice(maxima_mensuels.to_numpy(),m,replace=True)
    gev_echantillon=st.genextreme.fit(echantillon)
    gamma=-gev_echantillon[0]
    mu=gev_echantillon[1]
    sigma=gev_echantillon[2]
    gev_par_ml.append([gamma,mu,sigma])

gev_par_ml=np.array(gev_par_ml)
print(gev_par_ml.shape)

## 4.5.2 : Histogrammes des estimateurs bootstrapés

gamma_boot=gev_par_ml[:,0]
mu_boot=gev_par_ml[:,1]
sigma_boot=gev_par_ml[:,2]

# print(gamma_boot.shape)
# print(mu_boot.shape)
# print(sigma_boot.shape)

#gamma
plt.figure(7)
plt.hist(gamma_boot,bins=60, color="blue", edgecolor="black")
plt.title("Estimation de gamma bootstrapé")
plt.ylabel("Nombre d'occurence sur 500 échantillons")
plt.xlabel("Valeurs de gamma")
plt.grid(True)
plt.show()

#mu
plt.figure(8)
plt.hist(mu_boot,bins=60, color="orange", edgecolor="black")
plt.title("Estimation de mu bootstrapé")
plt.ylabel("Nombre d'occurence sur 500 échantillons")
plt.xlabel("Valeurs de mu")
plt.grid(True)
plt.show()

#sigma
plt.figure(9)
plt.hist(sigma_boot,bins=60,color="green", edgecolor="black")
plt.title("Estimation de sigma bootstrapé")
plt.ylabel("Nombre d'occurence sur 500 échantillons")
plt.xlabel("Valeurs de sigma")
plt.grid(True)
plt.show()

##4.5.3 : Standardisation et QQ-plots normaux

#gamma

gamma_boot_moy=np.mean(gamma_boot)

gamma_boot_std=np.std(gamma_boot)

z_gamma=(gamma_boot-gamma_boot_moy)/(gamma_boot_std)

plt.figure(10)
st.probplot(z_gamma,dist="norm",plot=plt)
plt.title("QQ-plot normal de l'estimateur bootstrapé de gamma")
plt.grid(True)
plt.show()

#mu

mu_boot_moy=np.mean(mu_boot)
mu_boot_std=np.std(mu_boot)

z_mu=(mu_boot-mu_boot_moy)/(mu_boot_std)

plt.figure(11)
st.probplot(z_mu,dist="norm",plot=plt)
plt.title("QQ-plot normal de l'estimateur boostrapé de mu")
plt.grid(True)
plt.show()

#sigma

sigma_boot_moy=np.mean(sigma_boot)
sigma_boot_std=np.std(sigma_boot)

z_sigma=(sigma_boot-sigma_boot_moy)/(sigma_boot_std)

plt.figure(12)
st.probplot(z_sigma, dist="norm", plot=plt)
plt.title("QQ-plot normal de l'estimateur bootstrapéde sigma")
plt.grid(True)
plt.show()

#Les distributions bootstrapées standardisées de mu_hat et sigma_hat  sont proches d’une loi normale, tandis que celle de gamma_hat présente des écarts plus marqués dans la queue droite. Ainsi, la normalité asymptotique annoncée par le cours apparaît globalement plausible numériquement, mais elle est moins convaincante pour le paramètre de forme dans notre échantillon.


## Étape 5.1 : Choix du seuil pour la méthode des excès (GPD)

# On reprend le tracer de l'étape 3 pour déterminer un seuil graphiquement

#dspread plot
plt.figure(13)
plt.plot(df_taux["date"].to_numpy(), df_taux["dspread"].to_numpy(), label="dspread Italie-Allemagne")
plt.title("Variation quotidienne du spread sur plus de 10 ans")
plt.ylabel("Delta Spread ou dspread")
plt.xlabel("dates")
plt.legend()
plt.grid(True)
plt.show()

# On prendra un seuil u=0.20 pour les excès pour garder assez de "cas rare" pour estimer la GPD

u=0.20

##Etape 5.2 : Construction des excès au-dessus du seuil

dspread_l=df_taux["dspread"].to_numpy()

exces=[]

for delt in dspread_l:
    if delt>=u:
        exces.append(delt)

print("Nombre d'excès:", len(exces),"\n") #65

exces=np.array(exces)

Y=exces-u
print("Valeurs des écarts des excès par rapport au seuil:", Y,"\n")

alpha_gpd_hat=len(exces)/len(dspread_l)

print("Proportion des excès:", alpha_gpd_hat)

## Étape 5.3 : Estimation des paramètres de la GPD par ML

param_gpd=st.genpareto.fit(Y,floc=0)

gamma_gpd_ml=param_gpd[0]
mu_gpd_ml=param_gpd[1]
sigma_gpd_ml=param_gpd[2]

print("gamma_gpd_ml:", gamma_gpd_ml)
print("mu_gpd_ml:",mu_gpd_ml)
print("sigma_gpd_ml:", sigma_gpd_ml)

## Etape 5.4 :Estimation des paramètres de la GPD par la méthode des moments

# D'après le cours lorsque Y ~ GPD
#Alors E(Y)=sigma_gp/(1-gamma) et V(Y)=sigma_gp^2/((1-gamma)^2*1(1-2gamma))

#Donc on obtient en posant: E(Y)^2/V(Y)=R_y que gamma=(1-R_y)/2
#et sigma=E(Y)*(1-gamma)

Y_moy=np.mean(Y)
Y_var=np.var(Y)

R_y=Y_moy**2/Y_var

gamma_gpd_mom=(1-R_y)/2
sigma_gpd_mom=Y_moy*(1-gamma_gpd_mom)

print("gamma_gpd_mom:",gamma_gpd_mom)
print("sigma_gpd_mom:", sigma_gpd_mom)

## Etape 5.5 : Diagnostics graphiques GPD

## 5.5.1 QQ-plot GPD

Y_triee=np.sort(Y)
#print(Y_triee)

n=len(Y_triee)

i=np.arange(1,n+1)

p_i=(i-0.5)/n
#print(p_i)

q_theo_ml=st.genpareto.ppf(p_i,c=-gamma_gpd_ml,loc=0,scale=sigma_gpd_ml)
#print(q_theo_ml)

mn_ml=min(min(Y_triee),min(q_theo_ml))
mx_ml=max(max(Y_triee), max(q_theo_ml))

x=np.linspace(mn_ml,mx_ml,30)

plt.figure(14)
plt.scatter(Y_triee,q_theo_ml)
plt.plot(x,x,color="red")
plt.title("QQ-plot pour l'estimation de la GPD par ML")
plt.ylabel("quantiles theoriques")
plt.xlabel("quantiles experimentaux ")
plt.grid(True)
plt.show()

q_theo_mom=st.genpareto.ppf(p_i,c=-gamma_gpd_mom,loc=0,scale=sigma_gpd_mom)

mn_mom=min(min(Y_triee), min(q_theo_mom))
mx_mom=max(max(Y_triee), max(q_theo_mom))

u=np.linspace(mn_mom,mx_mom)

plt.figure(15)
plt.scatter(Y_triee,q_theo_mom)
plt.plot(u,u,color="red")
plt.title("QQ-plot pour l'estimation de la GPD par methode des moments")
plt.ylabel("quantiles theoriques")
plt.xlabel("quantiles experimentaux")
plt.grid(True)
plt.show()

## 5.5.2 Stabilité des paramètres selon le seuil


seuils=[0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32,0.33,0.34]

n_s=len(seuils)

les_exces=[]

for s in seuils:
    L=[]
    for delt in dspread_l:
        if delt>s:
            L.append(delt)
    les_exces.append(L)

#print(les_exces)



Y_stab=[]
for i in range (n_s):
    exc_i=np.array(les_exces[i])
    Y_stab.append(exc_i-seuils[i])

#print(Y_stab)

print("La proportion d'exces pour un seuil:")
for i in range (n_s):
    alpha_hat=0
    alpha_hat=len(Y_stab[i])/len(dspread_l)
    print (seuils[i], "est:", alpha_hat)

#GPD par ML pour les différents seuils

param_pour_differents_seuils=[]
for i in range (len(Y_stab)):
    param_gpd_ml=st.genpareto.fit(Y_stab[i],floc=0)
    param_pour_differents_seuils.append(param_gpd_ml)

#print(param_pour_differents_seuils)

param_pour_differents_seuils = np.array(param_pour_differents_seuils)

gamma_list=param_pour_differents_seuils[:,0]
sigma_list=param_pour_differents_seuils[:,2]

# print(gamma_list)
# print(sigma_list)


plt.figure(16)
plt.plot(seuils,gamma_list,color="green",marker="o",linestyle="--")
plt.title("Evolution de gamma en fonction du seuil (estimation ML)")
plt.xlabel("Seuil")
plt.ylabel("Gamma")
plt.grid(True)
plt.show()

plt.figure(17)
plt.plot(seuils,sigma_list,color="red", marker="o", linestyle="--")
plt.title("Evolution de sigma en fonction du seuil (estimation ML)")
plt.xlabel("Seuil")
plt.ylabel("Sigma")
plt.grid(True)
plt.show()

#Les graphes de stabilité suggèrent qu’un seuil autour de u=0.20–0.22 constitue un compromis raisonnable : les estimations de gamma et sigma y restent relativement "stables", alors qu’elles deviennent très "instables" pour des seuils supérieurs à 0.30, probablement en raison, peut-être, d’un nombre d’excès trop faible.

## Étape 5.6: Estimation des quantiles extrêmes et des niveaux de retour par la méthode POT

#On va essayer de définir une probabilité de queu en sachant que :

#250 jours ≈ 1 an boursier
# 500 jours ≈ 2 ans
# 1000 jours ≈ 4 ans
# 2500 jours ≈ 10 ans environ

T_jours=np.array([250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1700, 2000, 2200, 2500])

p_gpd= 1/T_jours

# On a u=0.20 (cf. 5.1)

# print(p_gpd)
# print(alpha_gpd_hat)
# print(p_gpd < alpha_gpd_hat)


#Comme gamma_gpd_ml est différent de 0:

q_gpd_ml=u+sigma_gpd_ml/gamma_gpd_ml*((p_gpd/alpha_gpd_hat)**(-gamma_gpd_ml)-1)

print(q_gpd_ml)


q_gpd_mom = u + sigma_gpd_mom/gamma_gpd_mom * ((p_gpd/alpha_gpd_hat)**(-gamma_gpd_mom) - 1)

print(q_gpd_mom)

plt.figure(18)
plt.plot(T_jours,q_gpd_ml,color="red",marker="o", linestyle="--",label="quantiles ml")
plt.plot(T_jours, q_gpd_mom, color="green", marker="o", linestyle="--",label="quantiles mom")
plt.title("Quantiles extrêmes POT : ML vs MOM")
plt.ylabel("Quantiles extrêmes")
plt.xlabel("Nombre de jours")
plt.legend()
plt.grid(True)
plt.show()

##Étape 5.7 : Vérification numérique de la normalité asymptotique des estimateurs GPD


# Bootstrap
C=500

#Rappel
u=0.20
Y=exces-u #(cf 5.1)

np.random.seed(45)
taille=len(Y)

gpd_par_ml=[]

for _ in range (C):
    echantillon=np.random.choice(Y,taille,replace=True)
    gpd_echantillon=st.genpareto.fit(echantillon, floc=0)
    gamma=gpd_echantillon[0]
    sigma=gpd_echantillon[2]
    gpd_par_ml.append([gamma,sigma])

gpd_par_ml=np.array(gpd_par_ml)
print(gpd_par_ml.shape)

##
gamma_gpd_boot=gpd_par_ml[:,0]
sigma_gpd_boot=gpd_par_ml[:,1]

gamma_gpd_boot_moy=np.mean(gamma_gpd_boot)
gamma_gpd_boot_std=np.std(gamma_gpd_boot)

sigma_gpd_boot_moy=np.mean(sigma_gpd_boot)
sigma_gpd_boot_std=np.std(sigma_gpd_boot)


z_gpd_gamma=(gamma_gpd_boot-gamma_gpd_boot_moy)/gamma_gpd_boot_std

plt.figure(19)
st.probplot(z_gpd_gamma,dist="norm",plot=plt)
plt.title("QQ-plot normalité asymptotique pour gamma")
plt.grid(True)
plt.show()

z_gpd_sigma=(sigma_gpd_boot-sigma_gpd_boot_moy)/sigma_gpd_boot_std

plt.figure(20)
st.probplot(z_gpd_sigma,dist="norm",plot=plt)
plt.title("QQ-plot normalité asymptotique pour sigma")
plt.grid(True)
plt.show()


## Étape 6 : Comparaison des approches GEV et GPD et interprétation financière

## 6.1 Comparaison des approches GEV et GPD

# Les approches block maxima (GEV) et peak over threshold (GPD) conduisent à des résultats globalement cohérents sur les extrêmes de la série dspread.

# La méthode GEV repose sur les maxima observés sur des blocs mensuels, tandis que la méthode GPD modélise directement les excès au-dessus d’un seuil élevé.

# La méthode GEV met en évidence une queue droite modérée mais un ajustement un peu moins précis sur les valeurs les plus extrêmes, tandis que la méthode POT fournit des résultats plus stables entre maximum de vraisemblance et méthode des moments pour le seuil retenu.

## 6.2 Interprétation financière

# Dans ce projet, les extrêmes positifs de dspread représentent des élargissements brusques du spread BTP-Bund.

# Financièrement, ils peuvent être interprétés comme des épisodes de stress, au cours desquels la prime de risque associée à la dette italienne augmente relativement à celle de la dette allemande.

# Dans un contexte repo et optimisation du collatéral, ces épisodes sont importants car ils peuvent influencer la valorisation des titres, le choix des haircuts/décotes, le coût du financement et la gestion du risque de portefeuille.

## 6.3 Limites de l’étude et conclusion

# Cette étude présente néanmoins certaines limites : le choix du seuil POT reste partiellement empirique, le nombre d’excès devient faible pour les seuils élevés, et l’hypothèse d’indépendance des observations n’est qu’approximative pour une série financière.

# En conclusion, l’étude des extrêmes de la variation quotidienne du spread Italie–Allemagne met en évidence des épisodes rares mais marqués d’élargissement du spread.

# Les méthodes GEV et GPD donnent des résultats cohérents, avec une bonne robustesse de la méthode POT pour le seuil retenu à partir des diagnostics graphiques.

# Ces extrêmes peuvent être interprétés comme des épisodes de tension de marché, ce qui donne un sens financier direct à l’analyse dans la perspective d’un stage en repo et optimisation du collatéral :-)
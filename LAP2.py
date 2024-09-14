#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:50:04 2024

@author: mathisturquet
"""

# Importation des modules
import numpy as np
import matplotlib.pyplot as plt

#%%MEC6616-Aérodynamique numérique


#Fait par:
#Nicolas Tran (2075141)
#Mathis Turquet (2403145)


#%% Assisgnation des constantes donnés dans l'énoncé

class parametres1():
    L = 1     # Longueur
    rho = 1 #kg/m^3
    Gamma = 0.1 #kg/m.s
    x_0 = 0
    x_f = L
    u= 0.1 # Vitesse m/s
    phi_A = 1  # Temperature à x=A
    phi_B = 0  # Temperature à x=B
    N =  5   #Nombre de noeuds
class parametres2():
    L = 1     # Longueur
    rho = 1 #kg/m^3
    Gamma = 0.1 #kg/m.s
    x_0 = 0
    x_f = L
    u= 2.5 # Vitesse m/s
    phi_A = 1  # Temperature à x=A
    phi_B = 0  # Temperature à x=B
    N =  5   #Nombre de noeuds    
class parametres3():
    L = 1     # Longueur
    rho = 1 #kg/m^3
    Gamma = 0.1 #kg/m.s
    x_0 = 0
    x_f = L
    u= 2.5 # Vitesse m/s
    phi_A = 1  # Temperature à x=A
    phi_B = 0  # Temperature à x=B
    N =  20    #Nombre de noeuds    
    
    
prm1 = parametres1()
prm2 = parametres2()
prm3 = parametres3()

#%% Fonctions
def ConvDiff1DCentré(x,prm,n):
    """Fonction 
    Entrées:
        - x : Vecteur (array de la position)
        - prm : Objet class parametres()
        - n : Nombre de noeuds
    Sortie:
        - Vecteur (array) composé du flux en fonction de la position
            en régime stationnaire
    """
    
    # Constantes
    L = prm.L
    dx = L/n
    phi_A = prm.phi_A
    phi_B = prm.phi_B
    rho=prm.rho
    u=prm.u
    Gamma=prm.Gamma
    F=rho*u
    D=Gamma/dx

    D_W=D
    F_W=F
    D_E=D
    F_E=F
    D_A=D
    F_A=F
    D_B=D
    F_B=F
    
    #Equations aux noeuds internes
    a_w_int=D_W+F_W/2
    a_e_int=D_E-F_E/2
    S_u_int=0
    a_p_int=a_w_int+a_e_int+(F_E-F_W)
    
    #Equations au noeud à la frontière gauche (A)
    a_w_A=0
    a_e_A=D_E-F_E/2
    S_p_A=-(2*D_A+F_A)
    S_u_A=(2*D_A+F_A)*phi_A
    a_p_A=a_w_A+a_e_A+(F_E-F_W)-S_p_A
    
    #Equations au noeud à la frontière droite (B)
    a_w_B=D_W+F_W/2
    a_e_B=0
    S_p_B=-(2*D_B-F_B)
    S_u_B=(2*D_B-F_B)*phi_B
    a_p_B=a_w_B+a_e_B-S_p_B
    
    # Créer la matrice de phi
    phi = np.zeros(n)

    # Conditions limites et matrice b
    b = np.zeros(n)
    b[0:n] = S_u_int
    b[0] = S_u_A
    b[-1] = S_u_B

    # Matrice A
    A = np.zeros((n,n))
    A[0,0] = a_p_A
    A[0,1] = -a_e_A
    A[-1,-1] = a_p_B
    A[-1,-2] = -a_w_B
    for i in range(1, n-1):
        A[i,i-1] = -a_w_int
        A[i,i] = a_p_int
        A[i,i+1] = -a_e_int

    # Résolution du système matriciel AT=b
    phi = np.linalg.solve(A,b)
    return phi


def analytique(x,prm,n):
    """Fonction 
    Entrées:
        - prm : Objet class parametres()
               - L : Longueur
               - T_A : Temperature à x=A
               - T_B : Temperature à x=B
               - k : Conductivité thermique  W/m.K
               - A : Section  m^2
        - N = Nombre de noeuds
    Sortie:
        - Vecteur (array) composé de la température en fonction de la position
            en régime stationnaire de la solution analytique
    """
    # Constantes
    L = prm.L
    phi_A = prm.phi_A
    phi_B = prm.phi_B
    rho=prm.rho
    u=prm.u
    Gamma=prm.Gamma

    
    phi = np.zeros(n)
    for i in range(n):
        phi[i]=(np.exp(rho*u*x[i]/Gamma)-1)/(np.exp(rho*u*L/Gamma)-1)*(phi_B-phi_A)+phi_A
        
    
    
    return phi

def erreur_L1(sol_num, sol_anal,dx):
    return np.sum(np.abs(sol_num - sol_anal)*dx)/len(sol_num)

def erreur_L2(sol_num, sol_anal,dx):
    return np.sqrt(np.sum(dx*(sol_num - sol_anal)**2) / len(sol_num))

def erreur_Linf(sol_num, sol_anal):
    return np.max(np.abs(sol_num - sol_anal))

def OrdreConvergeance(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    slope = dy / dx
    return slope

def ConvDiff1DUpwind(x,prm,n):
    
    #Paramètres
    L = prm.L
    dx = L/n
    phi_A = prm.phi_A
    phi_B = prm.phi_B
    rho=prm.rho
    u=prm.u
    Gamma=prm.Gamma
    F=rho*u
    D=Gamma/dx

    D_W=D
    F_W=F
    D_E=D
    F_E=F
    D_A=D
    F_A=F
    
    #Equations aux noeuds internes
    a_w_int= D + F
    a_e_int= D
    S_u_int= 0
    a_p_int= a_w_int + a_e_int
    
    #Equations au noeud à la frontière gauche (A)
    a_w_A=0
    a_e_A=D
    S_p_A= -(2*D+F)
    S_u_A= (2*D+F)*phi_A
    a_p_A=  a_w_A + a_e_A + (F_E -F_W)-S_p_A
    
    #Equations au noeud à la frontière droite (B)
    a_w_B= D+F
    a_e_B=0
    S_p_B= -2*D
    S_u_B=2*D*phi_B
    a_p_B= a_w_B +a_e_B + (F_E -F_W)-S_p_B
    
    # Créer la matrice de phi
    phi = np.zeros(n)

    # Conditions limites et matrice b
    b = np.zeros(n)
    b[0:n] = S_u_int
    b[0] = S_u_A
    b[-1] = S_u_B
   

    # Matrice A
    A = np.zeros((n,n))
    A[0,0] = a_p_A
    A[0,1] = -a_e_A
    A[-1,-1] = a_p_B
    A[-1,-2] = -a_w_B
    for i in range(1, n-1):
        A[i,i-1] = -a_w_int
        A[i,i] = a_p_int
        A[i,i+1] = -a_e_int

    # Résolution du système matriciel Aphi=b
    phi = np.linalg.solve(A,b)
    return phi





#%% Problème 5.1
#Case 1
# Conditions initiales
dx1=prm1.L/prm1.N
dx100 = prm1.L/100

# Graphiques
x1 = np.linspace(prm1.x_0 + dx1 / 2, prm1.x_f - dx1 / 2, prm1.N)
phi1 = ConvDiff1DCentré(x1, prm1, prm1.N)
x_anal = np.linspace(prm1.x_0 + dx100 / 2, prm1.x_f - dx100 / 2, 100)
T_anal = analytique(x_anal, prm1, 100)
plt.figure(1)
plt.plot(x1, phi1, '.', label='Solution numérique Centré')
plt.plot(x_anal, T_anal, label='Solution analytique' )
plt.title(f'Solutions du problème 5.1 à n={prm1.N} et u={prm1.u} m/s (Case 1)')
plt.xlabel('Position (m)')
plt.ylabel('Phi')
plt.xlim(0, prm1.L)
plt.legend()


# Erreur
n_values = np.arange(5, 1001, 100)  
h_values = prm1.L / n_values
Erreur_L1 = []  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx1 = prm1.L / n
    x_e = np.linspace(prm1.L / n / 2, prm1.L - prm1.L / n / 2, n)
    
    # Solution numérique et analytique
    sol_num = ConvDiff1DCentré(x1, prm1, n)
    sol_anal = analytique(x_e, prm1, n)
    
    # Calcul des erreurs
    erreur_L1_value = erreur_L1(sol_num, sol_anal,dx1)
    erreur_L2_value = erreur_L2(sol_num, sol_anal,dx1)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L1.append(erreur_L1_value)
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(2)
plt.loglog(h_values, Erreur_L1,'b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'g', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'r', label="Erreur Linfini")
plt.loglog(h_values, Erreur_L1,'.','b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'.','g', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'.','r', label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title(f"Convergence de l'erreur du problème 5.1 à n={prm1.N} et u={prm1.u} m/s (Case 1)")
plt.show()

#Ordre de convergeance
A=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 5.1 (Case 1) est de :"+ A)


#Case 2
dx2=prm2.L/prm2.N
dx200 = prm2.L/100
# Graphiques
x2 = np.linspace(prm1.x_0 + dx2 / 2, prm1.x_f - dx2 / 2, prm2.N)
phi2 = ConvDiff1DCentré(x2, prm2, prm2.N)
x_anal = np.linspace(prm1.x_0 + dx200 / 2, prm1.x_f - dx200 / 2, 100)
T_anal2 = analytique(x_anal, prm2, 100)
plt.figure(1)
plt.plot(x2, phi2, marker = '.', label='Solution numérique Centré')
plt.plot(x_anal, T_anal2, label='Solution analytique' )
plt.title(f'Solutions du problème 5.1 à n={prm2.N} et u={prm2.u} m/s (Case 2)')
plt.xlabel('Position (m)')
plt.ylabel('Phi')
plt.xlim(0, prm1.L)
plt.legend()


# Erreur


n_values = np.arange(5, 1001, 100)  
h_values = prm1.L / n_values
Erreur_L1 = []  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx2 = prm2.L / n
    x_e = np.linspace(prm1.L / n / 2, prm1.L - prm1.L / n / 2, n)
    
    # Solution numérique et analytique
    sol_num = ConvDiff1DCentré(x2, prm2, n)
    sol_anal = analytique(x_e, prm2, n)
    
    # Calcul des erreurs
    erreur_L1_value = erreur_L1(sol_num, sol_anal,dx1)
    erreur_L2_value = erreur_L2(sol_num, sol_anal,dx1)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L1.append(erreur_L1_value)
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(2)
plt.loglog(h_values, Erreur_L1,'.','b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'.','g', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'.', 'r',label="Erreur Linfini")
plt.loglog(h_values, Erreur_L1,'b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'g', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'r', label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title(f"Convergence de l'erreur du problème 5.1 à n={prm2.N} et u={prm2.u} m/s (Case 2)")
plt.show()

#Ordre de convergeance
B=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 5.1 (Case 2) est de :"+ B)


#Case 3
dx3=prm3.L/prm3.N

# Graphiques
x3 = np.linspace(prm3.x_0 + dx3 / 2, prm3.x_f - dx3 / 2, prm3.N)
phi3 = ConvDiff1DCentré(x3, prm3, prm3.N)
x_anal3 = np.linspace(prm3.x_0 + dx200 / 2, prm3.x_f - dx200 / 2, 100)
T_anal3 = analytique(x_anal3, prm3, 100)
plt.figure(1)
plt.plot(x3, phi3, marker = '.', label='Solution numérique Centré')
plt.plot(x_anal3, T_anal3, label='Solution analytique' )
plt.title(f'Solutions du problème 5.1 à n={prm3.N} et u={prm3.u} m/s (Case 3)')
plt.xlabel('Position (m)')
plt.ylabel('Phi')
plt.xlim(0, prm1.L)
plt.legend()

# Erreur
n_values = np.arange(5, 1001, 100)  
h_values = prm3.L / n_values
Erreur_L1 = []  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx3 = prm3.L / n
    x_e = np.linspace(prm3.L / n / 2, prm3.L - prm3.L / n / 2, n)
    
    # Solution numérique et analytique
    sol_num = ConvDiff1DCentré(x3, prm3, n)
    sol_anal = analytique(x_e, prm3, n)
    
    # Calcul des erreurs
    erreur_L1_value = erreur_L1(sol_num, sol_anal,dx3)
    erreur_L2_value = erreur_L2(sol_num, sol_anal,dx3)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L1.append(erreur_L1_value)
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(2)
plt.loglog(h_values, Erreur_L1,'.','g', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'.','b', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'.','r', label="Erreur Linfini")
plt.loglog(h_values, Erreur_L1,'g',label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'b',label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'r',label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title(f"Convergence de l'erreur du problème 5.1 à n={prm2.N} et u={prm2.u} m/s (Case 3)")
plt.show()

#Ordre de convergeance
C=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 5.1 (Case 3) est de :"+ C)

#%% Problème 5.2
dx2=prm2.L/prm2.N

#Case 1
# Conditions initiales
dx1=prm1.L/prm1.N
dx100 =prm1.L/100

# Graphiques
x1 = np.linspace(prm1.x_0 + dx1 / 2, prm1.x_f - dx1 / 2, prm1.N)
phi1 = ConvDiff1DUpwind(x1, prm1, prm1.N)
x_anal = np.linspace(prm1.x_0 + dx100 / 2, prm1.x_f - dx100 / 2, 100)
T_anal = analytique(x_anal, prm1, 100)
plt.figure(1)
plt.plot(x1, phi1, '.', label='Solution numérique Upwind')
plt.plot(x_anal, T_anal, label='Solution analytique n points' )
plt.title(f'Solutions du problème 5.2 à n={prm1.N} et u={prm1.u} m/s (Case 1)')
plt.xlabel('Position (m)')
plt.ylabel('Phi')
plt.xlim(0, prm1.L)
plt.legend()


# Erreur
n_values = np.arange(5, 1001, 100)  
h_values = prm1.L / n_values
Erreur_L1 = []  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx1 = prm1.L / n
    x_e = np.linspace(prm1.L / n / 2, prm1.L - prm1.L / n / 2, n)
    
    # Solution numérique et analytique
    sol_num = ConvDiff1DUpwind(x1, prm1, n)
    sol_anal = analytique(x_e, prm1, n)
    
    # Calcul des erreurs
    erreur_L1_value = erreur_L1(sol_num, sol_anal,dx1)
    erreur_L2_value = erreur_L2(sol_num, sol_anal,dx1)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L1.append(erreur_L1_value)
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(2)
plt.loglog(h_values, Erreur_L1,'.','b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'.','g', label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'.','r',label="Erreur Linfini")
plt.loglog(h_values, Erreur_L1,'b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'g',label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'r', label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title(f"Convergence de l'erreur du problème 5.1 à n={prm1.N} et u={prm1.u} m/s (Case 1)")
plt.show()

#Ordre de convergeance
A=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 5.2 (Case 1) est de :"+ A)


#Case 2 
dx200 = prm2.L/100

x2 = np.linspace(prm2.x_0 + dx2 / 2, prm2.x_f - dx2 / 2, prm2.N)
T2 = ConvDiff1DUpwind(x2, prm2, prm2.N)
x_anal2 = np.linspace(prm2.x_0 + dx200 / 2, prm2.x_f - dx200 / 2, 100)
T_anal2 = analytique(x_anal2, prm2, 100)
plt.figure(3)
plt.plot(x2, T2, '.', label='Solution numérique Upwind')
plt.plot(x_anal2, T_anal2, label='Solution analytique' )
plt.title(f'Solutions du problème 5.2 à n={prm2.N} et u={prm2.u} m/s (Case 2)')
plt.xlabel('Position (m)')
plt.ylabel('Phi')
plt.xlim(0, prm2.L)
plt.legend()

n_values = np.arange(5, 1001, 100)  
h_values = prm2.L / n_values
Erreur_L1 = []  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx2 = prm2.L / n
    x_e = np.linspace(prm2.L/n/2, prm2.L - prm2.L/n/2, n)
    
    # Solution numérique et analytique
    sol_num = ConvDiff1DUpwind(x_e, prm2, n)
    sol_anal = analytique(x_e, prm2, n)
    
    # Calcul des erreurs
    erreur_L1_value = erreur_L1(sol_num, sol_anal,dx2)
    erreur_L2_value = erreur_L2(sol_num, sol_anal,dx2)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L1.append(erreur_L1_value)
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(4)
plt.loglog(h_values, Erreur_L1,'.','b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'.','g',label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'.','r', label="Erreur Linfini")
plt.loglog(h_values, Erreur_L1,'b', label="Erreur L1")
plt.loglog(h_values, Erreur_L2,'g',label="Erreur L2")
plt.loglog(h_values, Erreur_Linf,'r', label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title(f"Convergence de l'erreur du problème 5.2 à n={prm2.N} et u={prm2.u} m/s (Case 2)")
plt.show()

B=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 5.2 (Case 2) est de :"+ B)


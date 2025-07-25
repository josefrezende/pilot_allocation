import sys
import random
random.seed(None)
import math
import numpy as np
np.random.seed(None)
import scipy.stats as ss
import cvxpy as cp
from collections import OrderedDict
from itertools import islice
import matplotlib.pyplot as plt
import os.path

# valores de M (APs), K (UEs) e P (Pilotos)
M = int(sys.argv[1])
K = int(sys.argv[2])
P = int(sys.argv[3])

# numero de instancias
RODADAS = int(sys.argv[4])

def init_vars():

    global D, L, BW, rho_p, rho_u, tau
    global p_min, p_max, p
    global ruk_maxkcut, ruk_random, ruk_greedy, ruk_ibasic, ruk_basic, ruk_wgf, ruk_wgfsb

    # lado da area quadrada em metros
    D = 1000

    # frequencia (MHz), alturas dos APs e UEs (m)
    freq=1.9e3
    hAP=15
    hUE=1.65
    # constante usada no calculo do PL (dB)
    L = 46.3 + 33.9*math.log10(freq)-13.82*math.log10(hAP)-(1.1*math.log10(freq)-0.7)*hUE + (1.56*math.log10(freq)-0.8)

    # transmit power em W
    rho_chapeu_u = 0.1
    rho_chapeu_p = 0.1

    # bandwidth (MHz)
    BW = 20e6
    NoiseFigure_dB = 9
    K_Boltzmann = 1.380649e-23
    T0 = 290
    Noise_Power = K_Boltzmann * T0 * BW * 10**(NoiseFigure_dB/10)

    # SNR normalizada
    rho_p = rho_chapeu_p/Noise_Power
    rho_u = rho_chapeu_u/Noise_Power

    # tempo de treinamento
    tau = P

    # vetores para armazenamento das taxas (taxas iguais para todos os usuarios em cada rodada)
    ruk_maxkcut=np.zeros(RODADAS)
    ruk_random=np.zeros(RODADAS)
    ruk_greedy=np.zeros(RODADAS)
    ruk_ibasic=np.zeros(RODADAS)
    ruk_basic=np.zeros(RODADAS)
    ruk_wgf=np.zeros(RODADAS)
    ruk_wgfsb=np.zeros(RODADAS)
    
    # valores min e max de eta
    p_min = 0.0 * np.ones(K)  # minimum power at the UE k
    p_max = 1.0 * np.ones(K)  # maximum power at the UE k

    # eta de cada usuario
    p = cp.Variable(shape=(K,), pos=True)
    
def calc_beta():

    global beta, beta_k
    beta={}
    beta_k={}

    # geracao de amostras de uma distribuicao normal que modelam o shadow fading
    mu, sigma = 0, 1
    a = np.random.normal(mu,sigma,M)
    b = np.random.normal(mu,sigma,K)

    # distancias de referencia para o calculo do PL (m)
    d0=10
    d1=50
    # desvio padrao do shadow fading
    sigma_dB = 8

    # calculo dos betas
    for m in range(M):

        dnormal = np.random.normal(mu,sigma,K)

        posAP_X = random.randrange(0,D)
        posAP_Y = random.randrange(0,D)
        for k in range(K):
            # posicao dos UEs
            x = random.randrange(0,D)
            y = random.randrange(0,D)

            # wrapped distance in x and y
            distance = math.sqrt(min(abs(x-posAP_X),D-abs(x-posAP_X))**2 + min(abs(y-posAP_Y),D-abs(y-posAP_Y))**2)

            # calculate z
            z=dnormal[k]
                                    
            # calculate PL
            if (distance <= d0):
                PL = -L - 15*math.log10(d1) - 20*math.log10(d0)
            else:
                if (distance <= d1):
                    PL = -L - 15*math.log10(d1) - 20*math.log10(distance/1000)
                else:
                    PL = -L - 35*math.log10(distance/1000)

            beta[(m,k)] = 10**((PL+sigma_dB*z)/10)

    # soma de beta de todos os APs
    for k in range(K):
        beta_k[k]=0
        for m in range(M):
            beta_k[k] += beta[(m,k)]
            
# funcao que calcula o valor dos pesos entre os vertices (i,j) quando ocorre fusao no maxkcut
def funcao (i,j):
    newbeta=0
    a = i.split("-")
    b = len(j.split("-"))
    for k in a:
        newbeta += beta_k[int(k)]
    newbeta = b*newbeta
    return newbeta

# algoritmo max-k-cut
def maxkcut ():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}
    
    # cria o dicionario (matriz de adjacencia)
    matriz={}
    # inicializa matriz com valores de pesos
    for i in range(K):
        l={}
        for j in range(K):
            if i == j:
                l[str(j)] = -1
            else:
                l[str(j)] = beta_k[i]
        matriz[str(i)] = l

    # ajuste do valor de numPilotos para a geracao das estruturas
    if (P > K): numPilotos=K
    else: numPilotos=P
        
    while len(matriz) > numPilotos:
        # calcula menor soma dos pesos entre dois vertices 
        menor=999999999999999
        for i in matriz:
            for j in matriz[i]:
                if i == j:
                    continue
                if matriz[i][j] + matriz[j][i] < menor:
                    menorI, menorJ = i, j
                    menor = matriz[i][j] + matriz[j][i]

        # remove do dicionario vertice i e j com menor soma dos pesos menorI e menorJ
        del matriz[menorI];
        del matriz[menorJ];

        # acrescenta no dicionario vertice i-j
        novachave=str(menorI)+"-"+str(menorJ)
        matriz[novachave]={}

        for i in matriz:
            # remove do dicionario os pesos dos vertices i e j para todos os demais vertices 
            try:
                del matriz[i][menorI]
            except KeyError:
                pass
                
            try:
                del matriz[i][menorJ]
            except KeyError:
                pass
       
            # acrescenta no dicionario os pesos do novo vertice i-j para todos os demais vertices
            if i != novachave:
                matriz[i][novachave]=funcao(i,novachave)
                matriz[novachave][i]=funcao(novachave,i)
            else:
                matriz[i][novachave]=-1

    keys = list(matriz.keys())

    for i in range(numPilotos):
        usuarios_por_piloto[i] = [int (x) for x in keys[i].split("-")]
        for u in keys[i].split("-"):
            piloto[int(u)] = i

# algoritmo aleatorio
def randomic():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    for i in range(K):
        piloto[i] = random.randrange(0,P)
        if piloto[i] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[i]] = []
        usuarios_por_piloto[piloto[i]].append(i)

# algoritmo GREEDY
def greedy():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    # aloca pilotos randomicamente
    randomic()

    rate=np.zeros(K)

    # nr de iteracoes
    N=1000000
    
    for n in range(N):

        calc_gamma()

        eta = K * [1]
        calc_coeficientes(0)

        # calculas as taxas
        rate = calc_taxas(eta,0)

        # indice do usuario com a menor taxa
        user_minrate = np.argmin(rate)

        # encontra o piloto adequado
        bk_atual = 1000
        p_atual=-1
        for pil in range(P):
            bk=0
            # tem algum usuario no piloto pil?
            if (pil in usuarios_por_piloto):
                p1 = usuarios_por_piloto[pil].copy()
                if user_minrate in p1:
                    p1.remove(user_minrate)
                for m in range(M):
                    for k_linha in p1: 
                        bk += beta[(m,k_linha)]
            if bk < bk_atual:
                bk_atual = bk
                p_atual = pil

        # se nao houve troca de piloto, interrompe iteracao
        if (p_atual == piloto[user_minrate]): break
            
        # remove usuario do piloto em que ele se encontra    
        usuarios_por_piloto[piloto[user_minrate]].remove(user_minrate)
        if p_atual not in usuarios_por_piloto:
            usuarios_por_piloto[p_atual] = []
        # troca o usuario de piloto
        usuarios_por_piloto[p_atual].append(user_minrate)
        piloto[user_minrate] = p_atual

# algoritmo improved BASIC
def ibasic ():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    counter = np.zeros(P)

    # valor do artigo (delta = 5)
    zeta = max(5, math.ceil(K/P))

    # sort de beta_k
    sorted_bk = dict(sorted(beta_k.items(), key=lambda item: item[1]))
    allocated_pilots=0
    while (len(sorted_bk) != 0):
        first=list(sorted_bk.keys())[0]
        if allocated_pilots < P:
            piloto[first] = allocated_pilots
            counter[allocated_pilots] = 1
            allocated_pilots += 1
            if piloto[first] not in usuarios_por_piloto:
                usuarios_por_piloto[piloto[first]] = []
            usuarios_por_piloto[piloto[first]].append(first)
            del sorted_bk[first]
        else:
            # calcula APmaster
            master=-1
            beta_atual=0
            for m in range(M):
                if (beta[(m,first)] > beta_atual):
                    beta_atual = beta[(m,first)]
                    master=m

            # encontra o melhor piloto com menos de zeta usuarios        
            not_allocated=1
            already_tested = []
            while (not_allocated):
                bk = 0
                bk_atual = 1000
                for pil in range(P):
                    if (pil not in already_tested):
                        p1 = usuarios_por_piloto[pil].copy()
                        bk=0
                        for k in p1:
                            bk +=beta[(master,k)]
                        if bk < bk_atual:
                            bk_atual = bk
                            p_atual = pil
                        
                if (counter[p_atual] < zeta):
                    piloto[first] = p_atual
                    ++counter[p_atual]
                    not_allocated=0
                    if piloto[first] not in usuarios_por_piloto:
                        usuarios_por_piloto[piloto[first]] = []
                    usuarios_por_piloto[piloto[first]].append(first)
                    del sorted_bk[first]
                else:
                    already_tested.append(p_atual)

# algoritmo BASIC            
def basic ():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    for i in (i for i in range(P) if i < K):
        piloto[i] = i
        if piloto[i] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[i]] = []
        usuarios_por_piloto[piloto[i]].append(i)

    for user in range(P,K):

        # calcula bestAP
        bestAP=-1
        beta_atual=0
        for m in range(M):
            if (beta[(m,user)] > beta_atual):
                beta_atual = beta[(m,user)]
                bestAP=m

        p_atual = -1
        bk_atual = 1000
        for pil in range(P):
            if pil in usuarios_por_piloto:
                p1 = usuarios_por_piloto[pil].copy()
                bk=0
                for k in p1:
                    bk +=beta[(bestAP,k)]
                if bk < bk_atual:
                    bk_atual = bk
                    p_atual = pil

        piloto[user] = p_atual
        if piloto[user] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[user]] = []
        usuarios_por_piloto[piloto[user]].append(user)

def wgf():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    # calcula os N melhores APs 
    N=10
    APs = {}
    for k in range(K):
        beta_per_AP = {m: beta[(m,k)] for m in range(M)}
        bap = dict(sorted(beta_per_AP.items(), key=lambda item: item[1], reverse=True))
        APs[k] = list(islice(bap.keys(), N))

    # calcula os pesos
    w = {}
    for k in range(K):
        for k_linha in range(K):
            w[(k,k_linha)] = (sum(beta[(m,k_linha)] for m in APs[k_linha])/sum(beta[(m,k)] for m in APs[k]))**2+(sum(beta[(m,k)] for m in APs[k])/sum(beta[(m,k_linha)]  for m in APs[k_linha]))**2
            
    users=[]
    for k in range(K):
        users.append(k)
    
    for pil in range(P):
        userid = random.randrange(0,len(users))
        user = users[userid]
        piloto[user] = pil
        users.remove(user)
        if piloto[user] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[user]] = []
        usuarios_por_piloto[piloto[user]].append(user)
        
    while (len(users) > 0):
        userid = random.randrange(0,len(users))
        user = users[userid]
        S_atual = 1000
        p_atual = -1
        for pil in range(P):
            S=0
            for v in usuarios_por_piloto[pil]:
                S += w[(user,v)]
            if (S < S_atual):
                S_atual = S
                p_atual = pil
        piloto[user] = p_atual        
        if piloto[user] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[user]] = []
        usuarios_por_piloto[piloto[user]].append(user)
        users.remove(user)
    
def wgfsb():

    global usuarios_por_piloto, piloto
    usuarios_por_piloto={}
    piloto={}

    # calcula os pesos
    w = {}
    for k in range(K):
        for k_linha in range(K):
            w[(k,k_linha)] = beta_k[k_linha] + beta_k[k]
            
    users=[]
    for k in range(K):
        users.append(k)
    
    for pil in range(P):
        userid = random.randrange(0,len(users))
        user = users[userid]
        piloto[user] = pil
        users.remove(user)
        if piloto[user] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[user]] = []
        usuarios_por_piloto[piloto[user]].append(user)
        
    while (len(users) > 0):
        userid = random.randrange(0,len(users))
        user = users[userid]
        S_atual = 1000
        p_atual = -1
        for pil in range(P):
            S=0
            for v in usuarios_por_piloto[pil]:
                S += w[(user,v)]
            if (S < S_atual):
                S_atual = S
                p_atual = pil
        piloto[user] = p_atual        
        if piloto[user] not in usuarios_por_piloto:
            usuarios_por_piloto[piloto[user]] = []
        usuarios_por_piloto[piloto[user]].append(user)
        users.remove(user)
    
def calc_gamma():

    global gamma
    gamma={}

    for m in range(M):
        for k in range(K):
            bk=0
            p1 = usuarios_por_piloto[piloto[k]].copy()
            for k_linha in p1:
                bk += beta[(m,k_linha)]
            gamma[(m,k)] = tau * rho_p * beta[(m,k)]**2 / (tau * rho_p * bk + 1)
            
def calc_taxas(p,start):

    rate = np.zeros(K)
    
    # sinr de uplink por usuario k 
    for k in range(K):
        E, F, R = np.zeros(K), np.zeros(K), np.zeros(K)
        E = sum(e[(k,kp)]*p[kp] for kp in range(K) if k != kp)
        F = sum(f[(k,kp)]*p[kp] for kp in range(K))
        R = r[k]
        rate[k] = p[k]/(E + F + R)

    return(rate)

def mean_confidence_interval(data, confidence=0.95):

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), ss.sem(a)
    h = se * ss.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def escreve_arqs(rodada):

    efilename = "e_"+str(M)+"_"+str(K)+"_"+str(P)+"-"+str(rodada)
    ffilename = "f_"+str(M)+"_"+str(K)+"_"+str(P)+"-"+str(rodada)
    rfilename = "r_"+str(M)+"_"+str(K)+"_"+str(P)+"-"+str(rodada)
    efile = open(efilename, 'w')
    ffile = open(ffilename, 'w')
    rfile = open(rfilename, 'w')
    for k in range(K):
        rfile.write(str(r[k]))
        rfile.write('\n')
        for k_linha in range(K):
            if k_linha < (K-1):
                efile.write(str(e[(k,k_linha)]))
                efile.write('\t')
                ffile.write(str(f[(k,k_linha)]))
                ffile.write('\t')
            else:
                efile.write(str(e[(k,k_linha)]))
                efile.write('\n')
                ffile.write(str(f[(k,k_linha)]))
                ffile.write('\n')
    ffile.close
    efile.close
    rfile.close
    
def calc_coeficientes (rodada):

    global e, f, r
    e, f, r = {}, {}, {}

    for k in range(K):
        samepk = usuarios_por_piloto[piloto[k]].copy()
        samepk.remove(k) 
        for k_linha in range(K):
            n_e, d_e_f, n_f = 0, 0, 0
            for m in range(M):
                n_e += gamma[(m,k)]*beta[(m,k_linha)]/beta[(m,k)]
                d_e_f += gamma[(m,k)]
                n_f += gamma[(m,k)]*beta[(m,k_linha)]
            n_e = n_e**2
            d_r = d_e_f * rho_u
            d_e_f = d_e_f**2
            if k_linha in samepk:
                e[(k,k_linha)] = n_e/d_e_f
            else:
                e[(k,k_linha)] = 0.0
            f[(k,k_linha)] = n_f/d_e_f
        r[k] = 1.0/d_r

    # escreve_arqs(rodada)
     
def controle_de_potencia(rodada):

    # calcula os coeficientes da formula de sinr
    calc_coeficientes(rodada)

    # calcula denominador da formula
    x=[]
    for k in range(K):
        x.append(cp.sum(cp.hstack(e[(k,kp)]*p[kp] for kp in range(K) if k != kp))+cp.sum(cp.hstack(f[(k,kp)]*p[kp] for kp in range(K)))+r[k])
    denominador = cp.hstack(x)

    # coloca sinr_target como parametro
    sinr_target = cp.Parameter(nonneg=True)

    # define o problema
    objective = cp.Minimize(0)
    constraints = [p >= p_min, p <= p_max, p >= sinr_target*denominador]
    problema = cp.Problem(objective, constraints)

    init_target_max = 0.5
    sinr_target.value = init_target_max
    problema.solve(solver=cp.GUROBI, reoptimize=True)

    while (problema.status == "optimal"):
        init_target_max = 2 * init_target_max
        sinr_target.value = init_target_max
        problema.solve(solver=cp.GUROBI, reoptimize=True)

    # espaco de busca do sinr alvo
    sinr_target_min = 0
    sinr_target_max = init_target_max

    epsilon = 1e-5
    while ((sinr_target_max - sinr_target_min) > epsilon) or (problema.status == "infeasible"):

        sinr_target.value = (sinr_target_max+sinr_target_min)/2

        sol = problema.solve(solver=cp.GUROBI, reoptimize=True)

        if problema.status == "infeasible":
            sinr_target_max = sinr_target.value
        else:
            sinr_target_min = sinr_target.value

    eta_l = p.value.copy()

    return sinr_target.value, eta_l

# gera histograma
def stats(ruk,algo):

    cdffilename = "cdf_"+str(M)+"_"+str(K)+"_"+str(P)+"_"+str(algo)+".dat"
    if os.path.isfile(cdffilename):
        cdffile = open(cdffilename, 'a')
    else:
        cdffile = open(cdffilename, 'w')

    for r in range(RODADAS):
        cdffile.write(str(ruk[r]))
        cdffile.write('\n')
        
    # sorted_ruk = np.sort(ruk)
    # cdf = np.arange(RODADAS)+1
    # for r in range(RODADAS):
    #     line = [str(sorted_ruk[r]), str(cdf[r]/RODADAS)]
    #     cdffile.write("\t".join(line))
    #     cdffile.write("\n")
    cdffile.close
    
# main
# inicializa as variaveis
init_vars()

# algo = OrderedDict({"maxkcut": [maxkcut, ruk_maxkcut], "random": [randomic, ruk_random], "greedy": [greedy, ruk_greedy], "basic": [basic, ruk_basic], "ibasic": [ibasic, ruk_ibasic], "wgf": [wgf, ruk_wgf], "wgfsb": [wgfsb, ruk_wgfsb]})

algo = OrderedDict({"maxkcut": [maxkcut, ruk_maxkcut], "basic": [basic, ruk_basic]})

for rodada in range(RODADAS):

    # reposiciona APs e UEs 
    calc_beta()

    for a in algo:
        algo[a][0]()
        calc_gamma()
        sinropt, eta = controle_de_potencia(rodada)
        taxas = calc_taxas(eta,rodada)

        algo[a][1][rodada] = taxas[0]

for a in algo:
    stats(algo[a][1],a)

resultsfilename = "res_"+str(M)+"_"+str(K)+"_"+str(P)+".dat"
if os.path.isfile(resultsfilename):
    resf = open(resultsfilename, 'a')
else:
    resf = open(resultsfilename, 'w')

for rodada in range(RODADAS):
    line = [str(M) , str(K), str(P)]
    resf.write("\t".join(line))
    resf.write("\t")
    for a in algo:
        resf.write(str(algo[a][1][rodada]))
        resf.write("\t")
    resf.write("\n")


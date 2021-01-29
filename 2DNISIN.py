import kwant
import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import winsound
import os
from datetime import datetime
import scipy.sparse.linalg as sla
import matplotlib

import originpro as op 


sig_0 = np.identity(4)
s_0 = np.identity(2)
s_z = np.array([[1., 0.], [0., -1.]])
s_x = np.array([[0., 1.], [1., 0.]])
s_y = np.array([[0., -1j], [1j, 0.]])

tau_z = ta.array(np.kron(s_z, s_0))
tau_x = ta.array(np.kron(s_x, s_0))
tau_y = ta.array(np.kron(s_y, s_0))
sig_z = ta.array(np.kron(s_0, s_z))
sig_x = ta.array(np.kron(s_0, s_x))
tau_zsig_x = ta.array(np.kron(s_z, s_x))
tau_xsig_x = ta.array(np.kron(s_x, s_x))
tau_zsig_y = ta.array(np.kron(s_z, s_y))
tau_ysig_y= ta.array(np.kron(s_y, s_y))



def makeNISIN2D(params): 

# List all the params
    W=params["W"]
    L=params["L"]
    pot = params["pot"]
    ind = params["ind"]
    y0 = params["y0"]
    mu=params["mu"]
    B=params["B"]
    Delta=params["Delta"]
    alpha=params["alpha"]
    t=params["t"]
    barrier = params["barrier"]
    Smu = params["Smu"]
    gam = params["gam"]
    gam2 = params["gam2"]
    ## hard coded disorder parameters ##
    pot2 = params["pot2"]
    pot3 = params["pot3"]
    pot4 = params["pot4"]
    pot5 = params["pot5"]
    pot6 = params["pot6"]
    xs = params["xs"]
    ys = params["ys"]
    x3  =params["x3"]
    y3  =params["y3"]
    x4  =params["x4"]
    y4  =params["y4"]
    x5  =params["x5"]
    y5  =params["y5"]
    x6  =params["x6"]
    y6  =params["y6"]
    barl =params["barl"]
  
    #Disorder potentials, 
    def gaussian(x, y, xs, ys):
        chi=15/2*np.sqrt(2*np.log(2))
        chiy=5/2*np.sqrt(2*np.log(2)) #width should be uniformly covered
        return np.exp(-((x-xs)**2/(2*chi**2))-((y-ys)**2/(2*chiy**2)))  
    
    ##### Tip profile #####

    def lorentzianxy(x, y, ind,y0):
        return gam ** 2 /( gam ** 2 + (x - ind) ** 2 + (y - y0) ** 2)
    
    ##### Tip potential #####
          
    def potential(site, pot,ind,y0):
        (x, y) = site.pos
        
        return pot * lorentzianxy(x,y,ind,y0) + pot2*gaussian(x,y,xs,ys) +\
    pot3*gaussian(x, y, x3, y3) +pot4*gaussian(x, y, x4, y4) +pot5*gaussian(x,y,x5,y5)
       
#    This plots the disorder potentials
#    def potval(pot,ind,y0,pot2,xs,ys,L,W):
#        potv = []
#          
#        for y in range(0,W): 
#            xpotval = []
#            for x in range(0,L):
#                 reqpot = pot * lorentzianxy(x,y,ind,y0) + pot2*gaussian(x,y,xs,ys) +\
#   pot3*gaussian(x, y, x3, y3) +pot4*gaussian(x, y, x4, y4) + pot5*gaussian(x, y, x5, y5) +pot6*gaussian(x,y,x6,y6)
#                 
#                 xpotval.append(reqpot)
#            potv.append(xpotval)
#        plt.rcParams["figure.figsize"] = (12,1.9)   
#        con=plt.contourf(potv, 300, cmap="Greys" )
#        plt.title("Potential intensity, $V_{tip}$="+str(params["pot"])+" at ("+str(params["ind"])+","+str(params["y0"])+"), $\gamma$="+str(params["gam"])+", $V_{disorder}$ ="+str(params["pot2"])+" at ("+ str(params["xs"])+","+str(params["ys"]) +"), "+str(params["pot3"])+" at ("+ str(params["x3"])+","+str(params["y3"]) +" )")
#        #plt.figure().colorbar(con)
#    potval(pot,ind,y0,pot2,xs,ys,L,W)  
#    plt.savefig("./xtip_contours/Final_Data/potentialCont/2disorderpotential="+str(params["pot"])+"("+str(params["ind"])+","+str(params["y0"])+"), gamma"+str(params["gam"])+".png")
   
    ####### system setup #######
    
    def onsiteSc(site, pot,ind,y0):
       return (4 * t - Smu) * tau_z + B * sig_x + Delta * tau_x -\
   potential(site, pot,ind,y0)*tau_z
           
    def onsiteNormal(site, pot,ind,y0):
        return (4 * t - mu) * tau_z
    
    def onsiteBarrier(site, pot,ind,y0,):
        L=params["L"]
        return (4 * t - mu + barrier - potential(site, pot,ind,y0))*tau_z # 
   
    def hop_x(site, pot, ind,y0):        
        return -t * tau_z + 0.5j * alpha * tau_zsig_y 
    
    def hop_y(site, pot, ind,y0):    
           return -t * tau_z - .5j * alpha * tau_zsig_x

    
    # On each site, electron and hole orbitals.
    lat = kwant.lattice.square(norbs=4) 
    syst = kwant.Builder()
    
    # S
    syst[(lat(i, j) for i in range(1,L-1) for j in range(W))] = onsiteSc
    
    barrierLen = params["barl"];
    # I's
    syst[(lat(i, j) for i in range(barrierLen) for j in range(W))] = onsiteBarrier
    syst[(lat(i, j) for i in range(L-barrierLen, L)for j in range(W))] = onsiteBarrier
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)]= hop_x
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)]= hop_y
    
    # N's
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,0)),
                         conservation_law=-tau_z, particle_hole=tau_ysig_y)
    
    lead[(lat(0, j) for j in range(W))] = onsiteNormal
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)]= hop_x
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)]= hop_y
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst




############################################################################
################################1D ZBP Plot#################################
############################################################################


def ZBP(params, energies):
    E_data = []
    L=params["L"]
    xtip = np.linspace(-15,L+15,L+31)
    for e in energies:
        
        syst=makeNISIN2D(params=params)
        plt.rcParams["figure.figsize"]= (8,5)            
        syst=syst.finalized()
        smatrix = kwant.smatrix(syst, e, params=params)
        E_data.append(smatrix.transmission((1, 0), (1, 1)) +smatrix.transmission((0, 1), (0,0)) ) 
                 
    fig = plt.figure()
    plt.plot(energies, E_data)
    plt.title("L="+str(params["L"]) +", potential ="+str(params["pot"])+" at ("+ str(params["ind"])+","+str(params["y0"]) +")")
    plt.xlabel("Energy")
    plt.ylabel("Conductance$[e^{2}/h]$")
    plt.savefig("./xtip_contours/Final_Data/ZBP/pot="+str(params["pot"])+"Length="+str(params["L"])+"Barrierstrength="+str(params["barrier"])+".png")
    #plt.close()
    E_data=[]



############################################################################
##########################Eigenspectrum Plots###########################
############################################################################
    

#Energy spectrum as Zeeman field is varied
def plotSpectrum(params, energies):
    B_values = np.linspace(0.0, 1, 101)
   
    energies = []
    for B in tqdm(B_values):            
        params["B"] = B      
        syst = makeNISIN2D(params)
        syst= syst.finalized()
        plt.rcParams["figure.figsize"] = (8,5)
        H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
        H = H.tocsc()    
        eigs = sla.eigsh(H, k=8, sigma=0) 
        energies.append(np.sort(eigs[0]))
    plt.plot(B_values,  energies)    
    plt.xlabel("Zeeman field strength")
    plt.ylabel("Energy")
    plt.title("Energy Eigenvalues with Varying Zeeman Field, L="+str(params["L"])+", barrier="+str(params["barrier"])) #$V_{tip}$="+str(params["pot"])+" ("+str(params["ind"])+","+str(params["y0"])+"), L=" +str(params["L"]))
    plt.savefig("./xtip_contours/Final_Data/Zeeman/barrier="+str(params["barrier"])+"L="+str(params["L"])+".png")
  #  plt.close()
    energies = []
        




#Energy spectrum as global chemical potential in the superconductor is varied.
def plotSpectrumChem(params, energies):
    #os.mkdir('./xtip_contours/EnergyEigenvalues/varyingchem/barrier=3_0/zoom/')
    chem = [0.01*i for i in range(-20,60,1)]
    energies = []
    L = params["L"]
    pot = params["pot"]
    for m in tqdm(chem):
        params["Smu"] = m               
        syst = makeNISIN2D(params = params)
        plt.rcParams["figure.figsize"] = (8,5)
        syst= syst.finalized()
        H = syst.hamiltonian_submatrix(sparse=True,  params = params)    
        H = H.tocsc()    
        eigs = sla.eigsh(H, k=8, sigma=0)
        energies.append(np.sort(eigs[0]))
    plt.plot(chem,  energies)    
    plt.xlabel("Superconductor Chemical Potential")
    plt.ylabel("Energy")
    plt.title("Energy Eigenvalues Varying $\mu_{SC}$,  L="+str(params["L"])+", barrier="+str(params["barrier"]))#and disorder potential="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+"), L="+str(params["L"]))
    plt.savefig("./xtip_contours/Final_Data/Chempot/L="+str(params["L"])+"barrier="+str(params["barrier"])+".png")
    #plt.close()
    energies = []
    
    

    
def condchem(params, energies):
      
    cond = []
    chem = [0.001*i for i in range(0,350,1)]
    E_data = []    
    en = 0.00
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for c in chem:        
            params["Smu"]=c
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, en,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                    smatrix.transmission((0, 1), (0, 0)))       
   

    plt.plot(chem, E_data)
    plt.title("$\mu_{sc}$ vs Conductance, Energy="+str(en)+", L="+str(params["L"])+", $V_{tip}$="+str(params["pot"])+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $\gamma$="+str(params["gam"]))
    plt.xlabel("Superconductor chemical potential ($\mu_{sc}$)")
    plt.ylabel("Conductance [$e^2/h$]")
    plt.savefig("./xtip_contours/Final_Data/SmuvsG/Energy="+str(en)+",L="+str(params["L"])+"pot="+str(params["pot"])+"(" + str(params["ind"])+',' + str(params["y0"])+").png")


def condpot(params, energies):
      
    cond = []
    chem = [0.01*i for i in range(0,121,1)]
    E_data = []    
    en = 0.00
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for c in chem:        
            params["pot"]=c
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, en,params=params)
            E_data.append(smatrix.transmission((0, 1), (0, 0))+ smatrix.transmission((1,0), (1, 1)))      
   

    plt.plot(chem, E_data)
    plt.title("$V_{tip}$ vs Conductance, Energy="+str(en)+", L="+str(params["L"])+", tip at"+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $\gamma$="+str(params["gam"]))
    plt.xlabel("$V_{tip}$")
    plt.ylabel("Conductance [$e^2/h$]")
    plt.savefig("./xtip_contours/Final_Data/TipVSCond/Energy="+str(en)+",L="+str(params["L"])+"(" + str(params["ind"])+',' + str(params["y0"])+")gam="+str(params["gam"])+".png")


def condgam(params, energies):
      
    cond = []
    chem = [1*i for i in range(1,100,1)]
    E_data = []    
    en = 0.00
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for c in chem:        
            params["gam"]=c
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, en,params=params)
            E_data.append(smatrix.submatrix((0, 0), (0, 0)).shape[0] -
                    smatrix.transmission((0, 0), (0, 0)) +
                    smatrix.transmission((0, 1), (0, 0)))           # smatrix.transmission((1, 0), (1, 1)) +
                    #smatrix.transmission((0, 1), (0, 0)))        

    plt.plot(chem, E_data)
    plt.title("$\gamma$ vs Conductance, Energy="+str(en)+", L="+str(params["L"])+", tip at"+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $V_{tip}$="+str(params["pot"]))
    plt.xlabel("$\gamma$")
    plt.ylabel("Conductance [$e^2/h$]")
#    plt.savefig("./xtip_contours/Final_Data/GamVsCond/Energy="+str(en)+",L="+str(params["L"])+"(" + str(params["ind"])+',' + str(params["y0"])+")pot="+str(params["pot"])+".png")



def engam(params, energies):
      
    cond = []
    chem = [1*i for i in range(1,100,1)]
    E_data = []
  
    for c in tqdm(chem):        
            params["gam"]=c      
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
            H = H.tocsc()    
            eigs = sla.eigsh(H, k=4, sigma=0) 
            E_data.append(np.sort(eigs[0])[3])
            cond.append(np.transpose(E_data))
            
    plt.plot(chem, E_data)
    plt.title("$\gamma$ vs Energy of $E_{1+}$ mode, L="+str(params["L"])+", tip at"+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $V_{tip}$="+str(params["pot"]))
    plt.xlabel("$\gamma$")
    plt.ylabel("Energy of $E_{0}$ mode")
    plt.savefig("./xtip_contours/Final_Data/gamVsEnergy/E_1+L="+str(params["L"])+"(" + str(params["ind"])+',' + str(params["y0"])+")pot="+str(params["pot"])+".png")

def poten(params, energies):      
    chem = [0.01*i for i in range(-50,301,1)]
    E_data = []    
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for c in tqdm(chem):        
            params["pot"]=c      
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
            H = H.tocsc()    
            eigs = sla.eigsh(H, k=4, sigma=0) 
            E_data.append(np.sort(eigs[0])[2])
    plt.plot(chem, E_data)
    plt.title("$V_{tip}$ vs Energy of $E_{0+}$ mode, L="+str(params["L"])+", tip at"+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $\gamma$="+str(params["gam"]))
    plt.xlabel("$V_{tip}$")
    plt.ylabel("Energy of $E_{0+}$ mode")
    plt.savefig("./xtip_contours/Final_Data/potVsEnergy/2E0bar="+str(params["barrier"])+"gam="+str(params["gam"])+",L="+str(params["L"])+"(" + str(params["ind"])+',' + str(params["y0"])+").png")


    
############################################################################
#############################Conductance Contour Plots######################
############################################################################

def barCond(params, energies):
      
    cond = []
    length = [1*i for i in range(0,10,1)]
    E_data = []    
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for l in tqdm(length):
        for energy in energies:
            params["barl"]=l
            params["pot"]=0
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, energy,params=params)
            E_data.append(smatrix.submatrix((0, 0), (0, 0)).shape[0] -
                    smatrix.transmission((0, 0), (0, 0)) +
                    smatrix.transmission((0, 1), (0, 0)))
           
        cond.append(np.transpose(E_data))
        E_data = []
    cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(length,energies,cond, 200, cmap = 'plasma')
    plt.title("Conductance as barrier length is Varied, system Length="+str(params["L"]))
    plt.xlabel("barrier length")
    plt.ylabel("Bias V")


    
def VaryingZeeman(params, energies):
    cond = []
    chem = [0.001*i for i in range(300,502,2)]     
    E_data = []    
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for m in tqdm(chem):
        for energy in energies:
            params["B"]=m
            params["pot"]=0
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, energy,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                    smatrix.transmission((0, 1), (0, 0)))           
        cond.append(np.transpose(E_data))
        E_data = []
    cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(chem,energies,cond, 300, cmap = 'plasma')
    plt.title("G($E_{Z}$,$V_{bias}$), Length="+str(params["L"])+"bar="+str(params["barrier"]))
    plt.xlabel("Zeeman Field Strength")
    plt.ylabel("Bias Voltage")
    fig.colorbar(con, label="Conductance[$e^2/h$]")
    plt.savefig('./xtip_contours/Final_Data/Zeeman/bar='+str(params["barrier"])+'L='+str(params["L"])+'.png')
    plt.close()  

    
def VaryingGlobalChemPot(params, energies):
    cond = []
    chem = [0.001*i for i in range(-200,601,2)]
    E_data = []
    
    #os.mkdir('./xtip_contours/ChemicalPotential')
    for m in tqdm(chem):
        for energy in energies:
            params["Smu"]=m
            params["pot"]=0
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, energy,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                    smatrix.transmission((0, 1), (0, 0)))
        cond.append(np.transpose(E_data))
        E_data = []
    cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(chem,energies,cond, 200, cmap = 'plasma')
    plt.title("G($\mu_{SC}$,$V_{bias}$), Length="+str(params["L"])+"bar="+str(params["barrier"]))
    plt.xlabel("Superconductor Chemical Potential")     
    plt.ylabel("Bias Voltage")
    fig.colorbar(con, label="Conductance[$e^2/h$]")
    plt.savefig('./xtip_contours/Final_Data/Chempot/L='+str(params["L"])+'varyingchemmB=0_3.png')
    plt.close()
    
        
def VaryingGap(params, energies):    
    cond = []
    L = params["L"]
    pot = params["pot"]
    x_pos =  [1*i for i in range(-10,L+10,1)]
    E_data = []
    delta=[0.001*i for i in range(-400,400,5)]
    #os.mkdir('./xtip_contours/wavefunctionplots/varyingpot/')
    for d in tqdm(delta):        
        for energy in (energies):
            params["Delta"]=d
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized()
            smatrix = kwant.smatrix(syst, energy,params=params)
            E_data.append(smatrix.submatrix((0, 0), (0, 0)).shape[0] -
                smatrix.transmission((0, 0), (0, 0)) +
                smatrix.transmission((0, 1), (0, 0)))                                        
        cond.append(np.transpose(E_data))
        E_data = []
    cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(delta,energies,cond,300,cmap='viridis')
    plt.title("Conductance vs Superconducting Gap, L=30")
    plt.xlabel("Superconducting Gap")
    plt.ylabel("Bias Voltage ")
    plt.savefig('./xtip_contours/wavefunctionplots/varyingpot/varyingdelta.png')
    fig.colorbar(con)
    cond =[]
    plt.close()
    
    
    
def xtip_cont(params, energies):
    cond = []   
    L=params["L"]
    x_pos =  np.linspace(-L/4,L+L/4,L+L/2 +1)
    E_data = []
    ypos= np.linspace(5,20,16)
    #os.mkdir('./xtip_contours/wavefunctionplots/contswithlines/')  
   # for y in tqdm(ypos):
    for x in tqdm(x_pos):
        for energy in energies:                
            params["ind"]=x           
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst, energy,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                    smatrix.transmission((0, 1), (0, 0)) )                  
        cond.append(np.transpose(E_data))
        E_data = []                 
    cond = np.transpose(cond)
    fig = plt.figure()    
    con = plt.contourf(x_pos,energies,cond,300, cmap='plasma')  
    fig.colorbar(con,label="Condcutance[$e^2/h$]")
    plt.title("G($x_{tip}$,$V_{bias}$), barrier= "+str(params["barrier"])+" L=" +str(params["L"])+', $V_{tip}$=' + str(params["pot"])+", $\gamma=$"+str(params["gam"])+", $V_{disorder}$="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+"), "+str(params["pot3"])+" at ("+str(params["x3"])+","+str(params["y3"])+")")
    plt.xlabel("Position of tip ($x_{tip}$)")
    plt.ylabel("Bias Voltage")
    plt.savefig('./xtip_contours/Final_Data/SGM/sinusoidalbarrier='+str(params["barrier"])+'pot='+str(params["pot"])+ "gamma="+str(params["gam"]) +"L=" + str(params["L"])+"y="+str(params["y0"])+"dis="+str(params["pot2"])+"("+str(params["xs"])+","+str(params["ys"])+str(params["pot3"])+"("+str(params["x3"])+","+str(params["y3"])+")"+".png")   
    cond = []
    plt.close


############################################################################
###############################Scanning Gate################################
############################################################################
                  
def xtippotcond(params, energies):
    cond = []   
    L=params["L"]
    x_pos =  np.linspace(-L/4,L+L/4,L+L/2 +1)
    E_data = []
    vtip= [0.01*i for i in range(0,101,1)]
    #os.mkdir('./xtip_contours/wavefunctionplots/contswithlines/')  
   # for y in tqdm(ypos):
    for x in tqdm(x_pos):
        for p in vtip:                
            params["ind"]=x 
            params["pot"]= p
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst, 0,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                    smatrix.transmission((0, 1), (0, 0)) )
                  
        cond.append(np.transpose(E_data))
        E_data = []                 
    cond = np.transpose(cond)
    fig = plt.figure()    
    con = plt.contourf(x_pos,vtip,cond,300, cmap='plasma')  
    fig.colorbar(con,label="Condcutance[$e^2/h$]")
    plt.title("G($x_{tip}$,$V_{tip}$), barrier= "+str(params["barrier"])+" L=" +str(params["L"])+', y position=' + str(params["y0"])+", $\gamma=$"+str(params["gam"]))#+", $V_{disorder}$="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+"), "+str(params["pot3"])+" at ("+str(params["x3"])+","+str(params["y3"])+")")
    plt.xlabel("Position of tip ($x_{tip}$)")
    plt.ylabel("$V_{tip}$")
   # for ham in np.linspace(0,0.05,51):  
   # i=0.007
    #line1 = plt.plot([1*i for i in range(-10,40,1)], i * np.ones(50),'--',color='b')        
    plt.savefig("./xtip_contours/Final_Data/G(x,pot)/gamma="+str(params["gam"]) +"L=" + str(params["L"])+"y="+str(params["y0"])+".png")#+"dis="+str(params["pot2"])+"("+str(params["xs"])+","+str(params["ys"])+str(params["pot3"])+"("+str(params["x3"])+","+str(params["y3"])+")"+".png")   
    cond = []
    plt.close
    
def xtipytip_contconductance(params, energies):
    cond = []
    L = params["L"]    
    W = params["W"]
   # os.mkdir('./xtip_contours/wavefunctionplots/G(x,y) Global Chemical Potential/L=30 E=0/')
    x_pos =  np.linspace(-L/4-1,L+L/4+1,L+L/2+3)
    E_data = []
    energy = [0.001*i for i in range(0,1000,5)] 
    y_pos= np.linspace(-3*L/4,3*L/4,3*L/2+1) 
    pot=[0.1*i for i in range(0,25,5)]
    en=0.005
   # for en in tqdm(energy):
       # os.mkdir("./xtip_contours/L=100/G(x,y)/Energy="+str(en)+"/")
       # for p in pot:
    for y in tqdm(y_pos):
        for x in x_pos:                    
            params["y0"]=y
            params["ind"]=x           
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst,en,params=params)
            E_data.append(smatrix.transmission((1, 0), (1, 1)) +
                smatrix.transmission((0, 1), (0, 0)))                  
        cond.append(np.transpose(E_data))
        E_data = []
    #cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(x_pos,y_pos,cond,300, cmap='plasma')        
    plt.title("G(x,y), $V_{tip}$=" + str(params["pot"])+", Energy="+ str(en)+", $\gamma$="+str(params["gam"])+", L="+str(params["L"])+", $V_{disorder}$="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+"), "+str(params["pot3"])+" at ("+str(params["x3"])+","+str(params["y3"])+")")
    plt.xlabel("$x_{tip}$")
    plt.ylabel("$y_{tip}$")
    fig.colorbar(con,label="Conductance[$e^2/h$]")     
    plt.savefig("./xtip_contours/Final_Data/G(x,y)/L="+str(params["L"])+"pot=" + str(params["pot"])+', energy='+str(en)+"disorder="+str(params["pot2"])+"("+str(params["xs"])+","+str(params["ys"])+"),"+str(params["pot3"])+"("+str(params["x3"])+","+str(params["y3"])+").png")    
    plt.close()
    cond = []
      
def xtipytip_contenergy(params, energies):
    cond = []
    #os.mkdir("./xtip_contours/energyEigenvalues/E(x,y)/smu=0_1/")
    L = params["L"]
    x_pos =  np.linspace(-L/4-1,L+L/4+1,L+L/2+3)
    E_data = []
    y_pos= np.linspace(-3*L/4,3*L/4,3*L/2+1) 
    pot=[0.01*i for i in range(0,200,10)]

    for y in tqdm(y_pos):
        for x in x_pos:
            params["ind"] = x
            params["y0"] = y
            
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
            H = H.tocsc()    
            eigs = sla.eigsh(H, k=4, sigma=0) 
            E_data.append(np.sort(eigs[0])[3])
        cond.append(np.transpose(E_data))
        E_data = []
    #cond = np.transpose(cond)
    fig = plt.figure()
    con = plt.contourf(x_pos,y_pos,cond,300, cmap='viridis')
    plt.title("$E_{1+}$(x,y),L="+str(params["L"])+', $\gamma$='+str(params["gam"])+", $V_{tip}$=" + str(params["pot"])+", barrier ="+str(params["barrier"]))#+" at ("+ str(params["xs"])+","+str(params["ys"]) +")")
    plt.xlabel("$x_{tip}$")
    plt.ylabel("$y_{tip}$")
    fig.colorbar(con, label="Energy")
    plt.savefig('./xtip_contours/Final_Data/E(x,y)/bar='+str(params["barrier"])+'E1+L='+str(params["L"])+'pot=' +str(params["pot"])+'.png')  
    cond = []
    plt.close()

        

    
def condSmuPot(params, energies):
  
    L = params["L"]
    chem = [ 0.01*i for i in range (0, 46,1)]
    pot = [ 0.01*i for i in range (0,121,1)]
    E_data = []
    cond = []
   # en=[0.01*i for i in range (0, 15, 1)]
    en=0.01
    #os.mkdir('./xtip_contours/wavefunctionplots/contswithlines/')
   # for en in tqdm(en):
    for m in tqdm(chem):
        for p in pot:                
            params["pot"]=p
            params["Smu"] = m
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst, en,params=params)
            E_data.append(smatrix.transmission((0, 1), (0, 0)) +
                smatrix.transmission((1, 0), (1, 1)) )               
        cond.append(np.transpose(E_data))
        E_data = []                 
   # cond = np.transpose(cond)
    fig = plt.figure()    
    con = plt.contourf(pot,chem,cond,300, cmap='plasma')  #np.arange(0., 2.005 , 0.005),
    fig.colorbar(con,label="Conductance[$e^2/h$]")
    plt.title("G($V_{tip}$ , $\mu_{SC}$) at ("+str(params["ind"])+","+str(params["y0"])+"), L=" +str(params["L"])+", $\gamma=$"+str(params["gam"])+", $V_{bias}$="+str(en))#+" disorder potential="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+")")
    plt.xlabel("$V_{tip}$")
    plt.ylabel("$\mu_{SC}$")
   # for ham in np.linspace(0,0.05,51):  
   # i=0.007
    #line1 = plt.plot([1*i for i in range(-10,40,1)], i * np.ones(50),'--',color='b')        
    plt.savefig('./xtip_contours/Final_Data/G(pot,smu)/bar='+str(params["barrier"])+'E='+str(en)+'L='+str(params["L"])+ "gamma="+str(params["gam"]) +"("+str(params["ind"])+","+str(params["y0"])+").png")    
    plt.close()
    cond=[]
        
def condGamPot(params, energies):
  
    L = params["L"]
    chem = [ 1*i for i in range (1, 71,1)]
    pot = [ 0.01*i for i in range (0,101,1)]
    E_data = []
    cond = []
    en=0#[0.01*i for i in range (0, 15, 1)]
    #os.mkdir('./xtip_contours/wavefunctionplots/contswithlines/')

    for m in tqdm(chem):
        for p in pot:                
            params["pot"]=p
            params["gam"] = m
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst, en ,params=params)
            E_data.append(smatrix.transmission((0, 1), (0, 0)) +
                smatrix.transmission((1, 0), (1, 1)) )               
        cond.append(np.transpose(E_data))
        E_data = []                 
   # cond = np.transpose(cond)
    fig = plt.figure()    
    con = plt.contourf(pot,chem,cond,300, cmap='plasma')  #np.arange(0., 2.005 , 0.005),
    fig.colorbar(con,label="Conductance[$e^2/h$]")
    plt.title("G($V_{tip}$ , $\gamma$) with parallel B-field at ("+str(params["ind"])+","+str(params["y0"])+"), L=" +str(params["L"])+", $V_{bias}$="+str(en))#+" disorder potential="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+")")
    plt.xlabel("$V_{tip}$")
    plt.ylabel("$\gamma$")
   # for ham in np.linspace(0,0.05,51):  
   # i=0.007
    #line1 = plt.plot([1*i for i in range(-10,40,1)], i * np.ones(50),'--',color='b')        
    plt.savefig('./xtip_contours/Final_Data/G(pot,gam)/parallelE='+str(en)+'L='+str(params["L"])+"("+str(params["ind"])+","+str(params["y0"])+").png")    
    plt.close()
    cond=[]
        
    
        
def condBSmu(params, energies):
  
    L = params["L"]
    chem = [ 0.01*i for i in range (0, 51,1)]
    pot = [ 0.01*i for i in range (0,101,1)]
    E_data = []
    cond = []
    en=0#[0.01*i for i in range (0, 15, 1)]
    #os.mkdir('./xtip_contours/wavefunctionplots/contswithlines/')

    for m in tqdm(chem):
        for p in pot:                
            params["mu"]=p
            params["B"] = m
            syst = makeNISIN2D(params)
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            smatrix = kwant.smatrix(syst, en ,params=params)
            E_data.append(smatrix.transmission((0, 1), (0, 0)) +
                smatrix.transmission((1, 0), (1, 1)) )               
        cond.append(np.transpose(E_data))
        E_data = []                 
   # cond = np.transpose(cond)
    fig = plt.figure()    
    con = plt.contourf(pot,chem,cond,300, cmap='plasma')  #np.arange(0., 2.005 , 0.005),
    fig.colorbar(con,label="Conductance[$e^2/h$]")
    plt.title("G($V_{tip}$ , $\gamma$) tau_y at ("+str(params["ind"])+","+str(params["y0"])+"), L=" +str(params["L"])+", $\gamma=$"+str(params["gam"])+", $V_{bias}$="+str(en))#+" disorder potential="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+")")
    plt.xlabel("mu")
    plt.ylabel("B")
   # for ham in np.linspace(0,0.05,51):  
   # i=0.007
    #line1 = plt.plot([1*i for i in range(-10,40,1)], i * np.ones(50),'--',color='b')        
   # plt.savefig('./xtip_contours/Final_Data/G(pot,gam)/tau_yE='+str(en)+'L='+str(params["L"])+ "gamma="+str(params["gam"]) +"("+str(params["ind"])+","+str(params["y0"])+").png")    
    #plt.close()
    cond=[]
            
def EnergySmuPot(params, energies):
    
    L = params["L"]
    chem = [ 0.01*i for i in range (0, 46,1)]
    pot = [ 0.01*i for i in range (0,121,1)]
    E_data = []
    cond = []
    for m in tqdm(chem):
        for p in pot:
            params["Smu"]=m
            params["pot"]=p
            syst = makeNISIN2D(params)                     
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
            H = H.tocsc()    
            eigs = sla.eigsh(H, k=4, sigma=0) 
            E_data.append(np.sort(eigs[0])[2])
        cond.append(np.transpose(E_data))
        E_data = []
    fig = plt.figure()
    con = plt.contourf(pot,chem,cond,200, cmap='viridis') 
   # con = plt.contourf(pot,chem,cond, np.arange(-0.0, .011, 0.0000001),cmap='viridis',extend='both')       
    plt.title("$E_{0+}$($V_{tip}$,$\mu_{SC}$),B="+str(params["B"])+" (" + str(params["ind"])+', ' + str(params["y0"])+"), $\gamma$="+str(params["gam"])+", L="+str(params["L"])+", barrier="+str(params["barrier"])+", $V_{disorder}$="+str(params["pot2"])+" at ("+str(params["xs"])+","+str(params["ys"])+"), "+str(params["pot3"])+" at ("+str(params["x3"])+","+str(params["y3"])+")")
    plt.xlabel("$V_{tip}$ ")
    plt.ylabel("$\mu_{SC}$")   
    plt.colorbar(con, label="Energy")
    plt.savefig('./xtip_contours/Final_Data/E(Smu,pot)/disE0+b='+str(params["B"])+'bar='+str(params["barrier"])+'(' + str(params["ind"])+', ' + str(params["y0"])+')gamma='+str(params["gam"])+"L="+str(params["L"])+'.png')    


def EnergyGamPot(params, energies):    
    
    gaml = [ 1*i for i in range (1,61,1)]
    pot = [ 0.01*i for i in range (0,101,1)]
    E_data = []
    cond = []
    for g in tqdm(gaml):
        for p in pot:
            params["gam"]=g
            params["pot"]=p
            syst = makeNISIN2D(params)                     
            plt.rcParams["figure.figsize"] = (8,5)
            syst = syst.finalized() 
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)    
            H = H.tocsc()    
            eigs = sla.eigsh(H, k=4, sigma=0) 
            E_data.append(np.sort(eigs[0])[2])
        cond.append(np.transpose(E_data))
        E_data = []
    fig = plt.figure()
    con = plt.contourf(pot,gaml,cond,200, cmap='viridis')    
    plt.title("$E_{0+}$($V_{tip}$, tip width), tip position=(" + str(params["ind"])+', ' + str(params["y0"])+"), L="+str(params["L"]))
    plt.xlabel("$V_{tip}$ ")
    plt.ylabel("Tip Width")
    fig.colorbar(con, label="Energy")
   # os.mkdir("./xtip_contours/PlotsForPaper/L="+str(params["L"])+"/E(vtip,tipwidth)/")
    plt.savefig("./xtip_contours/Final_Data/E(pot,gam)/newbe0L="+str(params["L"])+"(" + str(params["ind"])+', ' + str(params["y0"])+").png") 
    
    #plt.close()
    
    


############################################################################
############################## Wavefunction ################################
############################################################################
  

# Wavefunction at a fixed energy 
def WavefunctionEnergy(params):
        powt=[0.01*i for i in range(0,51,1)]
        energy=0.00
        mode = 0
         
        syst=makeNISIN2D(params)
        syst = syst.finalized()
        wf=kwant.wave_function(syst, energy, params=params)
        psi = wf(0)[mode] + wf(1)[mode]
        rho = kwant.operator.Density(syst)
        density = rho(psi)
        fig = kwant.plotter.map(syst, density)
        fig.suptitle(" Position of tip= ("+str(params["ind"])+","+str(params["y0"])+"), Energy=0, $V_{tip}$="+str(y)+", $\gamma$="+str(params["gam"]))
        fig.savefig("./xtip_contours/Final_Data/Wavefunction/bar="+str(params["barrier"])+"L="+str(params["L"])+"("+str(params["ind"])+","+str(params["y0"])+")gamma="+str(params["gam"])+"pot"+str(y)+".png")
           # plt.close



# Wavefunction for eigenmodes
def WavefunctionModes(params):
    
    def sorted_eigs(ev):
        evals, evecs = ev
        evals, evecs = map(np.array, zip(*sorted(zip(evals, evecs.transpose()))))
        return evals, evecs.transpose()
    syst=makeNISIN2D(params)
    syst=syst.finalized()    
    ham_mat = syst.hamiltonian_submatrix(sparse = True, params = params)
    evals, evecs = sorted_eigs(sla.eigsh(ham_mat.tocsc(), k=4, sigma=0))
    evecs = np.abs(evecs[0::4])**2+np.abs(evecs[1::4])**2+np.abs(evecs[2::4])**2+np.abs(evecs[3::4])**2
    fig = kwant.plotter.map(syst, (evecs[:,0])+(evecs[:,0]), colorbar = True, oversampling = 1)
    fig.suptitle("$E_{0}$ wavefunction, L ="+str(params["L"])+", Position of tip= ("+str(params["ind"])+","+str(params["y0"])+"), $V_{tip}$="+str(params["pot"])+", $\gamma$="+str(params["gam"]))#", $\mu_{SC}$="+str(params["Smu"]))
    fig.savefig("./xtip_contours/Final_Data/Wavefunction/E_0="+str(params["L"])+"("+str(params["ind"])+","+str(params["y0"])+")gamma="+str(params["gam"])+"pot"+str(params["pot"])+".png")
   # 
       




def main():
   
    freq = 2500
    duration = 500
    
    
    energies =  [0.001 * i for i in range(-110, 111,1 )]
    params = dict(mu=0.4, Delta=.1, alpha=0.8, t=1.0, barrier=3.0, pot =00. ,W = 5, 
                  L = 60, ind =15, B = 0.3, Smu=0.0,y0=2,gam=30, gam2=30,pot2=-0.0,xs=15,ys=2,pot3=0., x3=45, y3=2, pot4=0.0, x4=25, y4=2,
                   barl=1,pot5=0.0 , x5=55, y5=2,pot6=0.0,x6=40, y6=2)
 
   # condchem(params, energies) 
    #condpot(params, energies)
    #xtipytip_contenergyglobal(params, energies)
    plotSpectrum(params, energies)
   # condgam(params, energies)
   # condSmuPot(params, energies)
   # EnergySmuPot(params, energies)
  #  WavefunctionEnergy(params)
   # xtippotcond(params, energies)
   # xtip_cont(params, energies)
   # condBSmu(params, energies)  
    #VaryingGlobalChemPot(params, energies)
   # VaryingZeeman(params, energies)  
   # xtipytip_contconductance(params, energies)
    #WavefunctionModes(params)   
   # plotSpectrumChem(params, energies)
  #  barCon(params,energies)
 #   ZBP(params, energies)
   # poten(params, energies)
    #engam(params, energies)
   # condGamPot(params, energies)
   # EnergyGamPot(params, energies)
   # xtipytip_contenergy(params, energies)c
   # winsound.Beep(freq,duration)  
if __name__ == '__main__':
    main()
  
  
  
  
  
  
  
  
  
  

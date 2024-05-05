from __future__ import division
from PySteppables import *       
import math                      
import numpy as np                   
import CompuCell
import sys
import random
RNG=random.SystemRandom()  #draw true random sequence, overkill but why not?
#Motility Variables
CtoM=52  #cell to adhesion value, self-consistent across simulations as 26 but x2 due to double interface formation
BASAL=100 #baseline motility for L929, due to their extremely motile behaviour
SCF=0.5 #self-attenuator weighing basal motility vs loss of motility due to adhesion
#Self-Cutoff
ENDMCS=50000 #call runtime here directly
#Mitosis Variables
RADAVG=3 #average radius of the gaussian distribution to choose random radius
RADDEV=.5 #standard deviation of target radius, too low and division couples, too high and you'll lose cells at the start
MTFORCEMIN=-3*10**(-3.88) #negative mitosis driving force fluctuation, usually only need to change the exponential part
MTFORCEMAX=4*10**(-3.88)  #positive mitosis driving force fluctuation, usually change only the exponential part
#Signaling Variables
CONEXPSCF=10000 #Steady state expression of ligand expressed on a sender cell. This ligand is unaffected by signaling.
THETA=0 #time lag for expression of your constitutive, non-signaling affected ligand, start at 0 for simplicity, but can be adjusted depending on experiment results if known for generalizability
XI=1000 #controls how fast the sender cells reaches steady state for your constitutive, non-signaling affected ligand
FASTAPPROX=5000 #force approx for function of above variables at the time step, saves calling the mcs and doing the caluclation, purely computational speed effeciency

BETAONE = 6000
BETATWO = 6000
KAPPA=25000

EPSILONYG=1
EPSILONBR=1

THRESHOLD=7000 #activation threshold to change state  

#Single Cell Trace Variables
MARKEDCELLS=[1,2,51,228] # ID of cells to track if you desire single cell points tracked, change to fit setup

#Sampling and Comp Speed
RESOL=100 #Data sampling rate, choose to satisfy nyquist theorem if necessary
USEDNODES=8 #Choose a power of 2, otherwise the grids overlap and your simulation will eventually randomly crash, follow the recommendations given in the manual by developers


class ELUGMSteppable(SteppableBasePy):

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):

        self.pW1 = self.addNewPlotWindow(
            _title='Calibrate',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Phi',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW1.addPlot('BWtoYG', _style='Dots', _color='gray', _size=3) 
        self.pW1.addPlot('BWtoBW', _style='Dots', _color='cyan', _size=3)
        self.pW1.addPlot('BWtoOL', _style='Dots', _color='orange', _size=3)
        self.pW1.addPlot('BWtoRK', _style='Dots', _color='red', _size=3)
        self.pW1.addPlot('RKtoYG', _style='Dots', _color='gray', _size=5)
        self.pW1.addPlot('RKtoBW', _style='Dots', _color='cyan', _size=5)        
        self.pW1.addPlot('RKtoOL', _style='Dots', _color='orange', _size=5)   #plots for morphospace/hetergenity measure
        self.pW1.addPlot('RKtoRK', _style='Dots', _color='red', _size=5)
        
        self.pW2 = self.addNewPlotWindow(
            _title='Psi',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Psi',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW2.addPlot('Hamiltonian', _style='Dots', _color='white', _size=3) #measure the system total energy
 
        self.pW3 = self.addNewPlotWindow(
            _title='Types',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Count',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW3.addPlot('Y', _style='Dots', _color='gray', _size=3)
        self.pW3.addPlot('G', _style='Dots', _color='green', _size=3)
        self.pW3.addPlot('B', _style='Dots', _color='blue', _size=3)   
        self.pW3.addPlot('R', _style='Dots', _color='red', _size=3)
        self.pW3.addPlot('O', _style='Dots', _color='orange', _size=3)
        self.pW3.addPlot('W', _style='Dots', _color='cyan', _size=3) #count the types of cells in the simulation, along with how mny activate due to signaling
        self.pW3.addPlot('L', _style='Dots', _color='yellow', _size=3)
        self.pW3.addPlot('K', _style='Dots', _color='pink', _size=3)        
        
        self.pW4 = self.addNewPlotWindow(
            _title='Point System',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Count',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW4.addPlot('GFP', _style='Dots', _color='green', _size=3)
        self.pW4.addPlot('BFP', _style='Dots', _color='blue', _size=3) #measure average points per cell type, can be tied to average flurosence intensity
        self.pW4.addPlot('YFP', _style='Dots', _color='yellow', _size=3)
        self.pW4.addPlot('RFP', _style='Dots', _color='red', _size=3)
        
        self.pW5 = self.addNewPlotWindow(
            _title='Single Point System',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Count',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW5.addPlot('R1', _style='Dots', _color='blue', _size=3)
        self.pW5.addPlot('R2', _style='Dots', _color='gray', _size=3)
        self.pW5.addPlot('R3', _style='Dots', _color='green', _size=3)   
        self.pW5.addPlot('R4', _style='Dots', _color='red', _size=3)
        self.pW5.addPlot('T1', _style='Dots', _color='blue', _size=3)
        self.pW5.addPlot('T2', _style='Dots', _color='gray', _size=3)
        self.pW5.addPlot('T3', _style='Dots', _color='green', _size=3)   
        self.pW5.addPlot('T4', _style='Dots', _color='red', _size=3)  #single cell point traces 

        self.pW6 = self.addNewPlotWindow(
            _title='Sphericity',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Count',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True)                
        self.pW6.addPlot('ALL', _style='Dots', _color='orange', _size=3)
        self.pW6.addPlot('CORE', _style='Dots', _color='green', _size=3)  #bright field and merged field sphericity      
                
        global YtoY,YtoG,GtoY,YtoB,BtoY,YtoR,RtoY,GtoG,GtoB,BtoG,GtoR,RtoG,BtoB,BtoR,RtoB,RtoR,OtoY,YtoO,OtoG,GtoO,OtoB,BtoO,OtoR,RtoO,OtoO,WtoY,YtoW,WtoG,GtoW,WtoB,BtoW,WtoR,RtoW,WtoO,OtoW,WtoW  #the adhesion matrix, call these values and store them for motility code
        global LtoY,YtoL,LtoG,GtoL,LtoB,BtoL,LtoR,RtoL,LtoO,OtoL,LtoW,WtoL,LtoL,KtoY,YtoK,KtoG,GtoK,KtoB,BtoK,KtoR,RtoK,KtoO,OtoK,KtoW,WtoK,KtoL,LtoK,KtoK
        YtoY=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','Y','Type2','Y']))
        YtoG=GtoY=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','Y','Type2','G']))
        YtoB=BtoY=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','Y','Type2','B']))
        YtoR=RtoY=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','Y','Type2','R']))
        GtoG=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','G','Type2','G']))
        GtoB=BtoG=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','G','Type2','B']))
        GtoR=RtoG=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','G','Type2','R']))
        BtoB=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','B','Type2','B']))
        BtoR=RtoB=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','B','Type2','R']))
        RtoR=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','R','Type2','R']))
        OtoY=YtoO=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','O','Type2','Y']))    
        OtoG=GtoO=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','O','Type2','G']))
        OtoB=BtoO=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','O','Type2','B']))
        OtoR=RtoO=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','O','Type2','R']))
        OtoO=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','O','Type2','O']))
        WtoY=YtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','Y']))
        WtoG=GtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','G']))
        WtoB=BtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','B']))
        WtoR=RtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','R']))
        WtoO=OtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','O']))
        WtoW=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','W','Type2','W']))
        
        LtoY=YtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','Y']))
        LtoG=GtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','G']))
        LtoB=BtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','B']))
        LtoR=RtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','R']))
        LtoO=OtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','O']))
        LtoW=WtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','W']))
        LtoL=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','L','Type2','L']))
        
        KtoY=YtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','Y']))
        KtoG=GtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','G']))
        KtoB=BtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','B']))
        KtoR=RtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','R']))
        KtoO=OtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','O']))
        KtoW=WtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','W']))
        KtoL=LtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','L']))
        KtoK=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','K','Type2','K']))        

        for cell in self.cellList:
            cell.dict["RDM"]=RNG.gauss(RADAVG,RADDEV) #assign the cells a random target radius
            cell.lambdaSurface=2.2                    #temporary value, will be changed in later code
            cell.targetSurface=4*math.pi*cell.dict["RDM"]**2  #spherical surface area
            cell.lambdaVolume=2.2                     #temporary value, will be changed in later code
            cell.targetVolume=(4/3)*math.pi*cell.dict["RDM"]**3 #spherical volume
            cell.dict["PTS"]=[KAPPA,0,0,0] #Order is green ,red, blue, yellow omg this is poorly organized
            cell.dict["P"]=[0,0]                      #activation counter, counts how many cells are active due to signaling at any given time

    def step(self,mcs):                 
        NUMTY=0 #number of type Y
        NUMTG=0 #number of type G
        NUMTB=0 #number of type B
        NUMTR=0 #number of type R
        NUMTO=0
        NUMTW=0
        NUMTL=0
        NUMTK=0
        
        GFPPTS=0
        RFPPTS=0 
        BFPPTS=0
        YFPPTS=0
        
        SYPSI=0 #system hamiltonian over all interaction over configuration

        CSABWYG=0
        CSABWBW=0
        CSABWOL=0
        CSABWRK=0
        
        CSARKYG=0
        CSARKBW=0
        CSARKOL=0
        CSARKRK=0
        
        BWCYG=0
        BWCBW=0
        BWCOL=0
        BWCRK=0

        RKCYG=0
        RKCBW=0
        RKCOL=0
        RKCRK=0
        
        SUMALLSF=0 #total bright field surface area
        SUMALLVL=0 #total bright field volume
        SUMCORESF=0 #total color field surface area
        SUMCOREVL=0 #total color field volume
        
        NAR=0 #number of activated red cells due to signaling
        NAG=0 #number of activated green cells due to signaling
        
        if mcs==1:
            self.changeNumberOfWorkNodes(USEDNODES) #set to necessary computational nodes                  

        for cell in self.cellList: #iterate over cell list
            CSAY=0 #each cell detect how much sirface area it shares with Y cells
            CSAG=0 #each cell detect how much sirface area it shares with G cells
            CSAB=0 #each cell detect how much sirface area it shares with B cells
            CSAR=0 #each cell detect how much sirface area it shares with R cells
            CSAM=0 #each cell detect how much sirface area it shares with medium
            CSAO=0
            CSAW=0
            CSAL=0
            CSAK=0
            
            PTSY=0 #each cell gains points from neighbor type Y
            PTSG=0 #each cell gains points from neighbor type G
            PTSB=0 #each cell gains points from neighbor type B
            PTSR=0 #each cell gains points from neighbor type R
            PTSO=0
            PTSW=0
            PTSL=0
            PTSK=0
            DTREPO=0 #delta reporter One
            DTREPT=0 #deta reporter Two
            DTREPH=0 #delta reporter three
            DTREPF=0 #reporter four please stop expalinding this code -CL

            for neighbor, commonSurfaceArea in self.getCellNeighborDataList(cell): #iterate for each cell its neighbors
                if neighbor is None: #none type refers to medium 
                    continue
                if neighbor.type==1: #gray cells
                    CSAY+=commonSurfaceArea #total common surface area with gray cells
                    PTSY+=0
                if neighbor.type==2: #green cells
                    CSAG+=commonSurfaceArea #total common surface area with green cells
                    PTSG+=commonSurfaceArea*neighbor.dict["PTS"][0]/neighbor.surface
                if neighbor.type==3: #blue cells
                    CSAB+=commonSurfaceArea #total common surface area with blue cells
                    PTSB+=0
                if neighbor.type==6: #cyan
                    CSAW+=commonSurfaceArea #total common surface area with red cells  
                    PTSW+=commonSurfaceArea*neighbor.dict["PTS"][0]/neighbor.surface                                                 
                if neighbor.type==5: #orange cells
                    CSAO+=commonSurfaceArea
#                    PTSO+=commonSurfaceArea*CONEXPSCF/neighbor.surface
                if neighbor.type==7: #yellow? cells
                    CSAL+=commonSurfaceArea #total common surface area with red cells  
#                    PTSL+=commonSurfaceArea*CONEXPSCF/neighbor.surface
                if neighbor.type==4: #red cells
                    CSAR+=commonSurfaceArea #total common surface area with red cells 
#                    PTSR+=commonSurfaceArea*CONEXPSCF/neighbor.surface 
                if neighbor.type==8: #pink cells
                    CSAK+=commonSurfaceArea #total common surface area with red cells  
#                    PTSK+=commonSurfaceArea*CONEXPSCF/neighbor.surface                    
            CSAM=cell.surface-(CSAY+CSAG+CSAB+CSAR+CSAO+CSAW+CSAL+CSAK) #alternative method to calculate common surface area with medium                

# VETTING CODE                                        
#             if cell.id==3:
#                 for neighbor, commonSurfaceArea in self.getCellNeighborDataList(cell): #iterate for each cell its neighbors
#                     if neighbor:
#                         print "NID", neighbor.id, "TY", neighbor.type, "NCSA", commonSurfaceArea, "DICT", neighbor.dict["PTS"][0], "NS", neighbor.surface
#                 print "CID", cell.id, "CT",cell.type, "CS", cell.surface
#                 print CSAY, CSAG, PTSG, CSAB, PTSB, CSAR, PTSR, CSAM
           
            if (cell.type==1 or cell.type==2 or cell.type==3 or cell.type==6):
                DTREPO=(1/(1+np.exp(-((PTSG+PTSW)-BETAONE))))-(1/KAPPA)*cell.dict["PTS"][2]         #BFP       GRBY 0123
                cell.dict["PTS"][2]+=DTREPO                                                          
                DTREPT=(1/(1+np.exp(((cell.dict["PTS"][2])-BETATWO))))-(1/KAPPA)*cell.dict["PTS"][0]    #GFP 
                cell.dict["PTS"][0]+=DTREPT                                                                                                      
#            if (cell.type==5 or cell.type==4 or cell.type==7 or cell.type==8):
#                DTREPH=(1/(1+np.exp(-((PTSY+PTSG+PTSB+PTSW)-BETA))))-(1/KAPPA)*cell.dict["PTS"][3]           #YFP  
#                cell.dict["PTS"][3]+=DTREPH
#                DTREPF=(1/(1+np.exp(-((cell.dict["PTS"][3])-BETA))))-(1/KAPPA)*cell.dict["PTS"][1]           #RFP mcherry      
#                cell.dict["PTS"][1]+=DTREPF                
                                                             
            if (cell.type==1 or cell.type==2 or cell.type==3 or cell.type==6): #FACS PLOTS AND CADHERIN LINAKGE TO STATE
                if cell.dict["PTS"][0]>=THRESHOLD and cell.dict["PTS"][2]<THRESHOLD:   #GFP+ BFP- state
                    cell.type=2
                if cell.dict["PTS"][2]>=THRESHOLD and cell.dict["PTS"][0]<THRESHOLD:   #GFP- BFP+ state
                    cell.type=3                 
                if cell.dict["PTS"][0]>=THRESHOLD and cell.dict["PTS"][2]>=THRESHOLD:   #GFP+ BFP+ state
                    cell.type=6
                if cell.dict["PTS"][0]<THRESHOLD and cell.dict["PTS"][2]<THRESHOLD:   #GFP- BFP- state gray color
                    cell.type=1
            if (cell.type==5 or cell.type==4 or cell.type==7 or cell.type==8):
                if cell.dict["PTS"][3]>=THRESHOLD and cell.dict["PTS"][1]<THRESHOLD:   #YFP+ RFP-
                    cell.type=7
                if cell.dict["PTS"][1]>=THRESHOLD and cell.dict["PTS"][3]<THRESHOLD:   #YFP- RFP+ state
                    cell.type=4                 
                if cell.dict["PTS"][1]>=THRESHOLD and cell.dict["PTS"][3]>=THRESHOLD:   #YFP+ RFP+ state
                    cell.type=8
                if cell.dict["PTS"][1]<THRESHOLD and cell.dict["PTS"][3]<THRESHOLD:   #YFP- RFP- state orange color
                    cell.type=5
                    
            if cell.type==1: #gray cells
                SUMALLSF+=CSAM #grays cell surface area count under bright field
                SUMALLVL+=cell.volume #gray cell volume count under bright field                                
                cell.lambdaSurface=2.2            #change depending on cell adhesitivity
                cell.lambdaVolume=2.2             #change depending on cell adhesitivity  
                NUMTY+=1                          #count the number if gray cells
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+YtoY*CSAY+YtoG*CSAG+YtoB*CSAB+YtoR*CSAR+YtoO*CSAO+YtoW*CSAW+YtoL*CSAL+YtoK*CSAK)/cell.surface #corrected cell motility, tune based on adhesive neighbors, vetted
                GFPPTS+=cell.dict["PTS"][0]
                BFPPTS+=cell.dict["PTS"][2]

            if cell.type==2: #green cells
                SUMALLSF+=CSAM          #green cells surface area under bright field
                SUMALLVL+=cell.volume   #green cell volume under bright field
                cell.lambdaSurface=2.2            #change depending on cell adhesitivity
                cell.lambdaVolume=2.2             #change depending on cell adhesitivity    
                NUMTG+=1                          #count the number of green cells
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+GtoY*CSAY+GtoG*CSAG+GtoB*CSAB+GtoR*CSAR+GtoO*CSAO+GtoW*CSAW+GtoL*CSAL+GtoK*CSAK)/cell.surface #corrected cell motility, tune based on adhesive neighbors, vetted
                GFPPTS+=cell.dict["PTS"][0]
                BFPPTS+=cell.dict["PTS"][2]
             
            if cell.type==3: #blue cells            
                SUMALLSF+=CSAM # blue surface area contributes to bright field surface area
                SUMALLVL+=cell.volume #blue volume contributes to bright field volume 
                SUMCORESF+=(CSAY+CSAG+CSAO+CSAR+CSAM+CSAL+CSAK) #goal is to asses stability of blue/w core, all other cells considered invis in this field
                SUMCOREVL+=cell.volume
                CSABWYG+=(CSAY+CSAG)/cell.surface
                if (CSAY+CSAG)>0:
                    BWCYG+=1                
                CSABWBW+=(CSAB+CSAW)/cell.surface
                if (CSAB+CSAW)>0:
                    BWCBW+=1 
                CSABWOL+=(CSAO+CSAL)/cell.surface
                if (CSAO+CSAL)>0:
                    BWCOL+=1 
                CSABWRK+=(CSAR+CSAK)/cell.surface
                if (CSAR+CSAK)>0:
                    BWCRK+=1
                cell.lambdaSurface=1.0           #change depending on cell adhesitivity
                cell.lambdaVolume=1.0            #change depending on cell adhesitivity      
                NUMTB+=1                         #count number of blue cells
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+BtoY*CSAY+BtoG*CSAG+BtoB*CSAB+BtoR*CSAR+BtoO*CSAO+BtoW*CSAW+BtoL*CSAL+BtoK*CSAK)/cell.surface # corrected cell motility, vetted
                GFPPTS+=cell.dict["PTS"][0]
                BFPPTS+=cell.dict["PTS"][2]
                
            if cell.type==6: #mixed color cell choice color
                SUMALLSF+=CSAM # blue surface area contributes to bright field surface area
                SUMALLVL+=cell.volume #blue volume contributes to bright field volume 
                SUMCORESF+=(CSAY+CSAG+CSAO+CSAR+CSAM+CSAL+CSAK) #goal is to asses stability of blue/w core, all other cells considered invis in this field
                SUMCOREVL+=cell.volume
                CSABWYG+=(CSAY+CSAG)/cell.surface
                if (CSAY+CSAG)>0:
                    BWCYG+=1                
                CSABWBW+=(CSAB+CSAW)/cell.surface
                if (CSAB+CSAW)>0:
                    BWCBW+=1 
                CSABWOL+=(CSAO+CSAL)/cell.surface
                if (CSAO+CSAL)>0:
                    BWCOL+=1 
                CSABWRK+=(CSAR+CSAK)/cell.surface
                if (CSAR+CSAK)>0:
                    BWCRK+=1      
                cell.lambdaSurface=1.0            #change depending on cell adhesitivity
                cell.lambdaVolume=1.0             #change depending on cell adhesitivity      
                NUMTW+=1
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+WtoY*CSAY+WtoG*CSAG+WtoB*CSAB+WtoR*CSAR+WtoO*CSAO+WtoW*CSAW+WtoL*CSAL+WtoK*CSAK)/cell.surface #corrected cell motility, vetted    
                GFPPTS+=cell.dict["PTS"][0]
                BFPPTS+=cell.dict["PTS"][2]

            if cell.type==5: #orange cells                   
                SUMALLSF+=CSAM          #red cells are visible under bright field and thus conribute surface area
                SUMALLVL+=cell.volume   #red cell volume contributes to bright field                                   
                cell.lambdaSurface=2.2            #change depending on cell adhesitivity
                cell.lambdaVolume=2.2             #change depending on cell adhesitivity      
                NUMTO+=1
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+OtoY*CSAY+OtoG*CSAG+OtoB*CSAB+OtoR*CSAR+OtoO*CSAO+OtoW*CSAW+OtoL*CSAL+OtoK*CSAK)/cell.surface #corrected cell motility, vetted           
                YFPPTS+=cell.dict["PTS"][3]
                RFPPTS+=cell.dict["PTS"][1] #count number of points of BR cells     

            if cell.type==7: #yellow cells                   
                SUMALLSF+=CSAM          
                SUMALLVL+=cell.volume                                  
                cell.lambdaSurface=2.2           
                cell.lambdaVolume=2.2                 
                NUMTL+=1
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+LtoY*CSAY+LtoG*CSAG+LtoB*CSAB+LtoR*CSAR+LtoO*CSAO+LtoW*CSAW+LtoL*CSAL+LtoK*CSAK)/cell.surface #corrected cell motility, vetted           
                YFPPTS+=cell.dict["PTS"][3]
                RFPPTS+=cell.dict["PTS"][1] #count number of points of BR cells    

            if cell.type==4: #red cells
                SUMALLSF+=CSAM          #red cells are visible under bright field and thus conribute surface area
                SUMALLVL+=cell.volume   #red cell volume contributes to bright field                         
                CSARKYG+=(CSAY+CSAG)/cell.surface
                if (CSAY+CSAG)>0:
                    RKCYG+=1                
                CSARKBW+=(CSAB+CSAW)/cell.surface
                if (CSAB+CSAW)>0:
                    RKCBW+=1 
                CSARKOL+=(CSAO+CSAL)/cell.surface
                if (CSAO+CSAL)>0:
                    RKCOL+=1 
                CSARKRK+=(CSAR+CSAK)/cell.surface
                if (CSAR+CSAK)>0:
                    RKCRK+=1 
                cell.lambdaSurface=1.0            #change depending on cell adhesitivity
                cell.lambdaVolume=1.0             #change depending on cell adhesitivity      
                NUMTR+=1                          #count number of red cells
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+RtoY*CSAY+RtoG*CSAG+RtoB*CSAB+RtoR*CSAR+RtoO*CSAO+RtoW*CSAW+RtoL*CSAL+RtoK*CSAK)/cell.surface #corrected cell motility, vetted
                YFPPTS+=cell.dict["PTS"][3]
                RFPPTS+=cell.dict["PTS"][1] #count number of points of BR cells   

            if cell.type==8: #pink cells
                SUMALLSF+=CSAM          #red cells are visible under bright field and thus conribute surface area
                SUMALLVL+=cell.volume   #red cell volume contributes to bright field                         
                CSARKYG+=(CSAY+CSAG)/cell.surface
                if (CSAY+CSAG)>0:
                    RKCYG+=1                
                CSARKBW+=(CSAB+CSAW)/cell.surface
                if (CSAB+CSAW)>0:
                    RKCBW+=1 
                CSARKOL+=(CSAO+CSAL)/cell.surface
                if (CSAO+CSAL)>0:
                    RKCOL+=1 
                CSARKRK+=(CSAR+CSAK)/cell.surface
                if (CSAR+CSAK)>0:
                    RKCRK+=1 
                cell.lambdaSurface=1.0            #change depending on cell adhesitivity
                cell.lambdaVolume=1.0             #change depending on cell adhesitivity      
                NUMTK+=1                          #count number of red cells
                cell.fluctAmpl=BASAL+SCF*(CtoM*CSAM+KtoY*CSAY+KtoG*CSAG+KtoB*CSAB+KtoR*CSAR+KtoO*CSAO+KtoW*CSAW+KtoL*CSAL+KtoK*CSAK)/cell.surface #corrected cell motility, vetted
                YFPPTS+=cell.dict["PTS"][3]
                RFPPTS+=cell.dict["PTS"][1] #count number of points of BR cells   

           
            
            SYPSI+=cell.fluctAmpl #sensitive measure to sorting events
            
#vetting code
            #print "NUMTO", NUMTO, "NUMTW", NUMTW

#             if mcs%RESOL==0: #record points for signle cells traces        
#                 if cell.id==MARKEDCELLS[0]:
#                     self.pW5.addDataPoint("R1", mcs, cell.dict["PTS"][0])        
#                     self.pW5.addDataPoint("T1", mcs, cell.type) 
#                     self.pW5.addDataPoint("R2", mcs, cell.dict["PTS"][1])        
#                     self.pW5.addDataPoint("T2", mcs, cell.type)                    
#                 if cell.id==MARKEDCELLS[1]:
#                     self.pW5.addDataPoint("R3", mcs, cell.dict["PTS"][0])        
#                     self.pW5.addDataPoint("T3", mcs, cell.type) 
#                     self.pW5.addDataPoint("R4", mcs, cell.dict["PTS"][1])        
#                     self.pW5.addDataPoint("T4", mcs, cell.type)
            
        
        if mcs%RESOL==0:
            
            self.pW2.addDataPoint("Hamiltonian", mcs, SYPSI)
            
            self.pW3.addDataPoint("Y", mcs, NUMTY)
            self.pW3.addDataPoint("G", mcs, NUMTG) 
            self.pW3.addDataPoint("B", mcs, NUMTB)   
            self.pW3.addDataPoint("R", mcs, NUMTR)
            self.pW3.addDataPoint("O", mcs, NUMTO)
            self.pW3.addDataPoint("W", mcs, NUMTW)
            self.pW3.addDataPoint("L", mcs, NUMTL)
            self.pW3.addDataPoint("K", mcs, NUMTK)

            self.pW4.addDataPoint("GFP", mcs, GFPPTS/(NUMTY+NUMTG+NUMTB+NUMTW))
            self.pW4.addDataPoint("BFP", mcs, BFPPTS/(NUMTY+NUMTG+NUMTB+NUMTW))
#            self.pW4.addDataPoint("YFP", mcs, YFPPTS/(NUMTO+NUMTL+NUMTR+NUMTK))
#            self.pW4.addDataPoint("RFP", mcs, RFPPTS/(NUMTO+NUMTL+NUMTR+NUMTK))
            
            if BWCYG==0:
                self.pW1.addDataPoint("BWtoYG", mcs, 0)
            if BWCYG>0:
                self.pW1.addDataPoint("BWtoYG", mcs, CSABWYG/BWCYG)
            if BWCBW==0:
                self.pW1.addDataPoint("BWtoBW", mcs, 0)
            if BWCBW>0:
                self.pW1.addDataPoint("BWtoBW", mcs, CSABWBW/BWCBW)
            if BWCOL==0:
                self.pW1.addDataPoint("BWtoOL", mcs, 0)
            if BWCOL>0:
                self.pW1.addDataPoint("BWtoOL", mcs, CSABWOL/BWCOL)
            if BWCRK==0:
                self.pW1.addDataPoint("BWtoRK", mcs, 0)
            if BWCRK>0:
                self.pW1.addDataPoint("BWtoRK", mcs, CSABWRK/BWCRK)
                
            if RKCYG==0:
                self.pW1.addDataPoint("RKtoYG", mcs, 0)
            if RKCYG>0:
                self.pW1.addDataPoint("RKtoYG", mcs, CSARKYG/RKCYG)
            if RKCBW==0:
                self.pW1.addDataPoint("RKtoBW", mcs, 0)
            if RKCBW>0:
                self.pW1.addDataPoint("RKtoBW", mcs, CSARKBW/RKCBW)
            if RKCOL==0:
                self.pW1.addDataPoint("RKtoOL", mcs, 0)
            if RKCOL>0:
                self.pW1.addDataPoint("RKtoOL", mcs, CSARKOL/RKCOL)
            if RKCRK==0:
                self.pW1.addDataPoint("RKtoRK", mcs, 0)
            if RKCRK>0:
                self.pW1.addDataPoint("RKtoRK", mcs, CSARKRK/RKCRK)
                            
            if SUMALLSF==0:
                self.pW6.addDataPoint("ALL", mcs, 0)                
            if SUMALLSF>0: 
                self.pW6.addDataPoint("ALL", mcs, ((math.pi**(1/3))*(6*SUMALLVL)**(2/3))/SUMALLSF)
            if SUMCORESF==0:
                self.pW6.addDataPoint("CORE", mcs, 0)
            if SUMCORESF>0:
                self.pW6.addDataPoint("CORE", mcs, ((math.pi**(1/3))*(6*SUMCOREVL)**(2/3))/SUMCORESF)
             
            if mcs==ENDMCS:
                fileName = "Calibrate" + str(mcs) + ".txt"
                self.pW1.savePlotAsData(fileName)                
                fileName = "PSI" + str(mcs) + ".txt"
                self.pW2.savePlotAsData(fileName)
                fileName = "FOU" + str(mcs) + ".txt"
                self.pW3.savePlotAsData(fileName)
                fileName = "SIG" + str(mcs) + ".txt"
                self.pW4.savePlotAsData(fileName)
                fileName = "SCSIG" + str(mcs) + ".txt"
                self.pW5.savePlotAsData(fileName)                 
                fileName = "Sphericity" + str(mcs) + ".txt"
                self.pW6.savePlotAsData(fileName)                 
                self.stopSimulation()                 
    def finish(self):
        pass

class ExtraFields(SteppableBasePy):
  def __init__(self,_simulator,_frequency=1):
    SteppableBasePy.__init__(self,_simulator,_frequency)
    
    self.scalarFieldGFP=self.createScalarFieldCellLevelPy("GFP")
    self.scalarFieldmCherry=self.createScalarFieldCellLevelPy("mCherry")
    self.scalarFieldBFP=self.createScalarFieldCellLevelPy("BFP")
    self.scalarFieldYFP=self.createScalarFieldCellLevelPy("YFP")

  def step(self,mcs):
    self.scalarFieldGFP.clear()
    self.scalarFieldmCherry.clear()
    self.scalarFieldBFP.clear()
    self.scalarFieldYFP.clear()
    
    for cell in self.cellList:       
        self.scalarFieldGFP[cell]=cell.dict["PTS"][0]/KAPPA
        self.scalarFieldmCherry[cell]=cell.dict["PTS"][1]/KAPPA
        self.scalarFieldBFP[cell]=cell.dict["PTS"][2]/KAPPA
        self.scalarFieldYFP[cell]=cell.dict["PTS"][3]/KAPPA
        
from PySteppablesExamples import MitosisSteppableBase

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
        self.setParentChildPositionFlag(0) #randomize child cell position, see developer manual
    def step(self,mcs):        
        cells_to_divide=[]          #gen cells to divide list
        for cell in self.cellList:
            cell.dict["RDM"]+=RNG.uniform(MTFORCEMIN,MTFORCEMAX) #make cells grow in target radius by this much
            cell.targetSurface=4*math.pi*cell.dict["RDM"]**2 #spherical surface area
            cell.targetVolume=(4/3)*math.pi*cell.dict["RDM"]**3 #spherical volume
            if cell.volume>2*(4/3)*math.pi*RADAVG**3: #divide at two times the mean radius initialized with               
                cells_to_divide.append(cell)           #add these cells to divide list
                
        for cell in cells_to_divide:
            self.divideCellRandomOrientation(cell)  #divide the cells

    def updateAttributes(self):
        self.parentCell.dict["RDM"]=RNG.gauss(RADAVG,RADDEV) #reassign new target radius
        self.parentCell.targetVolume=(4/3)*math.pi*self.parentCell.dict["RDM"]**3 #new target volume
        self.parentCell.targetSurface=4*math.pi*self.parentCell.dict["RDM"]**2 #new target surface area
        self.cloneParent2Child()  #copy characterstics to child cell, indlucig signaling
        self.childCell.dict["P"][0]=0 #reset the activation counter, we dont care about cells from activated parent
        self.childCell.dict["P"][1]=0 #reset the activation counter, we dont care about cells from activated parent
        
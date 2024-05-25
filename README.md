# Inhibitory-Circuits-for-SynDev

This github contains sample codes for the following manuscript: https://doi.org/10.1021/acssynbio.4c00230

This code runs on CompuCell3D v3.7.8 which can be found at the organization's website, along with instructions. Instructions can also be found at the page for the original model: https://github.com/lmorsut/Lam_Morsut_GJSM

The repository here contains example codes for the above manuscript.

I apologize for the labelling convention in the code, as I had originally used different colors for the coloring of cell states. 
However, the contrast ended up being poor so I recolored with the final chosen colors as published. 
I give the recoloring key below.

Please ignore any mention of fluroscent proteins (i.e. GFP, RFP, BFP, etc) in the code.

cell type=1=Y is gray

cell type=2=G is blue

cell type=3=B is red

cell type=6=R is cyan

cell.dict["PTS"][0]= not GFP as stated in the code but blue color

cell.dict["PTS"][1]= non specified color for activating amplifer

cell.dict["PTS"][2]= not BFP as stated in the code but red color

cell.dict["PTS"][3]= unused color

Again, I apologize for the confusion. Please feel free to recolor as desired.
Contact me for questions or help: calvin.lam.k@gmail.com

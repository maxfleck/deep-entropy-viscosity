import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feos.si import * # SI numbers and constants
from feos.pcsaft import *
from feos.eos import *
import feos

import os
#import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from filelock import FileLock
from torch.utils.data import random_split,DataLoader, TensorDataset
#from typing import Dict
#from ray import train, tune
#from ray.train import Checkpoint
#from ray.tune.schedulers import ASHAScheduler


def collision_integral( T, p):
    """
    computes analytical solution of the collision integral

    T: reduced temperature
    p: parameters

    returns analytical solution of the collision integral
    """
    A,B,C,D,E,F,G,H,R,S,W,P = p
    return A/T**B + C/np.exp(D*T) + E/np.exp(F*T) + G/np.exp(H*T) + R*T**B*np.sin(S*T**W - P)

def get_omega22(red_temperature):
    """
    computes analytical solution of the omega22 collision integral

    red_temperature: reduced temperature

    returns omega22
    """
    p22 = [ 
         1.16145,0.14874,0.52487,
         0.77320,2.16178,2.43787,
         0.0,0.0,-6.435/10**4,
         18.0323,-0.76830,7.27371
        ]
    return collision_integral(red_temperature,p22)

def get_CE_viscosity_reference_new( data ):
    """
    computes Chapman-Enskog viscosity reference for an array of temperatures
    uses pc-saft parameters

    temperature: array of temperatures
    saft_parameters: pc saft parameter object build with feos

    returns reference
    """
    epsilon = data["epsilon_k"]*KELVIN
    sigma   = data["sigma"]*ANGSTROM
    m       = data["m"]
    M       = data["molarweight"]*GRAM/MOL
    temperature = data["temperature"]*KELVIN
    density     = data["molar_density"]*MOL/METER**3
    red_temperature = temperature/epsilon
    red_density     = density*sigma**3*NAV

    omega22 = get_omega22(red_temperature)

    sigma2 = sigma**2
    M_SI = M

    sq1  = np.sqrt( M_SI * KB * temperature / NAV /np.pi /METER**2 / KILOGRAM**2 *SECOND**2 ) *METER*KILOGRAM/SECOND
    div1 = omega22 * sigma2 * m
    viscosity_reference = 5/16* sq1 / div1 #*PASCAL*SECOND
    
    data["red_temperature"] = red_temperature
    data["red_density"] = red_density
    data["ln_eta_ref_new"] = np.log( viscosity_reference /PASCAL/SECOND )
    return data

def get_resd_entropy(sub):
    
    M = sub["molarweight"]*(GRAM/MOL)
    m = sub["m"]
    
    identifier = Identifier( cas="000-00-0", name="dummy" )
    saftrec = PcSaftRecord( m=sub["m"], sigma=sub["sigma"], epsilon_k=sub["epsilon_k"],
                          kappa_ab=sub["kappa_ab"],epsilon_k_ab=sub["epsilon_k_ab"],mu=sub["mu"] )

    pr = PureRecord(identifier=identifier,model_record=saftrec,molarweight=sub["molarweight"])

    saft_paras = PcSaftParameters.from_records( [pr], np.array([[0.0]] ) )

    eos = EquationOfState.pcsaft(saft_paras)


    if sub["state"] == "liquid":
        state = State(eos,temperature=sub["temperature"]*KELVIN, 
                      pressure=sub["pressure"]*PASCAL,
                      density_initialization="liquid"
                     )
    elif sub["state"] == "vapor":
        state = State(eos,temperature=sub["temperature"]*KELVIN, 
                      pressure=sub["pressure"]*PASCAL,
                      density_initialization="vapor"
                     )        
    else:
        state = State(eos,temperature=sub["temperature"]*KELVIN, 
                      pressure=sub["pressure"]*PASCAL,
                     ) 
        
    sub["resd_entropy"]           = -state.specific_entropy(Contributions.ResidualNvt)/ KB /NAV *M/m
    sub["molar_density"]          = state.density / MOL * METER**3
    
    #crit = state.critical_point(eos)
    #sub["critical_molar_density"] = crit.density / MOL * METER**3
    #sub["critical_temperature"]   = crit.temperature / KELVIN        
    #sub["red_temperature_crit"]   = sub["temperature"]*KELVIN / crit.temperature
    #sub["red_density_crit"]       = state.density / crit.density

    return sub

class DeepEntropyDataset(Dataset ):
    
    # Constructor with defult values 
    def __init__(self, data, info, additional_features, train=False):

        self.features = info["features"]
        self.y_key    = info["y_pred"]
   
        self.additional_features = int(additional_features)
        self.X_min   = info["X_min"]
        self.y_min   = info["y_min"]
        self.X_range = info["X_range"]
        self.y_range = info["y_range"]
        
        features_norm = [ x + "_normed" for x in self.features ]
        y_key_norm = self.y_key + "_normed"        
        data[ features_norm ] = self.normX( data[self.features] )

        if self.additional_features >= 0:
            additional_feature_names = []        
            for i in range(self.additional_features):
                x = i+2
                name = "resd_entropy_"+str(int(x))
                additional_feature_names.append(name)
                data[name] =  data["resd_entropy_normed"]**x        
            features_norm = features_norm + additional_feature_names
        else:
            print("no resd_entropy used")
            self.features = features_norm.remove('resd_entropy_normed')
        self.X_norm = torch.Tensor( np.array( data[features_norm] ) )

        if train:
            data[ y_key_norm ]    = self.normy( data[self.y_key] )           
            self.y_norm = torch.Tensor( np.array( data[y_key_norm] ) )           
        
        self.len = self.X_norm.shape[0]
        return
    
    # Getter
    def __getitem__(self, index):
        sample = self.X_norm[index], self.y_norm[index]
        #if self.transform:
        #    sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len
    
    def normX(self,X):
        X = X - self.X_min
        X = X / self.X_range    
        return X
        
    def normy(self,y):
        y = y - self.y_min
        y = y / self.y_range 
        return y
    
    def unormX(self,X):
        X = X * self.X_range
        X = X + self.X_min
        return X
        
    def unormy(self,y):
        y = y * self.y_range
        y = y + self.y_min                
        return y  
    
class DeepEntropyNet(torch.nn.Module):

    def __init__(self, num_parameters, num_features,   
                 num_used_features=-1, 
                 paranet_layer_units=[32],
                 refnet_layer_units=[16],
                 batch_norm=True,
                 use_base=False
                ):
        super(DeepEntropyNet, self).__init__()

        self.num_parameters      = num_parameters
        self.num_features        = num_features
        if num_used_features==-1 :
            self.num_used_features   = num_features  
            num_used_features        = num_features
        else:
            self.num_used_features   = num_used_features        
        self.num_ignored_feats   = num_features - num_used_features       
        
        self.paranet_layer_units = [num_parameters]+paranet_layer_units+[num_used_features]
        print( self.paranet_layer_units )
        if refnet_layer_units:
            self.refnet_layer_units  = [num_parameters+2]+refnet_layer_units+[1]
            self.use_refnet          = True
        else:
            self.refnet_layer_units  = []
            self.use_refnet          = False
        print( self.refnet_layer_units )
        self.batch_norm          = batch_norm
        self.use_base            = use_base
        if self.use_base:
            self.paranet_layer_units[-1] += 1          
        
        print("\nparanet\n")
        paranet_modules = []
        Layers = self.paranet_layer_units
        L = len( Layers )
        for i,(input_size, output_size) in enumerate( zip(Layers, Layers[1:]) ):
            print("linear io",input_size, output_size)
            paranet_modules.append( torch.nn.Linear(input_size, output_size) ) 
            if i<L-2:
                if self.batch_norm:
                    print("batch_norm size", output_size)
                    paranet_modules.append( torch.nn.LayerNorm(output_size) ) 
                print("relu")
                paranet_modules.append( torch.nn.ReLU() )                    
        self.paranet_sequential = nn.Sequential(*paranet_modules)
        
        if self.use_refnet:
            print("\nrefnet\n")
            refnet_modules = []
            Layers = self.refnet_layer_units
            L = len( Layers )
            for i,(input_size, output_size) in enumerate( zip(Layers, Layers[1:]) ):
                print("linear io",input_size, output_size)
                refnet_modules.append( torch.nn.Linear(input_size, output_size) )       
                if i<L-2:
                    if self.batch_norm:
                        print("batch_norm size", output_size)
                        refnet_modules.append( torch.nn.LayerNorm(output_size) )                    
                    print("relu")
                    refnet_modules.append( torch.nn.ReLU() ) 
            self.refnet_sequential = nn.Sequential(*refnet_modules)
        else:
            self.refnet_sequential = None
        return

    def forward(self, x_in ):
        splitter = (1,2,self.num_parameters, self.num_used_features, self.num_ignored_feats)
        ref, _ , x, entropies, _ = torch.split( x_in, splitter, 1)  
        
        x = self.paranet_sequential(x)
        
        if self.use_base:
            x, base = torch.split( x, [self.num_used_features, 1], 1)
            base = torch.squeeze(base)
        else:
            base = 0      

        x = torch.sum( x*entropies, dim=1 )
        if self.use_base:
            x = x + base
        
        if self.use_refnet:
            splitter = (1,2+self.num_parameters, self.num_used_features, self.num_ignored_feats)
            _, rx , _ , _ = torch.split( x_in, splitter, 1)          
            rx = self.refnet_sequential(rx)                      
            ref = torch.squeeze( ref+rx )
        else:
            ref = torch.squeeze(ref)
        return x + ref
    
    def para_net(self, x_in):
        splitter = (1,2,self.num_parameters, self.num_used_features, self.num_ignored_feats)
        _, _ , x, entropies, _ = torch.split( x_in, splitter, 1)  
        
        x = self.paranet_sequential(x)
        
        if self.use_base:
            x, base = torch.split( x, [self.num_used_features, 1], 1)
            base = torch.squeeze(base)
        else:
            base = 0      

        x = torch.sum( x*entropies, dim=1 )
        if self.use_base:
            x = x + base

        return x
    
    def para_A(self, x_in):
        splitter = (1,2,self.num_parameters, self.num_used_features, self.num_ignored_feats)
        _, _ , x, _, _ = torch.split( x_in, splitter, 1)  
        
        x = self.paranet_sequential(x)

        return torch.squeeze(x)
    
    def ref_net(self, x_in):
        splitter = (1,2+self.num_parameters, self.num_used_features, self.num_ignored_feats)
        ref , rx , _ , _ = torch.split( x_in, splitter, 1) 
        if self.use_refnet:
            rx = self.refnet_sequential(rx)                      
            ref = torch.squeeze( ref+rx )            
            return torch.squeeze(ref), torch.squeeze(rx)
        else:
            print("refnet not used, return CE ref + zero")
            ref = torch.squeeze(ref)      
            return torch.squeeze(ref), torch.zeros(1)
    
    

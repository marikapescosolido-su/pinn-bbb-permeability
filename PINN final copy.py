#RQ: Determine the BBB permeability (k1) from PET data using a Physics-Informed Neural Network (PINN) for the FDG tracer in the brain.

#CONTEXT
#A patient gets injected with a radioactive sugar traces (here, FDG) and then goes 
#into a PET scanner, which takes repeated snapshots monitoring the radioactivity "flow" in the brain.
#FDG is a modified glucose molecule that despite undergoing the same phosphorylation process as natural glucose,
#turning into FDG-6 (the molecule gets attached to a phosphate group), it does not continuously get broken down for energy.
#When FDG is in its FDG-6 form, due to its bew chemical structure, it cannot be further metabolized and thus gets trapped in the neural cell.

#Hence, through PET can be visualized the radioacity level in each area of the brain after an interval t.
#This "radioactivity signal" is given by the sum of:
#- The tracer progressively injected in the blood and running towards the BBB.
    #This is known as the "plasma phase" and is indicated with Cp(t). It is the input of the whole system, the total FDG injected (minimal leackage is neglected).
    #Cp(t) is known.
#- The tracer crossing the BBB but not yet chemically trapped in brain cells.
    #This is known as the "free tissue phase" and is indicated with C1(t).
    #Note that C1(t)k1 indicates the FDG that actually traces the BBB (where K1 is the rate of forward progression).
    #Note that C1(t)k2 indicates the FDG that is leaving the brain and going back to the blood (where k2 is the rate of backward progression).
#- The tracer that has been trapped (and accumulating) in the brain cells and thus is not leaving the brain anymore.
    #This is known as the "trapped tissue phase" and is indicated with C2(t).
    #Note that C2(t)k3 indicates the FDG that is being trapped in the brain cells (where k3 is the rate of trapping).
    #Note that C2(t)k4 indicates the FDG that is leaving the brain cells and going back to the free tissue phase (where k4 is the rate of untrapping). However
    #in real tissue k4 is very slow and approximated to 0.

#The explanation above is summarised by the following diagram:
#   BLOOD              FREE TISSUE           TRAPPED TISSUE
#   Cp(t)   --K1-->      C1(t)      --k3-->     C2(t)
#           <--k2--                 <--k4--

#or semplified as:
#   BLOOD              FREE TISSUE           TRAPPED TISSUE
#   Cp(t)   --K1-->      C1(t)      --k3-->     C2(t)
#           <--k2--                 


#The PET scanner cannot distinguish these three measurements, but rather sees them as combined in radioactivity
#in each area over time as a unique signal: CPET(t) for "combined PET over time".

#CPET(T)=C1(t)+C2(t) in the most simplified form


#All rates k are described as fractions of time since they are moving betwween tissue compartments already inside the BBB.
#k1 moved from blood to brain tissue, crossing the BBB, which as two physically different spaces also do not have comparable units.
#Cp(t) is plasma concentration (tracer/blood volume), whereas C1(t) is (tracer/tissue volume). The understanding of how to complete
#this unit transition comes more natural by looking at the equations describing the motion of the FDG across the different phases mentioned.

#Considering that Cp(t) is known, the system of ODEs describing the motion of FDG across the different phases refers to the unknown C1(t) and C2(t), given by:

#dC1/dt  =   k1·Cp(t)      -      k2·C1(t)       -      k3·C1(t)      +      k4·C2(t)
 
#          ───────────           ───────────           ───────────           ───────────
#         arriving from         leaving back           leaving to           arriving back
#             blood               to blood             trapped pool       from trapped pool

#dC2/dt  =   k3·C1(t)       -      k4·C2(t)
#          ───────────          ───────────
#          arriving from         leaving back
#          free compartment      to free compartment

#Among the four rates (k1,k2,k2,k4), the one describing the BBB permeability is k1. The others are descriping the flux of FDG
#across the brain tissue compartments, which are not of interest for the purpose of this work. Hence, the goal is to estimate k1 from the PET signal CPET(t), which is known.

#Purpose: Recover the unknown k1 (and k2, k3, k4)from the known CPET(t) using a Physics-Informed Neural Network (PINN).

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#Graph the progression of BBB pearmeability over time.

#GENERATE SYNTHTIC DATA TO IMITATE THE EXPERIMENTAL PET DATA
#+ add also some noise not to make the data look perfect

#Physics parameters existing only to generate "real" data.
#These are the "experimental real" values obtained by the PET scanner using appropriate physics nomenclature.

k1_real = 0.89  # Forward rate from blood to free tissue (BBB permeability)
k2_real = 0.122  # Backward rate from free tissue to blood
k3_real = 0.057  # Forward rate from free tissue to trapped tissue
k4_real = 0.0   # Backward rate from trapped tissue to free tissue

#These values were taken from the follwing paper: "Tomographic measurement of local cerebral glucose metabolic rate 
#in humans with (F-18)2-fluoro-2-deoxy-D-glucose: validation of method"
#These values are unknown to the scientist when performing the PINN analysis.

#The Cp(t) value is known.
#In fact the PET raw output is a 4D image, meaning a 3D space, changing over time, t. If you 
#select a single voxel, you can see the time evolution of the radioactivity in that voxel, which is the CPET(t) value,
#the combined PET signal, mentioned previously.
#On the other hand, the Cp(t) is the plasma concentration of the FDG tracer in the blood (the injected sugar and input of the system).
#In a medical context the Cp(t) is obtained through periodic blood sampling, which is invasive and not always feasible. Or, throgh an image-derived input function (IDIF) from the PET images themselves, 
# which is less invasive but more complex and less accurate. The latter would meen having in the PET scan also a large blood vessel (such as the aorta or a heart chamber) visible and selecting a vexel 
# in them, treating its CPET(t) as the Cp(t) of the system. However, this is not always possible, and even when it is, the IDIF is not very accurate.
#Usually the first option (blood sampling) is adopted in combination with the Feng model.
#The Feng formula is a curve-fitting model that describes the time course of the plasma input function (Cp(t)) for FDG in humans. It is based on a sum of exponentials and is widely used in PET studies to model the plasma input function.
#Using the Feng model allows to transform those scattered blood samples data points (or IDIF), into a smooth mathematical function that can be used in the ODEs describing the FDG motion across the brain tissue compartments.
#Since for this synthetic data generation we are not interested in the Cp(t) accuracy, we will use a simple function to generate it, which is not realistic but serves the purpose of this work.
#In fact, it could be possible to use examples of sample blood measurements and then apply the Feng model,
#but given the complexity of the 7-parameter Feng model, it is not necessary for the purpose of this work, which wants to focus on something different (e.g. BBB permeability).

#Here, to generate a plausible known fake Cp(t) curve (to then generate CPET(t) dta through the ODEs) we will use the following simple single exponential equation:
#Cp(t)= Cp0 * exp(-t/tau
#where Cp0 is the initial plasma concentration of the tracer, and tau is a time constant that controls how quickly the concentration decays over time.
#Let us assume:
Cp0 = 3.0      # peak plasma concentration right after injection
tau_p = 1.5    # how fast it clears, in minutes

def Cp(t):
    return Cp0 * np.exp(-t/tau_p)

#At this point we can generate the synthetic "experimental" data for C1(t) and C2(t) by solving the ODEs numerically using the known Cp(t) and the real k values.
#From them (Cp(t),C1(t),C2(t)) we will calculate the CPET(t) signal and add some noise to simulate experimental measurements.
#So, first step, calculate the "true solution", or "un-noisy" CPET(t) signal (CPET_true(t)), by solving the ODEs numerically. Then, add some noise to simulate experimental measurements (CPET_exp(t))
#The latter is what would actually be handed to the PINN if I would be using real PET data, from which will then try to recover the unknown k1 value.

#To calculate CPET_true(t), let us calculate C1(t) and C2(t) since CPET(T)=C1(t)+C2(t).
#I mentioned previously that to calculate Cp0 could have been used also the selection of CPET(t) for a major blood vassel.
#However, if on one hand is hard to get a voxel with only a major blood vessel, it is hard not to get smaller blood vassels (capillaries) in the voxel. 
#This means that in a PET voxel there can never be only pure tissue, but also vassels.
#Why does this matter since the scanner only sees FDG? 
#It matters because the scanner detects radioactive decay and the scanner cannot tell if the decay is happening because of a FDG decaying in
#a brain cell or in the blood passing through the vassel (since the FDG was injected in the blood first, that now is circulating across the body)
#This means that when you select a voxel, you select both brain tissue and caplillaries, meaning that you are not measuring the FDG flux only across the brain tissue, but
#also across the blood (which is not of medical interest). This is why, to make CPET(t) accurate, a fixed paramter Vb (Vassel blood) should be included in the CPET(T) equation.
#Hence, CPET(t) equation would become:

#CPET(t)= Vb*Cp(t)+(1-Vb)*(C1(t)+C2(t))
#where can be given a value to Vb of:
Vb=0.05 #since it usually sits between 0.03 and 0.05


#NOW CALCUALTE C1 AND C2


# Generate some time points
t_min, t_max = 0.0, 60.0 #since a usual FDG brain scan runs for about an hour.
N_data = 10
t_data = np.linspace(t_min, t_max, N_data)
#generates a sequence of evenly spaced numbers over a specified 
# interval, controlled primarily by a start value, a stop value, 
# and the total number of points.




#Generate synthetic "experimental" heights with noise
np.random.seed(0)  # For reproducibility (repeatable output)
#It is a starting number used to initialize a pseudo-random number generator
# Because NN weights are initialized randomly before training begins.
# Without seed: If your model performs well today, you might run it tomorrow 
# and get worse results just because of a different random starting point.
# With seed: You can reproduce your exact metrics, training loss curves, 
# and final model weights every single run.
noise_level = 0.7
h_data_exact = true_solution(t_data)
h_data_noisy = h_data_exact + noise_level * np.random.randn(N_data)
#thus, the height "loss" depends on the random number multiplied by the noise.

# CONVERT THE TIME DATA INTO A TENSOR
# Convert to PyTorch tensors
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1) #input tensor
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1, 1) #output predicted tensor

#DEFINE MODEL

#This section defines the NN structure:
#it is sequentially adding 2 hidden layers with 20 neurons (nodes).
#This Python code defines and creates a Multi-Layer Perceptron (MLP)
class PINN(nn.Module):
    def __init__(self, n_hidden=20):
        super(PINN, self).__init__()
        # A simple MLP with 2 hidden layers
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, t):
        """
        Forward pass: input shape (batch_size, 1) -> output shape (batch_size, 1)
        """
        return self.net(t)

# Instantiate the model
model = PINN(n_hidden=20)



#NOW YOU NEED AN AUTOMATIC DIFFERENTIATIOR
#If you want a PINN you need to calculate dh/dt
def derivative(y, x):
    # Compute the derivative of y with respect to x using autograd from torch
    # y, x must be tensors with requires_grad=True for x
    return torch.autograd.grad(
        y, x, 
        grad_outputs=torch.ones_like(y), 
        create_graph=True
    )[0]

# DEFINE THE LOSS COMPONENTS (PINN)

#We have:
# (1) Data loss (fit noisy data)
# (2) ODE loss: dh/dt = v0-g*t
# (3) Initial condition loss: h(0) = h0

def physics_loss(model, t):
    # Compare d(h_pred)/dt with the known expression (v0-gt)
    # t must have requires_grad=True for autograd to work
    t.requires_grad_(True)

    h_pred = model(t) #Make NN model predict the height given the value of t.
    dh_dt_pred = derivative(h_pred, t) #The value just precited is put into the deivative to calculate the predicted dh/dt.
    #Derivative function was the automatic differentuator function.

    #For each t, physics says dh/dt = v0 - g*t
    dh_dt_true = v0 - g*t #this is the exact dh/dt given by physics.

    loss_ode = torch.mean((dh_dt_pred - dh_dt_true)**2) #loss fucntion as MSE.
    return loss_ode

def initial_condition_loss(model):
    # Compare h_pred(0) with h0, evaluate at t=0
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    h_pred_0 = model(t0) 
    return (h_pred_0 - h0).pow(2).mean()
# h0 is exact and was defined at the beginning among the parameters.
# It refers to the integration constant when deriving.

def data_loss(model, t_data, h_data):
    #MSE between precidted h(t_i) and noisy measurements h_data
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data)**2)
#this is simply the difference between the h predicted and the h given directly by the experiment.

# TRAINING SETUP

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#where lr is the learning rate. This controls the step size taken during each update. 
#At 0.01, you are telling it to take moderately large steps to learn quickly: 0.001 is the most common baseline default across deep learning.
#This selects Adam, one of the most popular optimization algorithms in deep learning.
#model.parameters(): This hands all the trainable weights and biases in your neural 
#network over to the optimizer so it knows what to adjust.

#Hyperparameters for weighting the loss terms
#WEIGHTS
lambda_data = 1.0
lambda_ode = 1.0
lambda_ic = 1.0

#can be written manually for a baseline attempt, but then you can write
# a softadapt/self-adaptive PINN.

#For logging
num_epochs = 2000 # (number of iterations to train the model).
print_every = 200 #printing once in every 200 iterations.

#Num epochs: you usually set a high epoch count and let a technique called Early Stopping decide when to halt.
#VALIDATION LOSS:
#As long as validation loss is decreasing, the model is learning.
#When validation loss stops improving and begins rising while training loss keeps dropping, your model is just memorizing the training data.
#You set a "patience" threshold (e.g., 10–20 epochs). If validation loss doesn't improve for that many epochs, stop automatically and restore the best weights.

# TRAINING LOOP

model.train() #set the model to training mode
for epoch in range(num_epochs):
    optimizer.zero_grad() #clear the gradients from the previous step

    # Compute losses
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode = physics_loss(model, t_data_tensor)
    l_ic = initial_condition_loss(model)

    #Combine losses with weights
    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

    # Backpropagation: central mathematical algorithm that allows a neural network to learn from its errors.
    loss.backward() #compute gradients
    optimizer.step() #update weights

    #Backpropagation works out who in the network was responsible for a mistake, and by how much.

    # Logging - Print progress
    if (epoch+1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Total Loss = {loss.item():.6f}, "
              f"Data Loss = {l_data.item():.6f}, "
              f"ODE Loss = {l_ode.item():.6f}, "
              f"IC Loss = {l_ic.item():.6f}")


# TWO-STEP CYCLE:
#Forward Pass: Input data flows forward through the network layers to produce a prediction, and the loss function 
# computes a single number saying how wrong that prediction was.
#Backward Pass (Backpropagation): The error flows backward through the network. Using the calculus chain rule, 
# PyTorch calculates the gradient—the direction and magnitude to tweak each weight to reduce the loss—starting 
# from the output layer back to the input.

#By plotting, see how good the model made the precitions.

#EVALUATE TRAINED MODEL

model.eval()
t_plot = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
#This corresponds to the x axis of the graph.
t_plot_tensor = torch.tensor(t_plot, requires_grad=True)
h_pred_plot = model(t_plot_tensor).detach().numpy()

# True solution (for comparison)
h_true_plot = true_solution(t_plot)

# Plot results
plt.figure(figsize=(8, 5))
#Three plots: noisy-laboratory-experiment data, exact physics solution, and PINN prediction.
plt.scatter(t_data, h_data_noisy, color='red', label='Noisy Data')
plt.plot(t_plot, h_true_plot, 'k--', label='Exact Solution')
plt.plot(t_plot, h_pred_plot, 'b', label='PINN Prediction')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.legend()
plt.title('PINN for Ball Trajectory')
plt.grid(True)
plt.show()
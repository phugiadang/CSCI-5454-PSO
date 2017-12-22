% Ref: M. Alam, "Codes in MATLAB for Training Artificial Neural Network using Particle Swarm Optimization", 2016.
clc 
tic
close all
clear all
rng default

% Prepare data
B = randi([0,1],1000,1);
A = randi([0,1],1000,1);
C = createData(A,B);
input = [A,B];
target = C.';
inputs=input';
targets=target';

% Setup network
inputDim=length(inputs(:,1));
outputDim=length(targets(:,1));
hiddenSize=3;
net=feedforwardnet(hiddenSize);
net=configure(net,inputs,targets);
totalVariable=inputDim*hiddenSize+hiddenSize+hiddenSize+outputDim;
for j=1:totalVariable
    lowbound(1,j)=-2.5;
    upbound(1,j)=2.5;
end
population=10;
for i=1:population
    for j=1:totalVariable
        xx(i,j)=lowbound(1,j)+rand*(upbound(1,j)-lowbound(1,j));
    end
end


fun=@(x) objectiveFunc(x,hiddenSize,inputDim,outputDim,net,inputs,targets);
x0=xx;

% pso initialization
x=x0;       % initial position 
v=0.1*x0;   % initial velocity
for i=1:population 
    f0(i,1)=fun(x0(i,:));
end
[fmin0,index0]=min(f0);
pbest=x0;               % initial pbest 
gbest=x0(index0,:);     % initial gbest

% pso algorithm
c1=1.0; c2=2.0; 
iter=1; maxite=1000;

while iter<=50
    % particles velocity updates
    w=0.1+rand*0.4;
    for i=1:population 
        for j=1:totalVariable 
            v(i,j)=w*v(i,j)+c1*rand*(pbest(i,j)-x(i,j))+c2*rand*(gbest(1,j)-x(i,j)); 
        end
    end
    % particles position updates
    for i=1:population   
        for j=1:totalVariable 
            x(i,j)=x(i,j)+v(i,j); 
        end
    end
    % checking boundary violations 
    for i=1:population 
        for j=1:totalVariable   
            if x(i,j)<lowbound(j)
                x(i,j)=lowbound(j);
            elseif x(i,j)>upbound(j)
                x(i,j)=upbound(j); 
            end
        end
    end
    % evaluating fitness 
    for i=1:population    
        f(i,1)=fun(x(i,:)); 
    end
    % updating pbest and fitness 
    for i=1:population  
        if f(i,1)<f0(i,1) 
            pbest(i,:)=x(i,:); 
            f0(i,1)=f(i,1);  
        end
    end
    [fmin,index]=min(f0);   % finding out the best particle
    ffmin(iter)=fmin;    % storing best fitness  
    ffiter=iter;         % storing iteration count  
    % updating gbest and best fitness
    if fmin<fmin0       
        gbest=pbest(index,:); 
        fmin0=fmin;  
    end
    % displaying iterative results
    if iter==1  
        disp(sprintf('Iteration    Best particle    Objective fun'));
    end
    disp(sprintf('%8g  %8g          %8.4f',iter,index,fmin0)); 
    iter=iter+1; 
end

xo=gbest;
fval=fun(xo);
xbest=xo; 
ybest=fun(xo); 
    
toc 
% Final neural network model
disp('Final nn model is net_f')
net_f = feedforwardnet(hiddenSize);
net_f=configure(net_f,inputs,targets); 
[a,b]=min(ybest); 
xo=xbest(b,:);
k=0;
for i=1:hiddenSize 
    for j=1:inputDim 
        k=k+1;
        xi(i,j)=xo(k); 
    end
end
for i=1:hiddenSize 
    k=k+1;
    xl(i)=xo(k);
    xb1(i,1)=xo(k+hiddenSize);
end
for i=1:outputDim
    k=k+1; 
    xb2(i,1)=xo(k);
end

% Weights and Bias
net_f.iw{1,1}=xi;
net_f.lw{2,1}=xl;
net_f.b{1,1}=xb1;
net_f.b{2,1}=xb2; 

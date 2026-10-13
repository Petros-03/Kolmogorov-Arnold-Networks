#Απαλλακτικη Εργασια Αριθμητικες Μεθοδοι
#splines και KANs

library(torch)

torch_manual_seed(42)


# Συνάρτηση για B-splines βάσει Cox-de Boor

cox_de_boor<- function(x,knots,i,k){
  if (k==0){
    cond<-(x>=knots[i])&(x<knots[i+1]) #οι συνθήκες της βάσης αναδρομής
    return(torch_where(cond,torch_ones_like(x),torch_zeros_like(x))) #οι τιμές, αν το x ανήκει στο διάστημα το cond τότε πα΄ρνει τιμή 1, αν όχ 0. το torch_where δουλεύει σαν ifelse
  } else {
    denom1<-knots[i+k]-knots[i]
    term1<-if(denom1>0) {
      ((x-knots[i])/denom1)*cox_de_boor(x,knots,i,k-1)
    } else {
      torch_zeros_like(x)
    }
    denom2<-knots[i+k+1]-knots[i+1]
    term2<-if (denom2>0) {
      ((knots[i+k+1]-x)/denom2)*cox_de_boor(x,knots,i+1,k-1)
    } else {
      torch_zeros_like(x)
    }
    return(term1+term2)
  }
}



#Δημιουργα KAN Layer
KANLayer<-nn_module(
  "KANLayer",
  
  initialize=function(inputs,outputs,grid_min=-1,grid_max=1,grid_size=5,k=3){
    self$inputs<-inputs
    self$outputs<-outputs
    self$k<-k
    
    step<-(grid_max-grid_min)/grid_size
    self$knots<-seq(grid_min-k*step,grid_max+k*step,by=step) #χωριζουμε το δαστημα σε κομβους
    B_num<-grid_size+k #οριζουμε το αρθμο των B-splne που θα χρεαστουμε που είναι p+k
    
    self$c<-nn_parameter(torch_randn(outputs,inputs,B_num)*0.1)  #οι συντελεστες των B-splines
    #το nn_parameter σημαινε πως η μεταβλητη δεν ειναι σταθερη κα πως αλλαζει
    #το torch_randn δημουργεί ένα πίνακα διαστάσεψν outputs x inputs επειδή τοσα splnes θα χρειαστούμε
    self$w_b<-nn_parameter(torch_randn(outputs,inputs)*0.1) #τα βάρη του SiLU
    self$w_s<-nn_parameter(torch_ones(outputs,inputs))  #σε περ΄πτωση που θελουμε να αφαιρέσουμε λη να ευξήσουμε την επρροή ενος spline
  },
  
  forward=function(x) {
    B_num<-dim(self$c)[3]
    
    B_list<-lapply(1:B_num,function(i) {
      cox_de_boor(x,self$knots,i,self$k)
    }) #δημιουργουμε τα b-splines
    B<-torch_stack(B_list,dim=3)
    
    spline<-torch_einsum("bim,oim->boi",list(B,self$c)) #πολ/νται τα c με τα b-splines και δημιουργουνται τα splines για καθε ακμη
    
    silu<-x/(1+torch_exp(-x)) 
    w_silu<-torch_einsum("oi,bi->boi",list(self$w_b,silu))  #πολ/νται τα βαρη με την συναρτηση SiLU
    
    phi<-w_silu+torch_einsum("oi,boi->boi",list(self$w_s,spline))  #ο συνδιασμος της SiLU κα του splne της καθε ακμης
    
    out<-torch_sum(phi,dim=3)
    
    return(out)
  }
)

#ρυθμζόμενο δίκτυο KAN 2 επιπεδων
KAN_Network<-nn_module(
  "KAN_Network",
  
  initialize=function(inputs,hidden_nodes=8,grid_size=5,k=3) {
    self$layer1<-KANLayer(inputs=inputs,outputs=hidden_nodes,grid_size=grid_size,k=k)
    self$layer2<-KANLayer(inputs=hidden_nodes,outputs=1,grid_size=grid_size, k=k)
  },
  
  forward = function(x) {
    out1<-self$layer1(x)
    out2<-self$layer2(out1)
    return(out2)
  }
)


#προσομοωμενα δεδομενα

#δημιουργ΄΄ια δεδομένων κα συνάρτησης y=sin(π*x1)+x2^2
set.seed(42)
n_samples<-1000

#παιρνουμς τυχαιους αρθμους
x1<-runif(n_samples,min=-1,max=1)
x2<-runif(n_samples,min=-1,max=1)

#βρισκουμε τιμες του y βάσει των x1 και x2
y<-sin(pi*x1)+(x2^2)

#μετατροπη σε tensors
X<-torch_tensor(cbind(x1,x2),dtype=torch_float())
Y<-torch_tensor(matrix(y,ncol=1),dtype=torch_float())

#οριζουμε τις παραμετρους του δικτυου
model1<-KAN_Network(inputs=2,hidden_nodes=3,grid_size=8,k=3)
optimizer<-optim_adam(model1$parameters,lr=0.05)
mse<-nnf_mse_loss

#εκπα΄δευση μοντέλου
n<-400
error_history<-numeric(n)

for (epoch in 1:n) {
  optimizer$zero_grad()
  predictions<-model1(X)
  error<-mse(predictions, Y)
  error_history[epoch]<-as.numeric(error)
  error$backward()
  optimizer$step()
  
  #καταγραφη του σφαλματος
  if (epoch%%50==0) {
    cat(sprintf("Εποχή: %d | Σφάλμα (MSE): %.4f\n",epoch,error_history[epoch]))
  }
}

#απεκονση του σφαλματος
plot(1:n,error_history,type="l",col="red",lwd=2,
     main="Εκπαίδευση Συμβολικής Παλινδρόμησης",
     xlab="Εποχές",ylab="Σφάλμα MSE")


#οπτικοποιουμε την καμπυλη για καθε μεταβλητη που εμαθε το δκτυο 
model1$eval()

# δημιουργουμε μηδενικο δανυσμα το οποο αντκαταστουμε στην θεση ενος x προκειμενου να δουμε τα αποτελεσματα του μη μηδενιου x
x_scan <- seq(-1, 1, length.out = 100)
zeros <- rep(0, 100)

#δοκιμαζουμε ως προς x1 δηλαδη το x2 ειναι το μηδενικο διανυσμα
X1 <- torch_tensor(cbind(x_scan, zeros), dtype = torch_float())
pred_X1 <- as.numeric(model1(X1))

# το ιδιο για το x2
X2 <- torch_tensor(cbind(zeros, x_scan), dtype = torch_float())
pred_X2 <- as.numeric(model1(X2))


#σχεδιαζουμε τις καμπυλες τις κάθε μεταβλητης

#για x1 πρεπει να μοαζειι με ημιτονο
plot(x_scan, pred_X1, type = "l", col = "blue", lwd = 3,
     main = "Προβλεπόμενη vs πραγματική συνάρτηση της x1 sin(π*x1)", xlab = "Μεταβλητή x1", ylab = "Σήμα του Δικτύου")
lines(x_scan, sin(pi * x_scan), col = "gray", lwd = 2, lty = 2) # Η πραγματική εξίσωση
legend("topleft", legend = c("Spline Δικτύου", "Πραγματικό sin"), col = c("blue", "gray"), lty = c(1, 2))

#για x2 πρεπει να μοιαζει με παρεμβολη
plot(x_scan, pred_X2, type = "l", col = "darkgreen", lwd = 3,
     main = "Προβλεπόμενη vs πραγματική συνάρτηση της x2 x2^2", xlab = "Μεταβλητή x2", ylab = "Σήμα του Δικτύου")
lines(x_scan, x_scan^2, col = "gray", lwd = 2, lty = 2) # Η πραγματική εξίσωση
legend("top", legend = c("Spline Δικτύου", "Πραγματικό x^2"), col = c("darkgreen", "gray"), lty = c(1, 2))













#############################################

#πραγματικα δεδομενα

rdata<-read.csv("Housing.csv")
rdata<-na.omit(rdata)

# Μετατροπή στηλών στους κατάλληλους τύπους
rdata$price<-as.numeric(rdata$price)
rdata$area<-as.numeric(rdata$area)
rdata$bedrooms<-as.numeric(rdata$bedrooms)
rdata$bathrooms<-as.numeric(rdata$bathrooms)
rdata$stories<-as.numeric(rdata$stories)
rdata$mainroad<-as.factor(rdata$mainroad)
rdata$guestroom<-as.factor(rdata$guestroom)
rdata$basement<-as.factor(rdata$basement)
rdata$hotwaterheating<-as.factor(rdata$hotwaterheating)
rdata$airconditioning<-as.factor(rdata$airconditioning)
rdata$parking<-as.numeric(rdata$parking)
rdata$prefarea<-as.factor(rdata$prefarea)
rdata$furnishingstatus<-as.factor(rdata$furnishingstatus)

price<-rdata$price
housing_data<-rdata[, -1] # Αφαιρούμε τη στήλη price από τα X

# μετατροπη κατηγορκων σε διτμες 0 και 1
rdata<-model.matrix(~.- 1,data=housing_data)
num_data<-ncol(rdata)

#κανονικοποηση των x στο [-1,1]
scale_min_max<-function(x) {2*((x-min(x))/(max(x)-min(x)))-1}

scaled_data<-matrix(0,nrow=nrow(rdata),ncol=num_data)
mins<-apply(rdata,2,min)
maxs<-apply(rdata,2,max)

for(i in 1:num_data) {
  if(maxs[i]>mins[i]) {
    scaled_data[,i]<-scale_min_max(rdata[,i])
  }
}

#κανονικοποιηση του Y
Y_scaled<-scale_min_max(price)
Y<-matrix(Y_scaled,ncol=1)

#μετατροπη σε tensors
scaled_data_tensor<-torch_tensor(scaled_data,dtype=torch_float())
Y<-torch_tensor(Y,dtype=torch_float())


n<-300
mse<-nnf_mse_loss


#εκπαιδευση ΚΑΝ και καταγραφη σφαλματος
kan_error<-numeric(n)

model2<-KAN_Network(inputs=num_data,hidden_nodes=8)
optimizer_kan<-optim_adam(model2$parameters,lr=0.05)

for (epoch in 1:n) {
  optimizer_kan$zero_grad()
  predictions_kan<-model2(scaled_data_tensor)
  error_kan<-mse(predictions_kan, Y)
  error_kan$backward()
  optimizer_kan$step()
  
  kan_error[epoch]<-error_kan$item()
  
  if (epoch%%50==0) {
    cat(sprintf("KAN | Εποχή: %3d | Σφάλμα (MSE): %.6f\n",epoch,kan_error[epoch]))
  }
}


#εκπαιδευση MLP και καταγραφη σφαλματος
mlp_error<-numeric(n)

MLP_Network<-nn_sequential(
  nn_linear(num_data, 8),
  nn_relu(),                  # Κλασική συνάρτηση ενεργοποίησης ReLU
  nn_linear(8, 1)
)

mlp<-MLP_Network
optimizer_mlp<-optim_adam(mlp$parameters, lr = 0.05)

for (epoch in 1:n) {
  optimizer_mlp$zero_grad()
  predictions_mlp<-mlp(scaled_data_tensor)
  error_mlp<-mse(predictions_mlp,Y)
  error_mlp$backward()
  optimizer_mlp$step()
  
  mlp_error[epoch]<-error_mlp$item()
  
  if (epoch%%50==0) {
    cat(sprintf("MLP | Εποχή: %3d | Σφάλμα (MSE): %.6f\n",epoch,mlp_error[epoch]))
  }
}



#συγκριση σφαλματων
max_loss<-max(c(kan_error,mlp_error))

#σφαλμα KAN
plot(1:n,kan_error,type="l",col="blue",lwd=2,
     ylim=c(0,max_loss),
     xlab="Εποχές (Epochs)", 
     ylab="Σφάλμα MSE (Κανονικοποιημένο)",
     main="Σύγκριση Σφάλματος KAN vs Κλασικό MLP")

#σφαλμα MLP
lines(1:n,mlp_error,col="red",lwd=2,lty=2)


legend("topright",legend=c("KAN (B-Splines)","MLP (ReLU)"), 
       col=c("blue","red"),lwd=2,lty=c(1,2))

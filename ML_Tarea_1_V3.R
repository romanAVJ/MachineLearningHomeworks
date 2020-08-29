usePackage <- function(p) 
{
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE, repos = "https://cran.itam.mx/")
  require(p, character.only = TRUE)
}

#igual usar solo las librerias que necesitamos
usePackage('dplyr')
usePackage('tidyverse')
usePackage('MASS')

smp_size <- floor(0.8 * nrow(iris))
set.seed(123)
train_ind <- sample(seq_len(nrow(iris)), size = smp_size)

train_data <- data.frame(iris[train_ind, ])
test_data <- data.frame(iris[-train_ind, ]) %>% dplyr::select(1:4)



apriori<-function(train_data_frame, Column_Class, Class){
  
  funct_df<-train_data[Column_Class] %>% 
    rename(Column_1=Column_Class)
  n_1<-funct_df %>%
    filter(Column_1==Class) %>% 
    count()
  n_2<-funct_df %>% 
    count()
  return(n_1/n_2)
  
}

media<-function(train_data_frame, Column_Class, Class,Column_Ind_Var){
  
  funct_df<-train_data_frame[c(Column_Ind_Var,Column_Class)] %>% 
    rename(Column_1=Column_Class) %>% 
    filter(Column_1==Class) %>% 
    dplyr::select(Column_Ind_Var) %>% 
    summarise_if(is.numeric, mean)
  return(funct_df)
}

var.covar<-function(train_data_frame, Column_Class, Class,Column_Ind_Var){
  
  funct_df<-train_data_frame[c(Column_Ind_Var,Column_Class)] %>% 
    rename(Column_1=Column_Class) %>% 
    #filter(Column_1==Class) %>% (aquí en la tarea menciona filtrar por clase pero en el libro menciona que es de todas las clases esa es mi duda)
    dplyr::select(Column_Ind_Var)
  return(cov(funct_df))
}

delta<-function(func_test_data, func_apriori, func_mean, func_var){
  x<-as.matrix(func_test_data)
  f_apri <- as.vector(func_apriori)
  f_mean <- as.matrix(func_mean)
  f_var<-solve(as.matrix(func_var))
  return ((x%*%f_var%*%t(f_mean))-as.numeric((((f_mean %*% f_var)%*%t(f_mean))/2)+log(f_apri)))
  
}


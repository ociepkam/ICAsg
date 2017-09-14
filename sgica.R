Rcpp::sourceCpp('cpp_to_R//icasg.cpp')

sgICA <- function(X) {
  return(ica(X, dim(X)[1], dim(X)[2]));
}